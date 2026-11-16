"""Locked, atomic, bullet-deduplicating Markdown memory writer for Kesoku (`MemoryWriter`)."""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import logging
import os
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kesoku.agent.memory_store import (
    bump_memory_index_version,
    default_pinned_cache,
    parse_memory_frontmatter,
    resolve_shared_profile_path,
    sync_role_memories,
)
from kesoku.db.manager import DatabaseManager

logger = logging.getLogger(__name__)

HISTORY_SECTION_NAME = "历史"
DEFAULT_SECTION_NAME = "记录"

_BULLET_LINE_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
_H2_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_TAG_TOKEN_RE = re.compile(r"(?:^|\s)(#[a-zA-Z0-9_:\-\u4e00-\u9fff]+)")
_TIME_TAG_RE = re.compile(r"^#time:(\d{4}-\d{2}-\d{2})$")
_SESSION_TAG_RE = re.compile(r"^#session:([a-zA-Z0-9_-]+)$")
_CJK_SEQ_RE = re.compile(r"[\u3005-\u3007\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_ASCII_WORD_RE = re.compile(r"[a-z0-9_]+")
_PUNCT_WS_RE = re.compile(r"[\s,，.。!！?？;；:：、()（）\[\]【】\"'“”‘’\-—_]+")


@dataclass
class MemoryWriteResult:
    """Summary of operations applied by `MemoryWriter.apply_ops`."""

    ops_applied: int = 0
    ops_dropped: int = 0
    touched_files: list[str] = field(default_factory=list)
    files_over_cap: list[str] = field(default_factory=list)
    reapplied_after_conflict: list[str] = field(default_factory=list)


@contextlib.contextmanager
def with_memory_lock(memories_dir: str) -> Iterator[None]:
    """Acquire an exclusive `fcntl.flock` on `$MEMORY_DIR/.memory.lock` (run on worker threads only)."""
    abs_dir = os.path.abspath(memories_dir)
    os.makedirs(abs_dir, exist_ok=True)
    lock_path = os.path.join(abs_dir, ".memory.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def atomic_write_text(target_path: str, content: str) -> int:
    """Atomically write UTF-8 `content` via `tmp + fsync + os.replace`, returning byte size."""
    abs_target = os.path.abspath(target_path)
    parent = os.path.dirname(abs_target)
    os.makedirs(parent, exist_ok=True)
    base = os.path.basename(abs_target)
    tmp_path = os.path.join(parent, f".{base}.{os.getpid()}.{threading.get_ident()}.tmp")
    encoded = content.encode("utf-8")
    try:
        with open(tmp_path, "wb") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, abs_target)
    except Exception:
        with contextlib.suppress(OSError):
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        raise
    return len(encoded)


def normalize_rel_memory_path(raw_file: str, memories_dir: str) -> str | None:
    """Validate and normalize a `$MEMORY_DIR`-relative `.md` path; return None if unsafe."""
    if not raw_file or not isinstance(raw_file, str):
        return None
    cleaned = raw_file.strip().strip("\"'")
    for prefix in ("$MEMORY_DIR/", "${MEMORY_DIR}/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    cleaned = cleaned.lstrip("/").replace("\\", "/")
    if not cleaned or cleaned.startswith(".") or "/." in cleaned:
        return None
    if not cleaned.lower().endswith(".md"):
        cleaned = f"{cleaned}.md"

    abs_mem = os.path.abspath(memories_dir)
    abs_target = os.path.abspath(os.path.join(abs_mem, cleaned))
    try:
        if os.path.commonpath([abs_mem, abs_target]) != abs_mem or abs_target == abs_mem:
            return None
    except ValueError:
        return None
    return os.path.relpath(abs_target, abs_mem).replace(os.sep, "/")


def validate_pinned_write(
    rel_path: str,
    new_content: str,
    pinned_files: Sequence[str],
) -> str | None:
    """Return an error string if `new_content` declares `visibility: PINNED` outside `pinned_files`."""
    fm, _ = parse_memory_frontmatter(new_content)
    if fm.visibility != "PINNED":
        return None
    allowed = {p.strip().lstrip("/").replace("\\", "/") for p in pinned_files if p.strip()}
    norm_rel = rel_path.strip().lstrip("/").replace("\\", "/")
    if norm_rel not in allowed:
        allowed_list = ", ".join(sorted(allowed)) or "(none)"
        return (
            f"Error: visibility: PINNED is not allowed for '{rel_path}'. "
            f"Allowed pinned files: {allowed_list}."
        )
    return None


@dataclass
class _ParsedBullet:
    """Structured representation of a single Markdown bullet item and its provenance tags."""

    core_text: str
    confidence: str | None = None
    time_date: str | None = None
    session_id: str | None = None
    superseded: bool = False
    extra_tags: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render back into `- <core_text> <tags>` format."""
        tags: list[str] = []
        if self.confidence in ("#direct", "#inferred"):
            tags.append(self.confidence)
        for t in self.extra_tags:
            if t not in tags:
                tags.append(t)
        if self.session_id:
            tags.append(f"#session:{self.session_id}")
        if self.time_date:
            tags.append(f"#time:{self.time_date}")
        if self.superseded:
            tags.append("#superseded")
        suffix = (" " + " ".join(tags)) if tags else ""
        return f"- {self.core_text.strip()}{suffix}"


def _parse_bullet_line(line: str) -> _ParsedBullet | None:
    """Parse a Markdown `- ...` bullet line into `_ParsedBullet`, or None if not a bullet."""
    m = _BULLET_LINE_RE.match(line)
    if not m:
        return None
    raw_body = m.group(1)
    tags = [tm.group(1) for tm in _TAG_TOKEN_RE.finditer(raw_body)]
    core = _TAG_TOKEN_RE.sub("", raw_body).strip()
    if not core:
        return None

    confidence: str | None = None
    time_date: str | None = None
    session_id: str | None = None
    superseded = False
    extra: list[str] = []

    for tag in tags:
        low = tag.lower()
        if low == "#direct":
            confidence = "#direct"
        elif low == "#inferred":
            if confidence != "#direct":
                confidence = "#inferred"
        elif low == "#superseded":
            superseded = True
        elif (tm := _TIME_TAG_RE.match(tag)) is not None:
            time_date = tm.group(1)
        elif (sm := _SESSION_TAG_RE.match(tag)) is not None:
            session_id = sm.group(1)
        else:
            if tag not in extra:
                extra.append(tag)

    return _ParsedBullet(
        core_text=core,
        confidence=confidence,
        time_date=time_date,
        session_id=session_id,
        superseded=superseded,
        extra_tags=extra,
    )


def _normalize_core(text: str) -> str:
    """Normalize bullet text for exact equality comparison."""
    stripped = _TAG_TOKEN_RE.sub("", text or "").strip()
    return _PUNCT_WS_RE.sub("", stripped.lower())


def _shingle_set(text: str) -> set[str]:
    """Build CJK-bigram + ASCII-word set for Jaccard similarity."""
    lower = _TAG_TOKEN_RE.sub("", text or "").lower()
    shingles: set[str] = set()
    for wm in _ASCII_WORD_RE.finditer(lower):
        shingles.add(wm.group(0))
    for cm in _CJK_SEQ_RE.finditer(lower):
        run = cm.group(0)
        if len(run) == 1:
            shingles.add(run)
        else:
            for i in range(len(run) - 1):
                shingles.add(run[i : i + 2])
    return shingles


def bullet_similarity(a: str, b: str) -> float:
    """Compute similarity `[0.0, 1.0]` between two bullet texts (1.0 on normalized equality, else Jaccard)."""
    na = _normalize_core(a)
    nb = _normalize_core(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    sa = _shingle_set(a)
    sb = _shingle_set(b)
    if not sa or not sb:
        return 0.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


# Fuzzy heading reuse: `bullet_similarity >= 0.45` or connector-free char Jaccard >= 0.60 (short CJK headings).
SECTION_FUZZY_THRESHOLD = 0.45
SECTION_CHAR_FUZZY_THRESHOLD = 0.60
_SECTION_FUZZY_EXCLUDED = frozenset({HISTORY_SECTION_NAME, "其他", DEFAULT_SECTION_NAME})
_SECTION_CONNECTOR_CHARS = frozenset("与和及的、&/ -—_")

# Slot superseding thresholds (in-place replacement instead of a second bullet).
SLOT_SIMILARITY_THRESHOLD = 0.72
_KEY_PREFIX_RE = re.compile(r"^\s*([^：:\n]{2,16}?)\s*[：:]\s*\S")
_COURSE_RE = re.compile(r"《([^》]{1,40})》")
_LESSON_RE = re.compile(r"第\s*([0-9]+|[一二三四五六七八九十百]+)\s*课")
_PROGRESS_WORD_RE = re.compile(r"完成|进度|推进|学到|学完|学习到|进行到|片段|开始学|上到|讲到")
_SEGMENT_RE = re.compile(r"片段|第\s*\d+\s*步")


def section_similarity(a: str, b: str) -> float:
    """Similarity between two `##` heading names (max of bigram Jaccard and connector-free char Jaccard)."""
    base = bullet_similarity(a, b)
    ca = {ch for ch in (a or "").lower() if not ch.isspace() and ch not in _SECTION_CONNECTOR_CHARS}
    cb = {ch for ch in (b or "").lower() if not ch.isspace() and ch not in _SECTION_CONNECTOR_CHARS}
    if not ca or not cb:
        return base
    char_j = len(ca & cb) / len(ca | cb)
    # Normalise so the char-level score maps onto the shared 0.45 threshold scale.
    scaled = char_j * (SECTION_FUZZY_THRESHOLD / SECTION_CHAR_FUZZY_THRESHOLD)
    return max(base, scaled)


def bullet_key_prefix(text: str) -> str | None:
    """Return the normalised `Key：` / `Key:` slot prefix (2..16 chars) of a bullet, or None."""
    core = _TAG_TOKEN_RE.sub("", text or "").strip()
    m = _KEY_PREFIX_RE.match(core)
    if not m:
        return None
    key = m.group(1).strip()
    if len(key) < 2 or key.isdigit() or key.lower() in {"http", "https", "ftp", "file", "mailto"}:
        return None
    if any(ch in key for ch in "，。,;；！？!?"):
        return None
    return _PUNCT_WS_RE.sub("", key.lower()) or None


_COURSE_QUALIFIER_RE = re.compile(r"新版|旧版|标准|初级|中级|高级|上册|下册|全册|第一册|第二册|第三册|第四册|\s+")
_SUBJECT_PROGRESS_RE = re.compile(r"([\u4e00-\u9fffA-Za-z]{2,8}?)(?:水平与)?(?:学习|课程)进度")


def _course_families(text: str) -> set[str]:
    """Course identities in a bullet: `book:<title stem>` for 《…》 titles, `subj:<subject>` for `<X>学习进度`."""
    s = text or ""
    families: set[str] = set()
    for m in _COURSE_RE.finditer(s):
        stem = _PUNCT_WS_RE.sub("", _COURSE_QUALIFIER_RE.sub("", m.group(1)))
        if len(stem) >= 2:
            families.add(f"book:{stem.lower()}")
    for m in _SUBJECT_PROGRESS_RE.finditer(s):
        families.add(f"subj:{m.group(1).lower()}")
    return families


def _lesson_numbers(text: str) -> set[str]:
    return {m.group(1) for m in _LESSON_RE.finditer(text)}


def is_same_course_progress(a: str, b: str) -> bool:
    """Return True when both bullets record study progress for the same course lesson (or segment run)."""
    if not (_PROGRESS_WORD_RE.search(a) and _PROGRESS_WORD_RE.search(b)):
        return False
    fa, fb = _course_families(a), _course_families(b)
    if not fa or not fb:
        return False
    if not (fa & fb):
        return False
    la, lb = _lesson_numbers(a), _lesson_numbers(b)
    if la and lb:
        return bool(la & lb)
    return bool(_SEGMENT_RE.search(a) and _SEGMENT_RE.search(b))


def is_same_memory_slot(existing: str, incoming: str) -> bool:
    """Slot rule: near-duplicate text, same `Key：` prefix, or same course-lesson progress."""
    if bullet_similarity(existing, incoming) >= SLOT_SIMILARITY_THRESHOLD:
        return True
    ka, kb = bullet_key_prefix(existing), bullet_key_prefix(incoming)
    if ka and kb and ka == kb:
        return True
    return is_same_course_progress(existing, incoming)


def strip_superseded_bullets(doc: _MemoryDocument) -> bool:
    """Drop legacy `#superseded` bullets everywhere and remove `## 历史` once empty. Returns True if changed."""
    changed = False
    kept_sections: list[_MemorySection] = []
    for sec in doc.sections:
        new_lines: list[str] = []
        for ln in sec.lines:
            pb = _parse_bullet_line(ln)
            if pb is not None and pb.superseded:
                changed = True
                continue
            new_lines.append(ln)
        sec.lines = new_lines
        if sec.name == HISTORY_SECTION_NAME and not any(ln.strip() for ln in sec.lines):
            changed = True
            continue
        kept_sections.append(sec)
    doc.sections = kept_sections
    return changed


@dataclass
class _MemorySection:
    """A `## Heading` section inside a Markdown memory file."""

    name: str
    lines: list[str] = field(default_factory=list)


@dataclass
class _MemoryDocument:
    """In-memory structured Markdown document preserving frontmatter, preamble, and `##` sections."""

    description: str
    visibility: str
    cues: list[str]
    preamble: list[str]
    sections: list[_MemorySection]

    @classmethod
    def parse(cls, raw_text: str) -> _MemoryDocument:
        """Parse raw Markdown content into a `_MemoryDocument`."""
        if not raw_text or not raw_text.strip():
            return cls(description="", visibility="DEFAULT", cues=[], preamble=[], sections=[])
        fm, body = parse_memory_frontmatter(raw_text)
        preamble: list[str] = []
        sections: list[_MemorySection] = []
        current: _MemorySection | None = None

        for line in body.splitlines():
            hm = _H2_HEADING_RE.match(line.strip())
            if hm:
                current = _MemorySection(name=hm.group(1).strip(), lines=[])
                sections.append(current)
            else:
                if current is None:
                    preamble.append(line)
                else:
                    current.lines.append(line)

        return cls(
            description=fm.description,
            visibility=fm.visibility,
            cues=list(fm.cues),
            preamble=preamble,
            sections=sections,
        )

    def get_or_create_section(self, name: str) -> _MemorySection:
        """Find section `name` (case-insensitive) or append a new `## <name>` section."""
        clean_name = name.strip().lstrip("#").strip() or DEFAULT_SECTION_NAME
        for sec in self.sections:
            if sec.name.lower() == clean_name.lower():
                return sec
        if clean_name not in _SECTION_FUZZY_EXCLUDED:
            best: _MemorySection | None = None
            best_score = 0.0
            for sec in self.sections:
                if sec.name in _SECTION_FUZZY_EXCLUDED:
                    continue
                score = section_similarity(sec.name, clean_name)
                if score >= SECTION_FUZZY_THRESHOLD and score > best_score:
                    best, best_score = sec, score
            if best is not None:
                return best
        new_sec = _MemorySection(name=clean_name, lines=[])
        # Keep `## 历史` at the bottom if it already exists.
        hist_idx = next((i for i, s in enumerate(self.sections) if s.name == HISTORY_SECTION_NAME), None)
        if hist_idx is not None and clean_name != HISTORY_SECTION_NAME:
            self.sections.insert(hist_idx, new_sec)
        else:
            self.sections.append(new_sec)
        return new_sec

    def has_content(self) -> bool:
        """Return True if any preamble or section contains non-blank lines."""
        if any(ln.strip() for ln in self.preamble):
            return True
        return any(ln.strip() for sec in self.sections for ln in sec.lines)

    def render(self, fallback_rel_path: str) -> str:
        """Serialize back into normalized Markdown with YAML frontmatter."""
        desc = " ".join((self.description or "").split()).strip()
        if not desc:
            if fallback_rel_path == "user/profile.md":
                desc = "用户核心档案（身份、健康、饮食、语言水平与长期偏好）"
            else:
                stem = os.path.splitext(os.path.basename(fallback_rel_path))[0].replace("_", " ")
                sec_names = "、".join(s.name for s in self.sections if s.lines and s.name != HISTORY_SECTION_NAME)[:80]
                desc = f"{stem}：{sec_names}" if sec_names else f"{stem} 记忆笔记"
        escaped_desc = desc.replace('"', '\\"')
        fm_lines = ["---", f'description: "{escaped_desc}"']
        if self.visibility and self.visibility != "DEFAULT":
            fm_lines.append(f"visibility: {self.visibility}")
        if self.cues:
            cues_yaml = ", ".join(f'"{c.replace(chr(34), "")}"' for c in self.cues if c)
            if cues_yaml:
                fm_lines.append(f"cues: [{cues_yaml}]")
        fm_lines.append("---")

        body_blocks: list[str] = []
        preamble_text = "\n".join(self.preamble).strip()
        if preamble_text:
            body_blocks.append(preamble_text)

        for sec in self.sections:
            non_empty_lines = [ln.rstrip() for ln in sec.lines if ln.strip()]
            if not non_empty_lines:
                continue
            body_blocks.append(f"## {sec.name}\n" + "\n".join(non_empty_lines))

        body_str = "\n\n".join(body_blocks)
        if body_str:
            return "\n".join(fm_lines) + "\n\n" + body_str + "\n"
        return "\n".join(fm_lines) + "\n"


def _normalize_tag_token(raw: str) -> str:
    """Normalize a tag string to lowercase `#...` form."""
    s = str(raw or "").strip()
    if not s:
        return ""
    return s.lower() if s.startswith("#") else f"#{s.lower()}"


def _matches_phrase(bullet: _ParsedBullet, phrase: str, sim_threshold: float = 0.60) -> bool:
    """Return True if `bullet` matches a `replaces` or `retract` target phrase."""
    norm_phrase = _normalize_core(phrase)
    norm_bullet = _normalize_core(bullet.core_text)
    if not norm_phrase or not norm_bullet:
        return False
    if norm_phrase in norm_bullet or norm_bullet in norm_phrase:
        return True
    return bullet_similarity(bullet.core_text, phrase) >= sim_threshold


class MemoryWriter:
    """Applies structured extractor ops to `$MEMORY_DIR` atomically under `flock`."""

    @classmethod
    def snapshot_files(
        cls,
        memories_dir: str,
        rel_paths: Sequence[str],
    ) -> dict[str, str]:
        """Compute lock-free SHA-256 digests of `rel_paths` prior to an LLM extractor call."""
        snaps: dict[str, str] = {}
        abs_mem = os.path.abspath(memories_dir)
        shared_profile = resolve_shared_profile_path(abs_mem)
        for rel in rel_paths:
            norm = normalize_rel_memory_path(rel, abs_mem)
            if not norm:
                continue
            abs_p = (
                shared_profile
                if (norm == "user/profile.md" and shared_profile is not None)
                else os.path.join(abs_mem, norm)
            )
            if os.path.isfile(abs_p):
                try:
                    with open(abs_p, "rb") as f:
                        snaps[norm] = hashlib.sha256(f.read()).hexdigest()
                except OSError:
                    snaps[norm] = ""
            else:
                snaps[norm] = ""
        return snaps

    @classmethod
    def apply_ops(
        cls,
        ops: Sequence[Mapping[str, Any]],
        *,
        role: str,
        memories_dir: str,
        db: DatabaseManager | None = None,
        session_id: str | None = None,
        event_date: str | None = None,
        pinned_files: Sequence[str] = ("user/profile.md",),
        rejected_tags: Sequence[str] = (),
        max_file_bytes: int = 12288,
        snapshots: Mapping[str, str] | None = None,
    ) -> MemoryWriteResult:
        """Validate, deduplicate, supersede, and atomically persist memory ops for `role`."""
        result = MemoryWriteResult()
        if not ops:
            return result

        abs_mem = os.path.abspath(memories_dir)
        allowed_pinned = {p.strip().lstrip("/").replace("\\", "/") for p in pinned_files if p.strip()}
        rejected_norm = {_normalize_tag_token(t) for t in rejected_tags if str(t).strip()}
        eff_date = event_date or datetime.date.today().isoformat()
        sid8 = (session_id or "")[:8] or None

        ops_by_file: dict[str, list[Mapping[str, Any]]] = {}
        for op in ops:
            if not isinstance(op, Mapping):
                result.ops_dropped += 1
                continue
            kind = str(op.get("op") or "").strip().lower()
            if kind not in {"upsert_bullet", "retract", "set_description"}:
                result.ops_dropped += 1
                continue

            rel_path = normalize_rel_memory_path(str(op.get("file") or ""), abs_mem)
            if rel_path is None:
                logger.warning(f"Dropping memory op targeting invalid path: {op.get('file')!r}")
                result.ops_dropped += 1
                continue

            raw_tags = op.get("tags") or []
            if isinstance(raw_tags, str):
                raw_tags = [raw_tags]
            norm_op_tags = {_normalize_tag_token(t) for t in raw_tags if str(t).strip()}
            inline_tags = {
                _normalize_tag_token(tm.group(1))
                for tm in _TAG_TOKEN_RE.finditer(str(op.get("text") or ""))
            }
            if rejected_norm & (norm_op_tags | inline_tags):
                logger.info(f"Dropping memory op on '{rel_path}' due to rejected tag in {raw_tags}")
                result.ops_dropped += 1
                continue

            if bool(op.get("pinned")) and rel_path not in allowed_pinned:
                logger.warning(
                    f"Dropping memory op declaring pinned=True on non-allow-listed file '{rel_path}'"
                )
                result.ops_dropped += 1
                continue

            ops_by_file.setdefault(rel_path, []).append(op)

        if not ops_by_file:
            return result

        shared_profile = resolve_shared_profile_path(abs_mem)
        lock_dirs = sorted(
            {
                os.path.dirname(shared_profile)
                if (rel_p == "user/profile.md" and shared_profile is not None)
                else abs_mem
                for rel_p in ops_by_file
            }
        )
        touched_pinned = False
        touched_shared_profile = False
        with contextlib.ExitStack() as stack:
            for l_dir in lock_dirs:
                stack.enter_context(with_memory_lock(l_dir))
            for rel_path, file_ops in ops_by_file.items():
                is_shared = rel_path == "user/profile.md" and shared_profile is not None
                abs_target = (
                    shared_profile
                    if (is_shared and shared_profile is not None)
                    else os.path.join(abs_mem, rel_path)
                )
                disk_raw = ""
                if os.path.isfile(abs_target):
                    try:
                        with open(abs_target, encoding="utf-8", errors="replace") as f:
                            disk_raw = f.read()
                    except OSError as e:
                        logger.error(f"Failed reading '{abs_target}' under memory lock: {e}")
                        continue

                disk_sha = hashlib.sha256(disk_raw.encode("utf-8")).hexdigest() if disk_raw else ""
                if snapshots is not None and rel_path in snapshots and snapshots[rel_path] != disk_sha:
                    result.reapplied_after_conflict.append(rel_path)
                    logger.info(
                        f"Concurrent edit detected on '{rel_path}' during extraction; "
                        f"re-applying {len(file_ops)} op(s) on fresh content."
                    )

                doc = _MemoryDocument.parse(disk_raw)
                # Legacy `#superseded` archive lines are clutter: replaced bullets are now rewritten in place.
                file_changed = strip_superseded_bullets(doc)

                for op in file_ops:
                    kind = str(op.get("op") or "").strip().lower()
                    if bool(op.get("pinned")) and rel_path in allowed_pinned:
                        doc.visibility = "PINNED"

                    if kind == "set_description":
                        new_desc = " ".join(str(op.get("description") or "").split()).strip()
                        if new_desc and new_desc != doc.description:
                            doc.description = new_desc
                            file_changed = True
                        result.ops_applied += 1
                        continue

                    if kind == "retract":
                        match_phrase = str(op.get("match") or "").strip()
                        if not match_phrase:
                            continue
                        removed_any = False
                        for sec in doc.sections:
                            if sec.name == HISTORY_SECTION_NAME:
                                continue
                            kept_lines: list[str] = []
                            for ln in sec.lines:
                                pb = _parse_bullet_line(ln)
                                if pb is not None and _matches_phrase(pb, match_phrase, sim_threshold=0.80):
                                    removed_any = True
                                    file_changed = True
                                else:
                                    kept_lines.append(ln)
                            sec.lines = kept_lines
                        if removed_any:
                            result.ops_applied += 1
                        else:
                            logger.debug(f"Retract target '{match_phrase}' not found in '{rel_path}'; no-op.")
                        continue

                    if kind == "upsert_bullet":
                        applied = cls._apply_upsert_bullet(
                            doc=doc,
                            op=op,
                            sid8=sid8,
                            eff_date=eff_date,
                        )
                        if applied:
                            file_changed = True
                        result.ops_applied += 1

                if not file_changed and not disk_raw:
                    continue
                if not doc.has_content() and not disk_raw:
                    continue

                rendered = doc.render(rel_path)
                if rendered == disk_raw:
                    continue

                written_bytes = atomic_write_text(abs_target, rendered)
                result.touched_files.append(rel_path)
                if is_shared:
                    touched_shared_profile = True
                if doc.visibility == "PINNED":
                    touched_pinned = True
                if max_file_bytes > 0 and written_bytes > max_file_bytes:
                    result.files_over_cap.append(rel_path)

        if result.touched_files:
            safe_role = role or "default"
            if db is not None:
                sync_role_memories(db, safe_role, abs_mem)
            else:
                bump_memory_index_version(safe_role)
            if touched_shared_profile:
                default_pinned_cache.invalidate()
            elif touched_pinned:
                default_pinned_cache.invalidate(safe_role)

        return result

    @classmethod
    def _build_incoming_bullet(
        cls,
        op: Mapping[str, Any],
        sid8: str | None,
        eff_date: str,
    ) -> _ParsedBullet | None:
        """Construct a normalized `_ParsedBullet` with auto `#session` and `#time` tags."""
        raw_text = str(op.get("text") or "").strip()
        if raw_text.startswith(("- ", "* ")):
            raw_text = raw_text[2:].strip()
        if not raw_text:
            return None

        parsed_inline = _parse_bullet_line(f"- {raw_text}")
        core = parsed_inline.core_text if parsed_inline else raw_text
        if not core:
            return None

        raw_tags = op.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]

        conf = parsed_inline.confidence if parsed_inline else None
        time_d = parsed_inline.time_date if parsed_inline else None
        sess_tag = parsed_inline.session_id if parsed_inline else None
        extra = list(parsed_inline.extra_tags) if parsed_inline else []

        for t in raw_tags:
            ts = str(t).strip()
            if not ts:
                continue
            if not ts.startswith("#"):
                ts = f"#{ts}"
            low = ts.lower()
            if low == "#direct":
                conf = "#direct"
            elif low == "#inferred":
                if conf != "#direct":
                    conf = "#inferred"
            elif (tm := _TIME_TAG_RE.match(ts)) is not None:
                time_d = tm.group(1)
            elif (sm := _SESSION_TAG_RE.match(ts)) is not None:
                sess_tag = sm.group(1)
            elif low != "#superseded" and ts not in extra:
                extra.append(ts)

        return _ParsedBullet(
            core_text=core,
            confidence=conf or "#inferred",
            time_date=time_d or eff_date,
            session_id=sess_tag or sid8,
            superseded=False,
            extra_tags=extra,
        )

    @classmethod
    def _apply_upsert_bullet(
        cls,
        *,
        doc: _MemoryDocument,
        op: Mapping[str, Any],
        sid8: str | None,
        eff_date: str,
    ) -> bool:
        """Apply a single `upsert_bullet` op with dedup and P0 contradiction/supersede semantics."""
        incoming = cls._build_incoming_bullet(op, sid8=sid8, eff_date=eff_date)
        if incoming is None:
            return False

        sec_name = str(op.get("section") or DEFAULT_SECTION_NAME).strip() or DEFAULT_SECTION_NAME
        target_sec = doc.get_or_create_section(sec_name)

        replaces = str(op.get("replaces") or "").strip()
        if replaces:
            other_secs = [s for s in doc.sections if s is not target_sec and s.name != HISTORY_SECTION_NAME]
            search_order = [target_sec, *other_secs]
            for sec in search_order:
                for idx, ln in enumerate(sec.lines):
                    existing = _parse_bullet_line(ln)
                    if existing is None or not _matches_phrase(existing, replaces, sim_threshold=0.60):
                        continue
                    # Contradiction rule (P0): #direct > #inferred
                    if existing.confidence == "#direct" and incoming.confidence != "#direct":
                        return False
                    # If same confidence tier and existing has a strictly newer #time, keep existing
                    if (
                        existing.confidence == incoming.confidence
                        and existing.time_date
                        and incoming.time_date
                        and incoming.time_date < existing.time_date
                    ):
                        return False

                    # Replace in place (no `## 历史` / `#superseded` archive) and purge other same-slot copies.
                    sec.lines[idx] = incoming.render()
                    cls._purge_slot_duplicates(doc, incoming, keep_sec=sec, keep_idx=idx)
                    return True
            logger.debug(f"replaces target '{replaces}' not found; treating as plain upsert.")

        # Bullet-level dedup (normalised equality or Jaccard >= 0.80) across target section first, then active sections
        dedup_order = [target_sec, *(s for s in doc.sections if s is not target_sec and s.name != HISTORY_SECTION_NAME)]
        for sec in dedup_order:
            for idx, ln in enumerate(sec.lines):
                existing = _parse_bullet_line(ln)
                if existing is None:
                    continue
                sim = bullet_similarity(existing.core_text, incoming.core_text)
                if sim < 0.80:
                    continue

                merged_conf = (
                    "#direct"
                    if (existing.confidence == "#direct" or incoming.confidence == "#direct")
                    else "#inferred"
                )
                ex_time = existing.time_date or ""
                in_time = incoming.time_date or ""
                incoming_newer_or_equal = in_time >= ex_time

                if existing.confidence == "#direct" and incoming.confidence != "#direct":
                    chosen_text = existing.core_text
                elif incoming_newer_or_equal:
                    chosen_text = incoming.core_text
                else:
                    chosen_text = existing.core_text

                merged_extra = list(existing.extra_tags)
                for tag in incoming.extra_tags:
                    if tag not in merged_extra:
                        merged_extra.append(tag)

                merged = _ParsedBullet(
                    core_text=chosen_text,
                    confidence=merged_conf,
                    time_date=max(ex_time, in_time) or None,
                    session_id=(incoming.session_id if incoming_newer_or_equal else existing.session_id)
                    or incoming.session_id
                    or existing.session_id,
                    superseded=False,
                    extra_tags=merged_extra,
                )
                rendered = merged.render()
                if sec.lines[idx] == rendered:
                    return False
                sec.lines[idx] = rendered
                return True

        # Slot superseding: same `Key：` prefix, same course-lesson progress, or Jaccard >= 0.72 -> in place.
        for sec in [target_sec, *(s for s in doc.sections if s is not target_sec and s.name != HISTORY_SECTION_NAME)]:
            for idx, ln in enumerate(sec.lines):
                existing = _parse_bullet_line(ln)
                if existing is None or existing.superseded:
                    continue
                if not is_same_memory_slot(existing.core_text, incoming.core_text):
                    continue
                if existing.confidence == "#direct" and incoming.confidence != "#direct":
                    return False
                if existing.time_date and incoming.time_date and incoming.time_date < existing.time_date:
                    return False
                merged_extra = list(incoming.extra_tags)
                for tag in existing.extra_tags:
                    if tag not in merged_extra:
                        merged_extra.append(tag)
                replacement = _ParsedBullet(
                    core_text=incoming.core_text,
                    confidence=incoming.confidence,
                    time_date=incoming.time_date,
                    session_id=incoming.session_id,
                    superseded=False,
                    extra_tags=merged_extra,
                )
                rendered = replacement.render()
                changed = sec.lines[idx] != rendered
                sec.lines[idx] = rendered
                changed = cls._purge_slot_duplicates(doc, incoming, keep_sec=sec, keep_idx=idx) or changed
                return changed

        target_sec.lines.append(incoming.render())
        return True

    @classmethod
    def _purge_slot_duplicates(
        cls,
        doc: _MemoryDocument,
        incoming: _ParsedBullet,
        *,
        keep_sec: _MemorySection,
        keep_idx: int,
    ) -> bool:
        """Remove other bullets occupying the same slot as `incoming` (keeps `keep_sec.lines[keep_idx]`)."""
        removed = False
        for sec in doc.sections:
            if sec.name == HISTORY_SECTION_NAME:
                continue
            new_lines: list[str] = []
            for idx, ln in enumerate(sec.lines):
                if sec is keep_sec and idx == keep_idx:
                    new_lines.append(ln)
                    continue
                pb = _parse_bullet_line(ln)
                if pb is not None and is_same_memory_slot(pb.core_text, incoming.core_text):
                    if pb.confidence == "#direct" and incoming.confidence != "#direct":
                        new_lines.append(ln)
                        continue
                    removed = True
                    continue
                new_lines.append(ln)
            sec.lines = new_lines
        return removed

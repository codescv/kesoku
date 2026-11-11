"""Tier 1 background episode memory extractor and role consolidator (`MemoryConsolidator`)."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import datetime
import hashlib
import importlib.resources
import json
import logging
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from kesoku.agent.context_dir import collect_context_entries
from kesoku.agent.context_policy import (
    ContextPolicy,
    infer_backfill_turn_policies,
    normalize_policy,
)
from kesoku.agent.history import segment_logical_turns
from kesoku.agent.llm import CLAUDE_THINKING_LEVELS, GEMINI_THINKING_LEVELS, BaseLLM
from kesoku.agent.memory_store import (
    bump_memory_index_version,
    default_pinned_cache,
    parse_memory_frontmatter,
    search_role_memories,
    sync_role_memories,
)
from kesoku.agent.memory_writer import (
    HISTORY_SECTION_NAME,
    MemoryWriter,
    MemoryWriteResult,
    _MemoryDocument,
    _MemorySection,
    _parse_bullet_line,
    atomic_write_text,
    bullet_similarity,
    normalize_rel_memory_path,
    with_memory_lock,
)
from kesoku.agent.session_digest import is_excluded_from_memory
from kesoku.config import KesokuConfig
from kesoku.constants import MessageRole, MessageType
from kesoku.context import KesokuContext
from kesoku.cron import cron_matches
from kesoku.db.manager import DatabaseManager
from kesoku.db.models import Message
from kesoku.utils import embedding as embedding_utils
from kesoku.utils.text import strip_injected_blocks

logger = logging.getLogger(__name__)

NEAR_CAPACITY_BYTES = 8192
CANDIDATE_BODY_MAX_CHARS = 2500
MAX_TOOL_OUTPUT_EXTRACT_CHARS = 200
MAX_EPISODE_SUMMARY_CHARS = 400
IDLE_SWEEP_INTERVAL_SECONDS = 600.0
RESCAN_INTERVAL_SECONDS = 60.0
CRON_POLL_INTERVAL_SECONDS = 30.0


def _guarded_malloc_trim() -> None:
    """Release freed glibc non-main-arena pages back to the OS on Linux; no-op elsewhere."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def resolve_extraction_thinking_level(provider: str, configured_level: str | None) -> str:
    """Map configured extraction thinking level to a valid provider level (`H4`: never None)."""
    prov = (provider or "gemini").strip().lower()
    desired = (configured_level or "minimal").strip().lower()
    if prov == "claude":
        if desired == "minimal" or desired not in CLAUDE_THINKING_LEVELS:
            return "low"
        return desired
    if desired in GEMINI_THINKING_LEVELS:
        return desired
    return "minimal"


def _slugify_section(name: str) -> str:
    """Convert a Markdown section heading into a safe filename slug."""
    cleaned = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff-]+", "_", name.strip().lower()).strip("_")
    if not cleaned:
        cleaned = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return cleaned[:40]


def _load_base_extractor_prompt() -> str:
    """Load the built-in `prompts/memory_extractor.md` rubric."""
    try:
        ref = importlib.resources.files("kesoku.agent.prompts") / "memory_extractor.md"
        return ref.read_text(encoding="utf-8").strip()
    except Exception:
        fallback = os.path.join(os.path.dirname(__file__), "prompts", "memory_extractor.md")
        with open(fallback, encoding="utf-8") as f:
            return f.read().strip()


def parse_extractor_json(raw_content: str) -> dict[str, Any]:
    """Strictly parse extractor JSON response, stripping optional markdown fences."""
    if not raw_content or not raw_content.strip():
        raise ValueError("Empty extractor LLM output")
    text = raw_content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        json_str = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError(f"No JSON object found in extractor output: {text[:120]!r}")
        json_str = text[start : end + 1]
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError("Extractor output root must be a JSON object")
    ops = data.get("ops", [])
    if ops is not None and not isinstance(ops, list):
        raise ValueError("'ops' field in extractor output must be a list")
    return data


def format_turn_for_extraction(turn: Sequence[Message], policy: ContextPolicy) -> str:
    """Render a single logical turn for `EpisodeExtractor` according to its `ContextPolicy`."""
    if policy == "isolated":
        return "[omitted — isolated context]"

    lines: list[str] = []
    if policy == "real_facts_only":
        lines.append("[scenario frame — keep only real-world facts]")

    for msg in turn:
        if msg.type == MessageType.THOUGHT or msg.sender == "Notification":
            continue
        ts_str = datetime.datetime.fromtimestamp(msg.timestamp).astimezone().strftime("%Y-%m-%d %H:%M")
        if msg.role == MessageRole.USER and msg.type == MessageType.TEXT:
            cleaned = strip_injected_blocks(msg.content or "")
            if cleaned:
                lines.append(f"[{ts_str}] USER: {cleaned}")
        elif msg.role == MessageRole.ASSISTANT and msg.type == MessageType.TEXT:
            cleaned = strip_injected_blocks(msg.content or "")
            if cleaned:
                lines.append(f"[{ts_str}] ASSISTANT: {cleaned}")
        elif msg.type == MessageType.TOOL_CALL:
            meta = msg.metadata or {}
            t_name = meta.get("tool_name") or msg.sender or "tool"
            lines.append(f"[{ts_str}] TOOL_CALL: {t_name}")
        elif msg.type == MessageType.TOOL_RESULT:
            raw_out = strip_injected_blocks(msg.content or "")
            if len(raw_out) > MAX_TOOL_OUTPUT_EXTRACT_CHARS:
                raw_out = raw_out[:MAX_TOOL_OUTPUT_EXTRACT_CHARS] + "…[truncated]"
            if raw_out:
                lines.append(f"[{ts_str}] TOOL_RESULT: {raw_out}")

    return "\n".join(lines)


@dataclass(frozen=True)
class EpisodeExtract:
    """Job extracting durable facts and session summary from unextracted completed turns."""

    role: str
    session_id: str
    reason: str = "batch"
    bypass_budget: bool = False
    force: bool = False


@dataclass(frozen=True)
class ConsolidateFile:
    """Job deduplicating, demoting, archiving, or splitting a single memory file."""

    role: str
    rel_path: str


@dataclass(frozen=True)
class ConsolidateRole:
    """Nightly or manual role-wide consolidation job."""

    role: str
    force: bool = False


@dataclass(frozen=True)
class ReindexMemories:
    """Background incremental sync of on-disk `.md` files into `memory_files`/`memory_chunks`."""

    role: str
    force: bool = False


ConsolidatorJob = EpisodeExtract | ConsolidateFile | ConsolidateRole | ReindexMemories


@dataclass
class ExtractionOutcome:
    """Detailed result of a single `EpisodeExtractor.extract_window` execution."""

    status: str
    episode_summary: str = ""
    open_threads: list[str] | None = None
    ops: list[dict[str, Any]] | None = None
    ops_applied: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    messages_processed: int = 0
    files_over_cap: list[str] | None = None


class EpisodeExtractor:
    """Builds candidate-aware extraction prompts and applies JSON memory ops."""

    def __init__(self, cfg: KesokuConfig) -> None:
        """Initialize extractor with workspace configuration."""
        self.cfg = cfg

    def _resolve_memories_dir(self, role: str) -> str:
        roles_dir = self.cfg.workspace.roles_dir
        if not os.path.isabs(roles_dir) and self.cfg.agent_working_dir:
            roles_dir = os.path.join(self.cfg.agent_working_dir, roles_dir)
        return os.path.abspath(os.path.join(roles_dir, role or "default", "memories"))

    def _load_custom_prompts(self) -> str:
        parts: list[str] = []
        for rel in self.cfg.agent.memory_extraction_prompts:
            p = rel
            if not os.path.isabs(p) and self.cfg.agent_working_dir:
                p = os.path.join(self.cfg.agent_working_dir, p)
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        parts.append(content)
                except OSError as e:
                    logger.warning(f"Failed reading custom memory_extraction_prompt '{p}': {e}")
        return "\n\n".join(parts)

    def prepare_prompt_sync(
        self,
        *,
        db: DatabaseManager,
        role: str,
        session_id: str,
        rendered_turns: Sequence[str],
        window_query_text: str,
        compaction_hints: Sequence[str] = (),
    ) -> tuple[str, dict[str, str], str]:
        """Assemble full extractor prompt and pre-LLM file SHA-256 snapshots on the worker thread."""
        mem_dir = self._resolve_memories_dir(role)
        sync_role_memories(db, role, mem_dir)

        pinned_files = list(self.cfg.agent.memory_pinned_files or ["user/profile.md"])
        snapshot_targets: list[str] = list(pinned_files)

        pinned_sections: list[str] = []
        for p_rel in pinned_files:
            norm_p = normalize_rel_memory_path(p_rel, mem_dir)
            if not norm_p:
                continue
            abs_p = os.path.join(mem_dir, norm_p)
            if os.path.isfile(abs_p):
                try:
                    with open(abs_p, encoding="utf-8", errors="replace") as f:
                        raw_p = f.read()
                    _, body_p = parse_memory_frontmatter(raw_p)
                    pinned_sections.append(f"### {norm_p} (PINNED)\n{body_p[:CANDIDATE_BODY_MAX_CHARS]}")
                except OSError:
                    pass
        pinned_text = "\n\n".join(pinned_sections) if pinned_sections else "(empty — not created yet)"

        k_cand = max(0, int(self.cfg.agent.memory_extraction_candidate_files))
        cand_matches = (
            search_role_memories(
                db,
                role,
                window_query_text or "*",
                limit=max(1, k_cand),
                min_score=0.38,
                memories_dir=mem_dir,
                include_pinned=False,
                allow_sync=False,
            )[:k_cand]
            if k_cand > 0 and window_query_text.strip()
            else []
        )
        cand_paths = {m.rel_path for m in cand_matches}
        snapshot_targets.extend(cand_paths)

        all_files = db.get_memory_files_by_role(role)
        size_by_path = {str(r["rel_path"]): int(r.get("size_bytes") or 0) for r in all_files}

        cand_blocks: list[str] = []
        for m in cand_matches:
            abs_c = os.path.join(mem_dir, m.rel_path)
            if not os.path.isfile(abs_c):
                continue
            try:
                with open(abs_c, encoding="utf-8", errors="replace") as f:
                    raw_c = f.read()
                _, body_c = parse_memory_frontmatter(raw_c)
                cap_flag = ""
                if size_by_path.get(m.rel_path, 0) > NEAR_CAPACITY_BYTES:
                    stem = os.path.splitext(m.rel_path)[0]
                    cap_flag = f" [NEAR_CAPACITY — put new subtopics in {stem}/<subtopic>.md]"
                cand_blocks.append(
                    f"### {m.rel_path} (description: {m.description}){cap_flag}\n{body_c[:CANDIDATE_BODY_MAX_CHARS]}"
                )
            except OSError:
                continue
        candidates_text = "\n\n".join(cand_blocks) if cand_blocks else "(none matched)"

        other_lines: list[str] = []
        pinned_norm_set = {normalize_rel_memory_path(p, mem_dir) for p in pinned_files}
        for row in all_files:
            rel_p = str(row["rel_path"])
            if rel_p in cand_paths or rel_p in pinned_norm_set:
                continue
            cap_note = ""
            if int(row.get("size_bytes") or 0) > NEAR_CAPACITY_BYTES:
                stem = os.path.splitext(rel_p)[0]
                cap_note = f" [NEAR_CAPACITY — put new subtopics in {stem}/<subtopic>.md]"
            other_lines.append(f"- {rel_p}: {row.get('description') or ''}{cap_note}")
        other_index_text = "\n".join(other_lines) if other_lines else "(none)"

        snapshots = MemoryWriter.snapshot_files(mem_dir, snapshot_targets)

        base_rubric = _load_base_extractor_prompt()
        custom_rubric = self._load_custom_prompts()
        hints_block = ""
        if compaction_hints:
            hints_block = "\n\n## Session Compaction Hints\n" + "\n---\n".join(compaction_hints)

        prompt_parts = [base_rubric]
        if custom_rubric:
            prompt_parts.append(f"## Workspace-Specific Extraction Rules\n{custom_rubric}")
        prompt_parts.extend(
            [
                f"## Current Pinned Profile (allow-listed pinned files: {', '.join(pinned_files)})\n{pinned_text}",
                f"## Candidate Existing Memory Files\n{candidates_text}",
                f"## Other Existing Memory Files (path & description only)\n{other_index_text}",
            ]
        )
        if hints_block:
            prompt_parts.append(hints_block.strip())
        prompt_parts.append(
            f"## Session Slice (#session:{session_id[:8]})\n" + "\n\n".join(rendered_turns) + "\n\nJSON Output:"
        )
        return "\n\n".join(prompt_parts), snapshots, mem_dir


def consolidate_file_sync(
    db: DatabaseManager,
    role: str,
    rel_path: str,
    memories_dir: str,
    *,
    max_file_bytes: int = 12288,
    pinned_max_chars: int = 2500,
) -> list[str]:
    """Deduplicate, demote over-budget PINNED bullets, archive history, or split oversized files."""
    abs_mem = os.path.abspath(memories_dir)
    norm_rel = normalize_rel_memory_path(rel_path, abs_mem)
    if not norm_rel:
        return []
    abs_target = os.path.join(abs_mem, norm_rel)
    if not os.path.isfile(abs_target):
        return []

    touched: list[str] = []
    demoted_ops: list[dict[str, Any]] = []

    with with_memory_lock(abs_mem):
        try:
            with open(abs_target, encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        except OSError:
            return []

        doc = _MemoryDocument.parse(raw_text)
        changed = False

        # 1. Intra-section bullet deduplication
        for sec in doc.sections:
            deduped_lines: list[str] = []
            seen_bullets: list[tuple[int, Any]] = []
            for ln in sec.lines:
                pb = _parse_bullet_line(ln)
                if pb is None:
                    deduped_lines.append(ln)
                    continue
                dup_pos = next(
                    (
                        pos
                        for pos, prev_pb in seen_bullets
                        if bullet_similarity(prev_pb.core_text, pb.core_text) >= 0.80
                    ),
                    None,
                )
                if dup_pos is not None:
                    changed = True
                    _, prev_pb = seen_bullets[dup_pos]
                    if pb.confidence == "#direct" and prev_pb.confidence != "#direct":
                        deduped_lines[dup_pos] = pb.render()
                        seen_bullets[dup_pos] = (dup_pos, pb)
                else:
                    pos = len(deduped_lines)
                    deduped_lines.append(pb.render())
                    seen_bullets.append((pos, pb))
            sec.lines = deduped_lines

        # 2. Pinned budget demotion: demote ## 历史 first, then #inferred (oldest #time first), then oldest #direct
        if doc.visibility == "PINNED" and pinned_max_chars > 0:
            rendered_now = doc.render(norm_rel)
            if len(rendered_now) > pinned_max_chars:
                # Move ## 历史 out first
                for sec in list(doc.sections):
                    if sec.name == HISTORY_SECTION_NAME and sec.lines:
                        for ln in sec.lines:
                            pb = _parse_bullet_line(ln)
                            if pb is not None:
                                demoted_ops.append(
                                    {
                                        "op": "upsert_bullet",
                                        "file": "topics/profile_archive.md",
                                        "pinned": False,
                                        "section": HISTORY_SECTION_NAME,
                                        "text": pb.core_text,
                                        "tags": [pb.confidence or "#inferred", *pb.extra_tags],
                                    }
                                )
                        sec.lines = []
                        changed = True

                # Collect demotion candidates sorted by (#inferred before #direct, oldest #time first)
                while len(doc.render(norm_rel)) > max(400, pinned_max_chars - 160):
                    candidates: list[tuple[int, str, _MemorySection, int, Any]] = []
                    for sec in doc.sections:
                        if sec.name == HISTORY_SECTION_NAME:
                            continue
                        for idx, ln in enumerate(sec.lines):
                            pb = _parse_bullet_line(ln)
                            if pb is None:
                                continue
                            prio = 0 if pb.confidence != "#direct" else 1
                            t_key = pb.time_date or "0000-00-00"
                            candidates.append((prio, t_key, sec, idx, pb))
                    if len(candidates) <= 1:
                        break
                    candidates.sort(key=lambda item: (item[0], item[1]))
                    _, _, chosen_sec, chosen_idx, chosen_pb = candidates[0]
                    chosen_sec.lines.pop(chosen_idx)
                    slug = _slugify_section(chosen_sec.name)
                    demoted_ops.append(
                        {
                            "op": "upsert_bullet",
                            "file": f"topics/{slug}.md",
                            "pinned": False,
                            "section": chosen_sec.name,
                            "text": chosen_pb.core_text,
                            "tags": [chosen_pb.confidence or "#inferred", *chosen_pb.extra_tags],
                        }
                    )
                    changed = True

        # 3. File byte cap (> max_file_bytes): archive ## 历史, then split largest sections into <stem>/<subtopic>.md
        rendered_check = doc.render(norm_rel)
        if max_file_bytes > 0 and len(rendered_check.encode("utf-8")) > max_file_bytes:
            stem = os.path.splitext(norm_rel)[0]
            ym = datetime.date.today().strftime("%Y-%m")

            # 3a. Archive ## 历史 if non-empty
            for sec in list(doc.sections):
                if sec.name == HISTORY_SECTION_NAME and sec.lines:
                    arch_rel = f"{stem}/archive-{ym}.md"
                    arch_abs = os.path.join(abs_mem, arch_rel)
                    arch_doc = _MemoryDocument.parse(
                        open(arch_abs, encoding="utf-8", errors="replace").read() if os.path.isfile(arch_abs) else ""
                    )
                    if not arch_doc.description:
                        arch_doc.description = f"{doc.description or stem} — 历史归档 ({ym})"
                    arch_sec = arch_doc.get_or_create_section(HISTORY_SECTION_NAME)
                    for ln in sec.lines:
                        if ln not in arch_sec.lines:
                            arch_sec.lines.append(ln)
                    atomic_write_text(arch_abs, arch_doc.render(arch_rel))
                    touched.append(arch_rel)
                    sec.lines = []
                    changed = True

            # 3b. Split largest sections into <stem>/<subtopic>.md if still over cap
            while len(doc.render(norm_rel).encode("utf-8")) > max_file_bytes:
                splittable = [
                    s
                    for s in doc.sections
                    if s.name != HISTORY_SECTION_NAME
                    and len(s.lines) > 1
                    and not (len(s.lines) == 1 and "$MEMORY_DIR/" in s.lines[0])
                ]
                if len(splittable) <= 1:
                    # Single giant section: archive oldest half of bullets
                    if splittable and len(splittable[0].lines) > 4:
                        big_sec = splittable[0]
                        half = len(big_sec.lines) // 2
                        moved_lines = big_sec.lines[:half]
                        big_sec.lines = big_sec.lines[half:]
                        arch_rel = f"{stem}/archive-{ym}.md"
                        arch_abs = os.path.join(abs_mem, arch_rel)
                        arch_doc = _MemoryDocument.parse(
                            open(arch_abs, encoding="utf-8", errors="replace").read()
                            if os.path.isfile(arch_abs)
                            else ""
                        )
                        if not arch_doc.description:
                            arch_doc.description = f"{doc.description or stem} — 归档 ({ym})"
                        arch_sec = arch_doc.get_or_create_section(big_sec.name)
                        arch_sec.lines.extend(moved_lines)
                        atomic_write_text(arch_abs, arch_doc.render(arch_rel))
                        if arch_rel not in touched:
                            touched.append(arch_rel)
                        changed = True
                    break

                largest = max(splittable, key=lambda s: sum(len(x) for x in s.lines))
                sub_slug = _slugify_section(largest.name)
                sub_rel = f"{stem}/{sub_slug}.md"
                sub_abs = os.path.join(abs_mem, sub_rel)
                sub_doc = _MemoryDocument.parse(
                    open(sub_abs, encoding="utf-8", errors="replace").read() if os.path.isfile(sub_abs) else ""
                )
                if not sub_doc.description:
                    sub_doc.description = f"{doc.description or stem} — {largest.name}"
                sub_sec = sub_doc.get_or_create_section(largest.name)
                for ln in largest.lines:
                    if ln not in sub_sec.lines:
                        sub_sec.lines.append(ln)
                atomic_write_text(sub_abs, sub_doc.render(sub_rel))
                if sub_rel not in touched:
                    touched.append(sub_rel)
                largest.lines = [f"- 详见 `$MEMORY_DIR/{sub_rel}`"]
                changed = True

        if changed:
            final_rendered = doc.render(norm_rel)
            if final_rendered != raw_text:
                atomic_write_text(abs_target, final_rendered)
                touched.append(norm_rel)

    if demoted_ops:
        res = MemoryWriter.apply_ops(
            demoted_ops,
            role=role,
            memories_dir=abs_mem,
            db=db,
            max_file_bytes=max_file_bytes,
        )
        for p in res.touched_files:
            if p not in touched:
                touched.append(p)

    if touched:
        sync_role_memories(db, role, abs_mem)
        bump_memory_index_version(role)
        default_pinned_cache.invalidate(role)
    return touched


def consolidate_role_sync(
    db: DatabaseManager,
    role: str,
    memories_dir: str,
    *,
    max_file_bytes: int = 12288,
    pinned_max_chars: int = 2500,
) -> list[str]:
    """Run role-wide deduplication, near-duplicate file merging, and oversized file splitting."""
    abs_mem = os.path.abspath(memories_dir)
    if not os.path.isdir(abs_mem):
        return []
    sync_role_memories(db, role, abs_mem)
    files = db.get_memory_files_by_role(role)
    if not files:
        return []

    touched: list[str] = []

    # Near-duplicate DEFAULT topic file check via chunk-0 embeddings (cosine >= 0.92)
    chunk_rows = [
        r
        for r in db.get_memory_chunks_for_role(role, include_pinned=False)
        if int(r["chunk_index"]) == 0 and r["chunk_embedding"]
    ]
    if len(chunk_rows) >= 2:
        vecs: list[tuple[str, int, np.ndarray]] = []
        size_map = {str(f["rel_path"]): int(f.get("size_bytes") or 0) for f in files}
        for r in chunk_rows:
            blob = r["chunk_embedding"]
            if len(blob) % 4 != 0:
                continue
            v = np.frombuffer(blob, dtype=np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 0.0:
                vecs.append((str(r["rel_path"]), size_map.get(str(r["rel_path"]), 0), v / norm))

        merged_away: set[str] = set()
        for i in range(len(vecs)):
            p_i, sz_i, v_i = vecs[i]
            if p_i in merged_away:
                continue
            for j in range(i + 1, len(vecs)):
                p_j, sz_j, v_j = vecs[j]
                if p_j in merged_away or v_i.shape != v_j.shape:
                    continue
                if float(np.dot(v_i, v_j)) >= 0.92:
                    keep_p, drop_p = (p_i, p_j) if sz_i >= sz_j else (p_j, p_i)
                    abs_drop = os.path.join(abs_mem, drop_p)
                    if os.path.isfile(abs_drop):
                        try:
                            with open(abs_drop, encoding="utf-8", errors="replace") as f:
                                drop_doc = _MemoryDocument.parse(f.read())
                            merge_ops: list[dict[str, Any]] = []
                            for sec in drop_doc.sections:
                                if sec.name == HISTORY_SECTION_NAME:
                                    continue
                                for ln in sec.lines:
                                    pb = _parse_bullet_line(ln)
                                    if pb is not None:
                                        merge_ops.append(
                                            {
                                                "op": "upsert_bullet",
                                                "file": keep_p,
                                                "pinned": False,
                                                "section": sec.name,
                                                "text": pb.core_text,
                                                "tags": [pb.confidence or "#inferred", *pb.extra_tags],
                                            }
                                        )
                            if merge_ops:
                                MemoryWriter.apply_ops(
                                    merge_ops,
                                    role=role,
                                    memories_dir=abs_mem,
                                    db=None,
                                    max_file_bytes=max_file_bytes,
                                )
                            with with_memory_lock(abs_mem):
                                os.unlink(abs_drop)
                            merged_away.add(drop_p)
                            touched.append(keep_p)
                        except OSError as e:
                            logger.warning(f"Failed merging duplicate memory file '{drop_p}' -> '{keep_p}': {e}")

    for row in db.get_memory_files_by_role(role):
        rel_p = str(row["rel_path"])
        vis = str(row.get("visibility") or "DEFAULT")
        sz = int(row.get("size_bytes") or 0)
        if sz > max_file_bytes or vis == "PINNED":
            sub_touched = consolidate_file_sync(
                db,
                role,
                rel_p,
                abs_mem,
                max_file_bytes=max_file_bytes,
                pinned_max_chars=pinned_max_chars,
            )
            for p in sub_touched:
                if p not in touched:
                    touched.append(p)

    sync_role_memories(db, role, abs_mem)
    bump_memory_index_version(role)
    return touched


class MemoryConsolidator:
    """Per-role single-writer background extraction and consolidation coordinator."""

    def __init__(
        self,
        context: KesokuContext,
        *,
        prewarm_embeddings: bool = True,
        enable_timers: bool = True,
    ) -> None:
        """Initialize `MemoryConsolidator` bound to `context`."""
        self.context = context
        self._prewarm_embeddings = prewarm_embeddings
        self._enable_timers = enable_timers
        self._executor: ThreadPoolExecutor | None = None
        self._role_queues: dict[str, asyncio.Queue[ConsolidatorJob]] = {}
        self._role_consumers: dict[str, asyncio.Task[None]] = {}
        self._queued_keys: set[tuple[str, ...]] = set()
        self._debounce_last_ts: dict[str, float] = {}
        self._debounce_timers: dict[str, asyncio.Task[None]] = {}
        self._compaction_hints: dict[str, list[str]] = defaultdict(list)
        self._role_last_consolidated_sig: dict[str, tuple[int, float]] = {}
        self._last_cron_minute: int | None = None
        self._timer_tasks: list[asyncio.Task[None]] = []
        self._running = False

        # Attach onto context & async DB so TurnExecutor and HistoryCompressor discover us automatically.
        try:
            self.context.memory_consolidator = self  # type: ignore[attr-defined]
            self.context.db._memory_consolidator = self  # type: ignore[attr-defined]
        except Exception:
            pass

    @property
    def sync_db(self) -> DatabaseManager:
        """Return synchronous `DatabaseManager`."""
        return self.context.sync_db

    def _resolve_memories_dir(self, role: str) -> str:
        cfg = self.context.config
        roles_dir = cfg.workspace.roles_dir
        if not os.path.isabs(roles_dir) and cfg.agent_working_dir:
            roles_dir = os.path.join(cfg.agent_working_dir, roles_dir)
        return os.path.abspath(os.path.join(roles_dir, role or "default", "memories"))

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kesoku-mem")
        return self._executor

    async def _run_on_worker(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a blocking callable on the dedicated 1-thread executor and call `malloc_trim(0)`."""
        loop = asyncio.get_running_loop()

        def _wrapped() -> Any:
            try:
                return fn(*args, **kwargs)
            finally:
                _guarded_malloc_trim()

        return await loop.run_in_executor(self._get_executor(), _wrapped)

    async def start(self) -> None:
        """Start background timers and pre-warm fastembed on the dedicated worker thread."""
        if self._running:
            return
        self._running = True
        if self._prewarm_embeddings and not os.environ.get("PYTEST_CURRENT_TEST"):
            # Pre-warm and index existing disk memories asynchronously without blocking Agent boot.
            asyncio.create_task(self._safe_prewarm())
        if self._enable_timers:
            self._timer_tasks = [
                asyncio.create_task(self._rescan_loop()),
                asyncio.create_task(self._idle_sweep_loop()),
                asyncio.create_task(self._nightly_cron_loop()),
            ]

    async def _safe_prewarm(self) -> None:
        try:
            await self._run_on_worker(embedding_utils.prewarm)
            for name in await self._run_on_worker(self._discover_roles_with_memories):
                self.enqueue(ReindexMemories(role=name))
        except Exception as e:
            logger.debug(f"Background embedding prewarm skipped: {e}")

    async def drain(self, wait_seconds: float = 20.0) -> None:
        """Wait up to `wait_seconds` for all role queues to finish processing pending jobs."""
        queues = [q for q in self._role_queues.values() if not q.empty() or q._unfinished_tasks > 0]  # type: ignore[attr-defined]
        if not queues:
            return
        try:
            async with asyncio.timeout(wait_seconds):
                await asyncio.gather(*(q.join() for q in queues))
        except TimeoutError:
            logger.warning(f"MemoryConsolidator drain timed out after {wait_seconds}s.")

    async def stop(self, wait_seconds: float = 20.0) -> None:
        """Cancel background timers, drain pending extraction/consolidation jobs, and stop consumers."""
        self._running = False
        for t in list(self._debounce_timers.values()):
            t.cancel()
        self._debounce_timers.clear()
        for t in self._timer_tasks:
            t.cancel()
        if self._timer_tasks:
            await asyncio.gather(*self._timer_tasks, return_exceptions=True)
        self._timer_tasks.clear()

        await self.drain(wait_seconds=wait_seconds)

        for task in list(self._role_consumers.values()):
            task.cancel()
        if self._role_consumers:
            await asyncio.gather(*self._role_consumers.values(), return_exceptions=True)
        self._role_consumers.clear()
        self._queued_keys.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _job_key(self, job: ConsolidatorJob) -> tuple[str, ...]:
        if isinstance(job, EpisodeExtract):
            return ("extract", job.role, job.session_id)
        if isinstance(job, ConsolidateFile):
            return ("file", job.role, job.rel_path)
        if isinstance(job, ConsolidateRole):
            return ("role", job.role)
        return ("reindex", job.role)

    def enqueue(self, job: ConsolidatorJob) -> bool:
        """Enqueue a job onto its role queue (deduplicating identical queued jobs)."""
        safe_role = job.role or "default"
        key = self._job_key(job)
        if key in self._queued_keys:
            # Upgrade existing EpisodeExtract to bypass_budget if a T2 close arrives while queued.
            if isinstance(job, EpisodeExtract) and job.bypass_budget:
                key_bypass = ("extract_bypass", safe_role, job.session_id)
                if key_bypass not in self._queued_keys:
                    self._queued_keys.add(key_bypass)
                    self._ensure_role_queue(safe_role).put_nowait(job)
                    return True
            return False

        self._queued_keys.add(key)
        q = self._ensure_role_queue(safe_role)
        q.put_nowait(job)
        return True

    def _ensure_role_queue(self, role: str) -> asyncio.Queue[ConsolidatorJob]:
        q = self._role_queues.get(role)
        if q is None:
            q = asyncio.Queue()
            self._role_queues[role] = q
        consumer = self._role_consumers.get(role)
        if consumer is None or consumer.done():
            self._role_consumers[role] = asyncio.create_task(self._consume_role_queue(role, q))
        return q

    async def _consume_role_queue(self, role: str, queue: asyncio.Queue[ConsolidatorJob]) -> None:
        while True:
            try:
                job = await queue.get()
            except asyncio.CancelledError:
                break
            key = self._job_key(job)
            self._queued_keys.discard(key)
            self._queued_keys.discard(("extract_bypass", role, getattr(job, "session_id", "")))
            try:
                await self._execute_job(job)
            except asyncio.CancelledError:
                queue.task_done()
                raise
            except Exception as e:
                logger.error(f"MemoryConsolidator job {job} failed: {e}", exc_info=True)
            finally:
                with contextlib.suppress(ValueError):
                    queue.task_done()

    def attach_compaction_hint(self, session_id: str, node_id: str) -> None:
        """Attach a newly created Level-0 `SummaryNode` ID for T3 compaction piggybacking."""
        if session_id and node_id:
            hints = self._compaction_hints[session_id]
            if node_id not in hints:
                hints.append(node_id)

    async def notify_turn(
        self,
        *,
        session_id: str,
        role: str,
        user_msg: Message,
        assistant_text: str = "",
        clicked_option: bool = False,
    ) -> None:
        """Record a completed turn and evaluate T1-immediate (debounced signal) and T1-batch triggers."""
        cfg = self.context.config
        if not cfg.agent.memory_extraction_enabled or is_excluded_from_memory(user_msg):
            return

        _ = assistant_text
        safe_role = role or "default"
        cleaned_user = strip_injected_blocks(user_msg.content or "")
        added_chars = len(cleaned_user)

        def _bump_cursor() -> dict[str, Any]:
            cur = self.sync_db.get_extraction_cursor(session_id)
            prev_turns = int(cur["pending_user_turns"]) if cur else 0
            prev_chars = int(cur["pending_chars"]) if cur else 0
            return self.sync_db.upsert_extraction_cursor(
                session_id,
                safe_role,
                pending_user_turns=prev_turns + 1,
                pending_chars=prev_chars + added_chars,
            )

        cursor = await self._run_on_worker(_bump_cursor)
        pending_turns = int(cursor.get("pending_user_turns") or 0)
        pending_chars = int(cursor.get("pending_chars") or 0)

        # T1-immediate: narrow configurable signal regex, never on clicked options, debounced per session
        sig_pat = (cfg.agent.memory_extraction_signal_regex or "").strip()
        if not clicked_option and sig_pat and cleaned_user:
            if re.search(sig_pat, cleaned_user):
                debounce_s = float(cfg.agent.memory_extraction_debounce_seconds)
                now = time.monotonic()
                last_ts = self._debounce_last_ts.get(session_id)
                if last_ts is None or (now - last_ts) >= debounce_s:
                    self._debounce_last_ts[session_id] = now
                    self.enqueue(EpisodeExtract(role=safe_role, session_id=session_id, reason="signal"))
                    return
                if session_id not in self._debounce_timers or self._debounce_timers[session_id].done():
                    delay = max(0.05, debounce_s - (now - last_ts))
                    self._debounce_timers[session_id] = asyncio.create_task(
                        self._delayed_signal_enqueue(safe_role, session_id, delay)
                    )
                return

        # T1-batch: >= min_turns or >= min_chars (requiring >= 20 meaningful chars so pure "ok" spam waits)
        min_turns = int(cfg.agent.memory_extraction_min_turns)
        min_chars = int(cfg.agent.memory_extraction_min_chars)
        if pending_chars >= 20 and (
            (min_turns > 0 and pending_turns >= min_turns) or (min_chars > 0 and pending_chars >= min_chars)
        ):
            self.enqueue(EpisodeExtract(role=safe_role, session_id=session_id, reason="batch"))

    async def _delayed_signal_enqueue(self, role: str, session_id: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            self._debounce_last_ts[session_id] = time.monotonic()
            self.enqueue(EpisodeExtract(role=role, session_id=session_id, reason="signal_debounced"))
        except asyncio.CancelledError:
            pass
        finally:
            self._debounce_timers.pop(session_id, None)

    async def notify_close(
        self,
        session_id: str,
        role: str | None = None,
        reason: str = "close",
    ) -> bool:
        """Evaluate T2 session-close trigger (`pending_user_turns >= 1` and `pending_chars >= 20`)."""
        cfg = self.context.config
        if not cfg.agent.memory_extraction_enabled or not session_id:
            return False

        def _check_close() -> tuple[bool, str]:
            cur = self.sync_db.get_extraction_cursor(session_id)
            digest = self.sync_db.get_session_digest(session_id)
            if digest and str(digest.get("chatbot_id") or "").lower() == "cronjob":
                return False, "default"
            eff_role = role or (str(cur["role"]) if cur else "") or (str(digest["role"]) if digest else "")
            if not eff_role:
                sess = self.sync_db.get_session(session_id)
                eff_role = (sess.role_name if sess and sess.role_name else None) or "default"

            if cur is not None:
                is_ready = int(cur.get("pending_user_turns") or 0) >= 1 and int(cur.get("pending_chars") or 0) >= 20
                return is_ready, eff_role

            msgs = self.sync_db.get_unextracted_messages(session_id, after_ts=0.0, after_id="", limit=50)
            user_msgs = [m for m in msgs if m.role == MessageRole.USER and not is_excluded_from_memory(m)]
            total_chars = sum(len(strip_injected_blocks(m.content or "")) for m in user_msgs)
            return (len(user_msgs) >= 1 and total_chars >= 20), eff_role

        should_run, eff_role = await self._run_on_worker(_check_close)
        if should_run:
            return self.enqueue(
                EpisodeExtract(role=eff_role, session_id=session_id, reason=reason, bypass_budget=True)
            )
        return False

    async def run_idle_sweep_once(self, now_ts: float | None = None) -> int:
        """Scan cursors with pending turns idle beyond `memory_extraction_idle_seconds` (T4)."""
        cfg = self.context.config
        if not cfg.agent.memory_extraction_enabled:
            return 0
        ref_now = float(now_ts if now_ts is not None else time.time())
        idle_cutoff = ref_now - float(cfg.agent.memory_extraction_idle_seconds)

        def _find_idle() -> list[tuple[str, str]]:
            out: list[tuple[str, str]] = []
            for cur in self.sync_db.list_extraction_cursors(min_pending_turns=1):
                if int(cur.get("pending_chars") or 0) < 20 or int(cur.get("failures") or 0) >= 3:
                    continue
                sid = str(cur["session_id"])
                digest = self.sync_db.get_session_digest(sid)
                last_active = float(digest["last_active_at"]) if digest else 0.0
                if last_active <= idle_cutoff:
                    out.append((str(cur["role"] or "default"), sid))
            return out

        candidates = await self._run_on_worker(_find_idle)
        count = 0
        for r_name, sid in candidates:
            if self.enqueue(EpisodeExtract(role=r_name, session_id=sid, reason="idle_sweep")):
                count += 1
        return count

    async def run_nightly_tick_once(self, now: datetime.datetime | None = None) -> int:
        """Evaluate nightly `memory_consolidation_cron` (T5) and retry deferred failed sessions."""
        cfg = self.context.config
        sched = (cfg.agent.memory_consolidation_cron or "").strip()
        if not sched:
            return 0
        dt = now or datetime.datetime.now()
        minute_epoch = int(dt.timestamp() // 60)
        if self._last_cron_minute == minute_epoch or not cron_matches(sched, dt):
            return 0
        self._last_cron_minute = minute_epoch

        def _discover_roles_and_deferred() -> tuple[set[str], list[tuple[str, str]]]:
            roles: set[str] = set()
            roles_root = cfg.workspace.roles_dir
            if not os.path.isabs(roles_root) and cfg.agent_working_dir:
                roles_root = os.path.join(cfg.agent_working_dir, roles_root)
            if os.path.isdir(roles_root):
                for entry in os.listdir(roles_root):
                    if os.path.isdir(os.path.join(roles_root, entry, "memories")):
                        roles.add(entry)
            deferred: list[tuple[str, str]] = []
            for cur in self.sync_db.list_extraction_cursors(min_pending_turns=1):
                roles.add(str(cur["role"] or "default"))
                if int(cur.get("failures") or 0) >= 3:
                    deferred.append((str(cur["role"] or "default"), str(cur["session_id"])))
            return roles, deferred

        roles, deferred = await self._run_on_worker(_discover_roles_and_deferred)
        enqueued = 0
        for r_name, sid in deferred:
            if self.enqueue(EpisodeExtract(role=r_name, session_id=sid, reason="nightly", force=True)):
                enqueued += 1
        for r_name in sorted(roles):
            if self.enqueue(ConsolidateRole(role=r_name)):
                enqueued += 1
        return enqueued

    def _discover_roles_with_memories(self) -> list[str]:
        cfg = self.context.config
        roles_root = cfg.workspace.roles_dir
        if not os.path.isabs(roles_root) and cfg.agent_working_dir:
            roles_root = os.path.join(cfg.agent_working_dir, roles_root)
        if not os.path.isdir(roles_root):
            return []
        return [
            name
            for name in os.listdir(roles_root)
            if os.path.isdir(os.path.join(roles_root, name, "memories"))
        ]

    async def _rescan_loop(self) -> None:
        while self._running:
            await asyncio.sleep(RESCAN_INTERVAL_SECONDS)
            try:
                for name in await self._run_on_worker(self._discover_roles_with_memories):
                    self.enqueue(ReindexMemories(role=name))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Background memory rescan error: {e}")

    async def _idle_sweep_loop(self) -> None:
        while self._running:
            await asyncio.sleep(IDLE_SWEEP_INTERVAL_SECONDS)
            try:
                await self.run_idle_sweep_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Memory idle sweep error: {e}")

    async def _nightly_cron_loop(self) -> None:
        while self._running:
            await asyncio.sleep(CRON_POLL_INTERVAL_SECONDS)
            try:
                await self.run_nightly_tick_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Memory nightly cron tick error: {e}")

    def _resolve_extractor_llm(self) -> BaseLLM:
        """Resolve the low-cost extraction LLM (`thinking_level='minimal'` on Gemini, `'low'` on Claude)."""
        cfg = self.context.config
        provider = cfg.agent.context_llm or cfg.agent.llm
        thinking = resolve_extraction_thinking_level(provider, cfg.agent.memory_extraction_thinking_level)
        model_override = (cfg.agent.memory_extraction_model or "").strip() or None
        if getattr(self.context, "_llm", None) is not None and model_override is None:
            # Respect injected mock LLM in unit tests while recording resolved thinking level for verification.
            mock_llm = self.context._llm
            if mock_llm is not None:
                mock_llm._last_extraction_thinking_level = thinking  # type: ignore[attr-defined]
                return mock_llm
        return self.context.get_llm(
            provider=provider,
            use_context_compression=True,
            model_name=model_override,
            thinking_level=thinking,
        )

    async def _execute_job(self, job: ConsolidatorJob) -> None:
        if isinstance(job, ReindexMemories):
            mem_dir = self._resolve_memories_dir(job.role)
            await self._run_on_worker(sync_role_memories, self.sync_db, job.role, mem_dir, force=job.force)
            return

        if isinstance(job, ConsolidateFile):
            cfg = self.context.config
            mem_dir = self._resolve_memories_dir(job.role)
            await self._run_on_worker(
                consolidate_file_sync,
                self.sync_db,
                job.role,
                job.rel_path,
                mem_dir,
                max_file_bytes=cfg.agent.memory_file_max_bytes,
                pinned_max_chars=cfg.agent.memory_pinned_max_chars,
            )
            return

        if isinstance(job, ConsolidateRole):
            cfg = self.context.config
            mem_dir = self._resolve_memories_dir(job.role)
            cur_sig = await self._run_on_worker(self.sync_db.get_memory_role_signature, job.role)
            if not job.force and self._role_last_consolidated_sig.get(job.role) == cur_sig:
                return
            await self._run_on_worker(
                consolidate_role_sync,
                self.sync_db,
                job.role,
                mem_dir,
                max_file_bytes=cfg.agent.memory_file_max_bytes,
                pinned_max_chars=cfg.agent.memory_pinned_max_chars,
            )
            self._role_last_consolidated_sig[job.role] = await self._run_on_worker(
                self.sync_db.get_memory_role_signature, job.role
            )
            return

        if isinstance(job, EpisodeExtract):
            await self.run_episode_extract(job)

    async def run_episode_extract(
        self,
        job: EpisodeExtract,
        *,
        dry_run: bool = False,
        safe_upper: float | tuple[float, str] | None = None,
    ) -> ExtractionOutcome:
        """Execute a single `EpisodeExtract` job safely (`C4`, `R2-1`, `H4`, `C5`)."""
        cfg = self.context.config
        started_at = time.time()
        safe_role = job.role or "default"
        session_id = job.session_id

        cursor = await self._run_on_worker(self.sync_db.get_extraction_cursor, session_id)
        failures = int(cursor.get("failures") or 0) if cursor else 0
        last_run_at = float(cursor["last_run_at"]) if (cursor and cursor.get("last_run_at")) else 0.0

        # Backoff & 3-failure deferral gate
        if not job.force and failures >= 3 and job.reason != "nightly":
            return ExtractionOutcome(status="deferred_after_failures")
        if not job.force and 1 <= failures < 3 and last_run_at > 0:
            backoff_s = min(600.0, 30.0 * (2 ** (failures - 1)))
            if (started_at - last_run_at) < backoff_s:
                return ExtractionOutcome(status="backoff")

        # Daily token budget gate (when exhausted, only T2 bypass_budget runs)
        budget = int(cfg.agent.memory_extraction_daily_token_budget)
        if budget >= 0 and not job.bypass_budget and not job.force:
            spent_today = await self._run_on_worker(self.sync_db.get_daily_extraction_tokens, safe_role)
            if spent_today >= budget:
                await self._run_on_worker(
                    lambda: self.sync_db.log_extraction(
                        role=safe_role,
                        session_id=session_id,
                        reason=job.reason,
                        started_at=started_at,
                        finished_at=time.time(),
                        status="budget_exhausted",
                    )
                )
                return ExtractionOutcome(status="budget_exhausted")

        after_ts = float(cursor.get("last_extracted_msg_ts") or 0.0) if cursor else 0.0
        after_id = str(cursor.get("last_extracted_msg_id") or "") if cursor else ""

        msgs = await self._run_on_worker(
            self.sync_db.get_unextracted_messages,
            session_id,
            after_ts,
            after_id,
            safe_upper,
            200,
        )
        if not msgs:
            return ExtractionOutcome(status="empty_window")

        turns = segment_logical_turns(msgs)
        if not turns:
            return ExtractionOutcome(status="empty_window")

        # Resolve per-turn context_policy
        staging_dir = os.path.join(cfg.workspace.sessions_dir, session_id)
        disk_entries = await collect_context_entries(staging_dir, include_policy_only=True)
        inferred_policies = infer_backfill_turn_policies(turns, disk_entries=disk_entries, cfg=cfg)

        turn_policies: list[ContextPolicy] = []
        for turn, inf_pol in zip(turns, inferred_policies, strict=True):
            explicit = next(
                (
                    normalize_policy(m.metadata.get("context_policy"))
                    for m in turn
                    if m.role == MessageRole.USER and isinstance(m.metadata, Mapping)
                ),
                None,
            )
            turn_policies.append(explicit if explicit is not None else inf_pol)

        last_msg = msgs[-1]

        # All-isolated window -> 0 LLM tokens, mark extracted immediately!
        if all(p == "isolated" for p in turn_policies):
            if not dry_run:
                await self._advance_cursor_after_success(
                    session_id=session_id,
                    role=safe_role,
                    last_msg=last_msg,
                    reason=job.reason,
                    tokens_added=0,
                )
                await self._run_on_worker(
                    lambda: self.sync_db.log_extraction(
                        role=safe_role,
                        session_id=session_id,
                        reason=job.reason,
                        started_at=started_at,
                        finished_at=time.time(),
                        status="skipped_isolated",
                    )
                )
            return ExtractionOutcome(status="skipped_isolated", messages_processed=len(msgs))

        rendered_turns = [
            format_turn_for_extraction(t, pol) for t, pol in zip(turns, turn_policies, strict=True)
        ]
        window_query_parts: list[str] = []
        for t, pol in zip(turns, turn_policies, strict=True):
            if pol == "isolated":
                continue
            for m in t:
                if m.role in (MessageRole.USER, MessageRole.ASSISTANT) and m.type == MessageType.TEXT:
                    cleaned = strip_injected_blocks(m.content or "")
                    if cleaned:
                        window_query_parts.append(cleaned)
        window_query = "\n".join(window_query_parts)[-2000:]

        # Gather any attached T3 compaction hints
        hint_node_ids = list(self._compaction_hints.get(session_id, ()))
        compaction_summaries: list[str] = []
        if hint_node_ids:
            all_nodes = await self.context.db.get_all_summary_nodes(session_id)
            by_nid = {n.id: n.summary for n in all_nodes}
            compaction_summaries = [by_nid[nid] for nid in hint_node_ids if nid in by_nid]

        extractor = EpisodeExtractor(cfg)
        try:
            prompt, snapshots, mem_dir = await self._run_on_worker(
                lambda: extractor.prepare_prompt_sync(
                    db=self.sync_db,
                    role=safe_role,
                    session_id=session_id,
                    rendered_turns=rendered_turns,
                    window_query_text=window_query,
                    compaction_hints=compaction_summaries,
                )
            )
            llm = self._resolve_extractor_llm()
            res = await llm.generate(prompt=prompt)
            in_tok = int(res.prompt_tokens or 0)
            out_tok = int(res.candidates_tokens or 0)
            if in_tok == 0 and out_tok == 0 and res.total_tokens:
                in_tok = int(res.total_tokens)

            parsed = parse_extractor_json(res.content)
            raw_ops = [op for op in (parsed.get("ops") or []) if isinstance(op, Mapping)]
            ep_summary = " ".join(str(parsed.get("episode_summary") or "").split()).strip()[:MAX_EPISODE_SUMMARY_CHARS]
            raw_threads = parsed.get("open_threads") or []
            open_threads = (
                [str(x).strip()[:80] for x in raw_threads if str(x).strip()][:3]
                if isinstance(raw_threads, list)
                else []
            )
            event_date = datetime.datetime.fromtimestamp(last_msg.timestamp).astimezone().strftime("%Y-%m-%d")

            if dry_run:
                return ExtractionOutcome(
                    status="dry_run",
                    episode_summary=ep_summary,
                    open_threads=open_threads,
                    ops=[dict(o) for o in raw_ops],
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    messages_processed=len(msgs),
                )

            def _apply_and_persist() -> MemoryWriteResult:
                wr = MemoryWriter.apply_ops(
                    raw_ops,
                    role=safe_role,
                    memories_dir=mem_dir,
                    db=self.sync_db,
                    session_id=session_id,
                    event_date=event_date,
                    pinned_files=cfg.agent.memory_pinned_files,
                    rejected_tags=cfg.agent.memory_rejected_tags,
                    max_file_bytes=cfg.agent.memory_file_max_bytes,
                    snapshots=snapshots,
                )
                if ep_summary or open_threads:
                    sess = self.sync_db.get_session(session_id)
                    emb_bytes: bytes | None = None
                    if ep_summary:
                        try:
                            title_prefix = f"{sess.title}\n" if (sess and sess.title) else ""
                            vec = embedding_utils.get_embedding(f"{title_prefix}{ep_summary}")
                            if vec:
                                emb_bytes = embedding_utils.vector_to_bytes(vec)
                        except Exception as emb_err:
                            logger.warning(f"Failed embedding episode summary for '{session_id}': {emb_err}")
                    self.sync_db.upsert_session_digest(
                        session_id=session_id,
                        role=safe_role,
                        title=sess.title if sess else None,
                        started_at=sess.created_at if sess else float(msgs[0].timestamp),
                        last_active_at=float(last_msg.timestamp),
                        summary=ep_summary if ep_summary else None,
                        open_threads=open_threads,
                        summary_upto_ts=float(last_msg.timestamp),
                        embedding=emb_bytes,
                    )
                return wr

            write_res = await self._run_on_worker(_apply_and_persist)
            self._compaction_hints.pop(session_id, None)

            await self._advance_cursor_after_success(
                session_id=session_id,
                role=safe_role,
                last_msg=last_msg,
                reason=job.reason,
                tokens_added=in_tok + out_tok,
            )
            await self._run_on_worker(
                lambda: self.sync_db.log_extraction(
                    role=safe_role,
                    session_id=session_id,
                    reason=job.reason,
                    started_at=started_at,
                    finished_at=time.time(),
                    status="ok",
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    ops_applied=write_res.ops_applied,
                )
            )
            for over_rel in write_res.files_over_cap:
                self.enqueue(ConsolidateFile(role=safe_role, rel_path=over_rel))

            return ExtractionOutcome(
                status="ok",
                episode_summary=ep_summary,
                open_threads=open_threads,
                ops=[dict(o) for o in raw_ops],
                ops_applied=write_res.ops_applied,
                input_tokens=in_tok,
                output_tokens=out_tok,
                messages_processed=len(msgs),
                files_over_cap=list(write_res.files_over_cap),
            )
        except Exception as e:
            err_msg = str(e)[:500]
            logger.warning(f"EpisodeExtract failed for session '{session_id}' (role='{safe_role}'): {e}")
            if not dry_run:
                def _record_failure() -> None:
                    cur = self.sync_db.get_extraction_cursor(session_id)
                    prev_fail = int(cur.get("failures") or 0) if cur else 0
                    self.sync_db.upsert_extraction_cursor(
                        session_id,
                        safe_role,
                        last_run_at=time.time(),
                        last_reason=job.reason,
                        failures=prev_fail + 1,
                    )
                    self.sync_db.log_extraction(
                        role=safe_role,
                        session_id=session_id,
                        reason=job.reason,
                        started_at=started_at,
                        finished_at=time.time(),
                        status="error",
                        error=err_msg,
                    )

                await self._run_on_worker(_record_failure)
            return ExtractionOutcome(status="error")

    async def _advance_cursor_after_success(
        self,
        *,
        session_id: str,
        role: str,
        last_msg: Message,
        reason: str,
        tokens_added: int,
    ) -> None:
        """Advance `memory_extraction_cursors` strictly to `last_msg`, preserving any newer completed turns."""

        def _update() -> None:
            cur = self.sync_db.get_extraction_cursor(session_id)
            prev_tokens = int(cur.get("tokens_spent") or 0) if cur else 0
            remaining = self.sync_db.get_unextracted_messages(
                session_id,
                after_ts=float(last_msg.timestamp),
                after_id=last_msg.id,
                limit=200,
            )
            rem_turns = sum(1 for m in remaining if m.role == MessageRole.USER and not is_excluded_from_memory(m))
            rem_chars = sum(
                len(strip_injected_blocks(m.content or ""))
                for m in remaining
                if m.role in (MessageRole.USER, MessageRole.ASSISTANT) and m.type == MessageType.TEXT
            )
            self.sync_db.upsert_extraction_cursor(
                session_id,
                role,
                last_extracted_msg_ts=float(last_msg.timestamp),
                last_extracted_msg_id=last_msg.id,
                pending_user_turns=rem_turns,
                pending_chars=rem_chars,
                last_run_at=time.time(),
                last_reason=reason,
                tokens_spent=prev_tokens + max(0, int(tokens_added)),
                failures=0,
            )

        await self._run_on_worker(_update)

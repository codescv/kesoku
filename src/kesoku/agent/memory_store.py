"""Markdown-based role memory indexing, pinned core memory cache, and hybrid search for Kesoku."""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kesoku.constants import MessageRole, MessageType
from kesoku.db.manager import DatabaseManager, _batch_semantic_scores
from kesoku.db.models import Message
from kesoku.utils import embedding as embedding_utils
from kesoku.utils.path import PathResolver
from kesoku.utils.text import chunk_message_text
from kesoku.utils.ui_markup import collapse_ui_blocks

logger = logging.getLogger(__name__)

# Matches legacy `[question: stem || opt1 | opt2]` blocks in assistant messages.
_QUESTION_BLOCK_RE = re.compile(r"\[question:\s*([^\]]+)\]")

# Matches YAML frontmatter at the very beginning of a Markdown file.
_FRONTMATTER_RE = re.compile(r"\A\s*---\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", re.DOTALL)
# Matches Markdown ATX headings (# H1 .. ###### H6).
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# Matches fenced code block delimiters (``` or ~~~).
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
# Strips injected wrapper blocks from user messages when building a multi-turn search query.
_INJECTED_BLOCKS_RE = re.compile(
    r"<(?:instructions|memory_pinned\b[^>]*|recent_sessions\b[^>]*|context\b[^>]*|facts|memory|"
    r"background_context\b[^>]*|relevant_historical_context)>.*?"
    r"</(?:instructions|memory_pinned|recent_sessions|context|facts|memory|"
    r"background_context|relevant_historical_context)>\s*",
    re.DOTALL,
)
_CURRENT_MESSAGE_TAG_RE = re.compile(r"</?current_message\b[^>]*>\s*")
# Tokenizes ASCII identifiers/words (>=2 chars) and CJK character runs (>=2 chars) for lexical boosting.
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_.\-/]{2,}")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]{2,}")

MAX_LEXICAL_BOOST = 0.10
VALID_VISIBILITIES = frozenset({"DEFAULT", "PINNED", "HIDDEN"})

_version_lock = threading.Lock()
memory_index_version: dict[str, int] = defaultdict(int)


def get_memory_index_version(role: str) -> int:
    """Return the current in-process memory index generation counter for `role`."""
    safe_role = role or "default"
    with _version_lock:
        return memory_index_version[safe_role]


def bump_memory_index_version(role: str) -> int:
    """Increment and return the in-process memory index generation counter for `role`."""
    safe_role = role or "default"
    with _version_lock:
        memory_index_version[safe_role] += 1
        return memory_index_version[safe_role]


@dataclass
class MemoryFrontmatter:
    """Parsed YAML frontmatter metadata for a Markdown memory file."""

    description: str = ""
    visibility: str = "DEFAULT"
    cues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Render as the single-line description string."""
        return self.description

    def __eq__(self, other: object) -> bool:
        """Compare against another `MemoryFrontmatter` or a plain description `str`."""
        if isinstance(other, MemoryFrontmatter):
            return (self.description, self.visibility, self.cues) == (
                other.description,
                other.visibility,
                other.cues,
            )
        if isinstance(other, str):
            return self.description == other
        return NotImplemented


@dataclass(frozen=True)
class MemoryMatch:
    """A ranked role memory file or episode returned by semantic + lexical search."""

    rel_path: str
    memory_path: str
    description: str
    score: float
    mtime: float
    visibility: str = "DEFAULT"
    kind: str = "file"


def _strip_matching_quotes(val: str) -> str:
    """Strip a single pair of matching surrounding quotes."""
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in {'"', "'"}:
        return val[1:-1].strip()
    return val


def _extract_yaml_fields(yaml_block: str) -> tuple[str, str, list[str]]:
    """Parse `description`, `visibility`, and `cues` from a YAML frontmatter block without heavy deps."""
    lines = yaml_block.splitlines()
    description = ""
    visibility = "DEFAULT"
    cues: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if stripped.startswith("description:"):
            raw_val = stripped[len("description:") :].strip()
            if raw_val in {"|", ">", "|-", ">-"}:
                collected: list[str] = []
                i += 1
                while i < len(lines):
                    sub = lines[i]
                    if sub and not sub[0].isspace():
                        break
                    if sub.strip():
                        collected.append(sub.strip())
                    i += 1
                description = " ".join(collected)
                continue
            description = _strip_matching_quotes(raw_val)
        elif stripped.startswith("visibility:"):
            raw_vis = _strip_matching_quotes(stripped[len("visibility:") :]).upper()
            if raw_vis in VALID_VISIBILITIES:
                visibility = raw_vis
        elif stripped.startswith("cues:"):
            raw_cues = stripped[len("cues:") :].strip()
            if raw_cues.startswith("[") and raw_cues.endswith("]"):
                inner = raw_cues[1:-1].strip()
                if inner:
                    cues = [_strip_matching_quotes(x) for x in inner.split(",") if _strip_matching_quotes(x)]
            elif raw_cues:
                cues = [_strip_matching_quotes(x) for x in raw_cues.split(",") if _strip_matching_quotes(x)]
            else:
                i += 1
                while i < len(lines):
                    sub = lines[i]
                    if sub and not sub[0].isspace() and not sub.lstrip().startswith("-"):
                        break
                    sub_s = sub.strip()
                    if sub_s.startswith("-"):
                        item = _strip_matching_quotes(sub_s[1:])
                        if item:
                            cues.append(item)
                    elif sub_s:
                        break
                    i += 1
                continue
        i += 1
    return description, visibility, cues


def _fallback_description_from_body(body: str) -> str:
    """Derive a fallback description from the first Markdown heading or non-blank line."""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            return heading_match.group(2).strip()
        return stripped.lstrip("-*+># ").strip()
    return ""


def parse_memory_frontmatter(text: str) -> tuple[MemoryFrontmatter, str]:
    """Extract `MemoryFrontmatter(description, visibility, cues)` and Markdown body from a memory file.

    Args:
        text: Raw UTF-8 content of a `.md` memory file.

    Returns:
        A tuple of `(MemoryFrontmatter, markdown_body)`.
    """
    if not text:
        return MemoryFrontmatter(), ""

    description = ""
    visibility = "DEFAULT"
    cues: list[str] = []
    body = text
    match = _FRONTMATTER_RE.match(text)
    if match:
        yaml_block = match.group(1)
        body = text[match.end() :]
        description, visibility, cues = _extract_yaml_fields(yaml_block)

    if not description:
        description = _fallback_description_from_body(body)

    clean_desc = " ".join(description.split()).strip()
    if len(clean_desc) > 240:
        clean_desc = clean_desc[:239].rstrip() + "…"
    return MemoryFrontmatter(description=clean_desc, visibility=visibility, cues=cues), body.strip()


def chunk_memory_markdown(
    rel_path: str,
    description: str,
    body: str,
    max_chunk_chars: int = 600,
    cues: Sequence[str] = (),
) -> list[str]:
    r"""Split a Markdown memory into multi-granularity chunks with ancestor heading breadcrumbs.

    Args:
        rel_path: Path of the memory file relative to `$MEMORY_DIR`.
        description: Single-line description of the memory file.
        body: Markdown body (without frontmatter).
        max_chunk_chars: Maximum character budget per body chunk before paragraph splitting.
        cues: Optional retrieval cue phrases from frontmatter appended to chunk 0.

    Returns:
        List of non-empty text chunks ready for embedding.
    """
    desc_str = str(description or "")
    header_parts = [f"# {rel_path}"]
    if desc_str:
        header_parts.append(desc_str)
    if cues:
        header_parts.append(" ".join(str(c) for c in cues if c))
    header_tag = "\n".join(header_parts).strip()
    chunks: list[str] = [header_tag]

    stripped_body = body.strip()
    if not stripped_body:
        return chunks

    file_context_line = f"# {rel_path} ({desc_str})" if desc_str else f"# {rel_path}"

    sections: list[tuple[list[str], list[str]]] = []
    ancestors: dict[int, str] = {}
    current_lines: list[str] = []
    in_fence = False

    def _current_breadcrumbs() -> list[str]:
        return [ancestors[level] for level in sorted(ancestors)]

    for line in stripped_body.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        if not in_fence:
            heading_match = _HEADING_RE.match(line)
            if heading_match:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append((_current_breadcrumbs(), section_text.splitlines()))
                current_lines = []

                level = len(heading_match.group(1))
                for deeper in [k for k in ancestors if k >= level]:
                    del ancestors[deeper]
                ancestors[level] = line.strip()
                continue

        current_lines.append(line)

    tail_text = "\n".join(current_lines).strip()
    if tail_text or ancestors:
        sections.append((_current_breadcrumbs(), tail_text.splitlines() if tail_text else []))

    for breadcrumbs, lines in sections:
        prefix_parts = [file_context_line, *breadcrumbs]
        prefix = "\n".join(prefix_parts)
        section_body = "\n".join(lines).strip()
        if not section_body:
            chunks.append(prefix)
            continue

        sub_chunks = chunk_message_text(
            section_body,
            threshold=max(120, max_chunk_chars),
        )
        for sub in sub_chunks:
            sub_stripped = sub.strip()
            if sub_stripped:
                chunks.append(f"{prefix}\n{sub_stripped}")

    return chunks


def sync_role_memories(
    db: DatabaseManager,
    role: str,
    memories_dir: str | None = None,
    *,
    force: bool = False,
) -> int:
    """Incrementally synchronize on-disk `.md` files under `$MEMORY_DIR` with SQLite embeddings.

    Persists `visibility` and `size_bytes`, embeds only changed files, and bumps
    `memory_index_version[role]` whenever the index changes.

    Args:
        db: Synchronous `DatabaseManager` instance.
        role: Role persona identifier.
        memories_dir: Optional override directory path; defaults to `PathResolver.get_role_memories_dir(role)`.
        force: If True, clears existing index entries for `role` and re-embeds all files.

    Returns:
        Number of memory files that were newly indexed or updated.
    """
    safe_role = role or "default"
    target_dir = os.path.abspath(memories_dir or PathResolver.get_role_memories_dir(safe_role))

    if force:
        db.clear_role_memory_index(safe_role)

    existing_rows = db.get_memory_files_by_role(safe_role)
    existing_by_path: dict[str, dict[str, Any]] = {row["rel_path"]: row for row in existing_rows}

    if not os.path.isdir(target_dir):
        if existing_rows:
            db.delete_memory_files_by_ids([int(r["id"]) for r in existing_rows])
            bump_memory_index_version(safe_role)
        elif force:
            bump_memory_index_version(safe_role)
        return 0

    disk_files: dict[str, str] = {}
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for fname in sorted(files):
            if fname.startswith(".") or not fname.lower().endswith(".md"):
                continue
            abs_path = os.path.join(root, fname)
            rel_path = os.path.relpath(abs_path, target_dir).replace(os.sep, "/")
            disk_files[rel_path] = abs_path

    deleted_ids = [
        int(row["id"])
        for rel_path, row in existing_by_path.items()
        if rel_path not in disk_files
    ]
    if deleted_ids:
        db.delete_memory_files_by_ids(deleted_ids)

    updated_count = 0
    now = time.time()

    for rel_path, abs_path in disk_files.items():
        try:
            stat = os.stat(abs_path)
        except OSError as e:
            logger.warning(f"Failed to stat memory file '{abs_path}': {e}")
            continue

        mtime = float(stat.st_mtime)
        prev = existing_by_path.get(rel_path)
        if prev is not None and not force and abs(mtime - float(prev["mtime"])) < 1e-4:
            continue

        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        except OSError as e:
            logger.warning(f"Failed to read memory file '{abs_path}': {e}")
            continue

        if not raw_text.strip():
            if prev is not None:
                db.delete_memory_files_by_ids([int(prev["id"])])
                deleted_ids.append(int(prev["id"]))
            continue

        raw_bytes = raw_text.encode("utf-8")
        size_bytes = len(raw_bytes)
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        if prev is not None and not force and prev.get("content_hash") == content_hash:
            continue

        fm, body = parse_memory_frontmatter(raw_text)
        chunk_texts = chunk_memory_markdown(rel_path, fm.description, body, cues=fm.cues)

        try:
            embeddings = embedding_utils.get_embeddings(chunk_texts)
        except Exception as e:
            logger.error(f"Failed to compute embeddings for memory '{rel_path}': {e}", exc_info=True)
            embeddings = []

        chunk_tuples: list[tuple[int, str, bytes]] = []
        if embeddings and len(embeddings) == len(chunk_texts):
            for idx, (c_text, emb) in enumerate(zip(chunk_texts, embeddings, strict=True)):
                chunk_tuples.append((idx, c_text, embedding_utils.vector_to_bytes(emb)))
        else:
            for idx, c_text in enumerate(chunk_texts):
                chunk_tuples.append((idx, c_text, b""))

        db.upsert_memory_file(
            role=safe_role,
            rel_path=rel_path,
            description=fm.description,
            content_hash=content_hash,
            mtime=mtime,
            updated_at=now,
            chunks=chunk_tuples,
            visibility=fm.visibility,
            size_bytes=size_bytes,
        )
        updated_count += 1

    if updated_count > 0 or deleted_ids or force:
        bump_memory_index_version(safe_role)

    return updated_count


def _extract_query_tokens(query: str) -> list[str]:
    """Extract lowercase ASCII tokens and CJK 2-grams/runs for lexical boosting."""
    lower = query.lower()
    tokens: list[str] = []
    seen: set[str] = set()

    for match in _ASCII_TOKEN_RE.finditer(lower):
        tok = match.group(0).strip("._-/")
        if len(tok) >= 2 and tok not in seen:
            seen.add(tok)
            tokens.append(tok)

    for match in _CJK_RUN_RE.finditer(lower):
        run = match.group(0)
        if len(run) <= 4:
            if run not in seen:
                seen.add(run)
                tokens.append(run)
        else:
            for i in range(len(run) - 1):
                bg = run[i : i + 2]
                if bg not in seen:
                    seen.add(bg)
                    tokens.append(bg)
    return tokens


def _compute_lexical_boost(
    query_tokens: list[str],
    rel_path: str,
    description: str,
    content_text: str,
) -> float:
    """Compute a small bounded lexical score boost `[0.0, MAX_LEXICAL_BOOST]` for exact token hits."""
    if not query_tokens:
        return 0.0

    meta_haystack = f"{rel_path}\n{description}".lower()
    body_haystack = content_text.lower()

    meta_hits = sum(1 for tok in query_tokens if tok in meta_haystack)
    body_hits = sum(1 for tok in query_tokens if tok in body_haystack)

    total = len(query_tokens)
    meta_ratio = meta_hits / total
    body_ratio = body_hits / total

    boost = 0.06 * meta_ratio + 0.04 * body_ratio
    return min(MAX_LEXICAL_BOOST, boost)


@dataclass
class _RoleChunkMatrixEntry:
    """Cached L2-normalized float32 chunk matrix and metadata for fast pre-turn search."""

    version: int
    db_signature: tuple[int, float]
    chunk_meta: list[dict[str, Any]]
    valid_indices: list[int]
    unit_matrix: np.ndarray | None
    dim: int


_matrix_cache_lock = threading.Lock()
_role_matrix_cache: dict[tuple[str, str, bool], _RoleChunkMatrixEntry] = {}


def _get_or_build_role_matrix(
    db: DatabaseManager,
    role: str,
    include_pinned: bool,
    expected_dim: int,
) -> _RoleChunkMatrixEntry:
    """Return the cached normalized chunk matrix for `(db, role, include_pinned)`, rebuilding on version bump."""
    key = (os.path.abspath(db.db_path), role, include_pinned)
    cur_ver = get_memory_index_version(role)
    cur_sig = db.get_memory_role_signature(role)

    with _matrix_cache_lock:
        cached = _role_matrix_cache.get(key)
        if (
            cached is not None
            and cached.version == cur_ver
            and cached.db_signature == cur_sig
            and (cached.dim == expected_dim or expected_dim == 0 or cached.unit_matrix is None)
        ):
            return cached

    chunk_rows = db.get_memory_chunks_for_role(role, include_pinned=include_pinned)
    chunk_meta: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    blobs: list[bytes] = []
    inferred_dim = expected_dim

    for idx, row in enumerate(chunk_rows):
        chunk_meta.append(
            {
                "file_id": int(row["file_id"]),
                "rel_path": str(row["rel_path"]),
                "description": str(row["description"] or ""),
                "visibility": str(row["visibility"] or "DEFAULT"),
                "mtime": float(row["mtime"]),
                "content": str(row["content"] or ""),
            }
        )
        blob = row["chunk_embedding"]
        if blob and len(blob) % 4 == 0:
            row_dim = len(blob) // 4
            if inferred_dim == 0:
                inferred_dim = row_dim
            if row_dim == inferred_dim:
                valid_indices.append(idx)
                blobs.append(blob)

    unit_matrix: np.ndarray | None = None
    if valid_indices and inferred_dim > 0:
        raw_mat = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(valid_indices), inferred_dim).copy()
        norms = np.linalg.norm(raw_mat, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            unit_matrix = np.where(norms > 0.0, raw_mat / norms, 0.0).astype(np.float32, copy=False)

    entry = _RoleChunkMatrixEntry(
        version=cur_ver,
        db_signature=cur_sig,
        chunk_meta=chunk_meta,
        valid_indices=valid_indices,
        unit_matrix=unit_matrix,
        dim=inferred_dim,
    )
    with _matrix_cache_lock:
        _role_matrix_cache[key] = entry
    return entry


def search_role_memories(
    db: DatabaseManager,
    role: str,
    query: str,
    limit: int = 5,
    min_score: float = 0.50,
    memories_dir: str | None = None,
    *,
    include_pinned: bool = False,
    allow_sync: bool = False,
) -> list[MemoryMatch]:
    """Search a role's Markdown memories using `MAX(chunk_similarity)` + lexical boost.

    Args:
        db: Synchronous `DatabaseManager` instance.
        role: Role persona identifier.
        query: Natural language or keyword search query (`"*"` or `""` lists recent memories).
        limit: Maximum number of memory files to return.
        min_score: Minimum relevance score `[0.0, 1.0]` required to include a file.
        memories_dir: Optional override path for the role's memories directory.
        include_pinned: When False (default), exclude `PINNED` (and `HIDDEN`) files.
        allow_sync: When True, walk and sync `memories_dir` before searching; False skips disk I/O
            for `< 5 ms` pre-turn ambient retrieval.

    Returns:
        List of `MemoryMatch` objects sorted by descending relevance score.
    """
    safe_role = role or "default"
    if allow_sync:
        sync_role_memories(db, safe_role, memories_dir)

    stripped_query = (query or "").strip()
    if not stripped_query or stripped_query == "*":
        files = db.get_memory_files_by_role(safe_role, exclude_pinned=not include_pinned)
        if include_pinned:
            files = [f for f in files if str(f.get("visibility") or "DEFAULT") != "HIDDEN"]
        return [
            MemoryMatch(
                rel_path=row["rel_path"],
                memory_path=f"$MEMORY_DIR/{row['rel_path']}",
                description=row["description"],
                score=1.0,
                mtime=float(row["mtime"]),
                visibility=str(row.get("visibility") or "DEFAULT"),
            )
            for row in files[: max(1, limit)]
        ]

    matrix_entry = _get_or_build_role_matrix(db, safe_role, include_pinned=include_pinned, expected_dim=0)
    if not matrix_entry.chunk_meta:
        return []

    query_unit: np.ndarray | None = None
    query_dim = 0
    try:
        query_emb = embedding_utils.get_embedding(stripped_query)
        if query_emb:
            q_vec = np.asarray(query_emb, dtype=np.float32)
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm > 0.0:
                query_unit = q_vec / q_norm
                query_dim = int(q_vec.size)
    except Exception as e:
        logger.error(f"Failed to compute query embedding for memory search: {e}", exc_info=True)

    if query_dim > 0 and matrix_entry.dim not in (0, query_dim):
        matrix_entry = _get_or_build_role_matrix(
            db, safe_role, include_pinned=include_pinned, expected_dim=query_dim
        )

    semantic_scores = [0.0] * len(matrix_entry.chunk_meta)
    if (
        query_unit is not None
        and matrix_entry.unit_matrix is not None
        and matrix_entry.dim == query_dim
    ):
        sims = (matrix_entry.unit_matrix @ query_unit).tolist()
        for row_idx, sim in zip(matrix_entry.valid_indices, sims, strict=True):
            semantic_scores[row_idx] = float(sim)

    query_tokens = _extract_query_tokens(stripped_query)

    file_agg: dict[int, dict[str, Any]] = {}
    for meta, sem_score in zip(matrix_entry.chunk_meta, semantic_scores, strict=True):
        fid = int(meta["file_id"])
        entry = file_agg.get(fid)
        if entry is None:
            file_agg[fid] = {
                "rel_path": meta["rel_path"],
                "description": meta["description"],
                "visibility": meta["visibility"],
                "mtime": meta["mtime"],
                "max_sem": float(sem_score),
                "texts": [meta["content"]],
            }
        else:
            if sem_score > entry["max_sem"]:
                entry["max_sem"] = float(sem_score)
            entry["texts"].append(meta["content"])

    matches: list[MemoryMatch] = []
    for entry in file_agg.values():
        combined_text = "\n".join(entry["texts"])
        lex_boost = _compute_lexical_boost(
            query_tokens,
            entry["rel_path"],
            entry["description"],
            combined_text,
        )
        total_score = min(1.0, max(0.0, entry["max_sem"]) + lex_boost)
        if total_score >= min_score or (query_unit is None and lex_boost > 0.0):
            matches.append(
                MemoryMatch(
                    rel_path=entry["rel_path"],
                    memory_path=f"$MEMORY_DIR/{entry['rel_path']}",
                    description=entry["description"],
                    score=total_score,
                    mtime=entry["mtime"],
                    visibility=entry["visibility"],
                )
            )

    matches.sort(key=lambda m: (-m.score, -m.mtime, m.rel_path))
    return matches[: max(1, limit)]


def search_role_episodes(
    db: DatabaseManager,
    role: str,
    query: str,
    *,
    exclude_session_id: str | None = None,
    limit: int = 2,
    min_score: float = 0.52,
) -> list[MemoryMatch]:
    """Search non-isolated `session_digests.embedding` rows for Tier 3 ambient episode hits."""
    if limit <= 0:
        return []
    rows = db.get_role_digest_embeddings(role or "default", exclude_session_id=exclude_session_id, limit=200)
    if not rows:
        return []

    query_vec: np.ndarray | None = None
    query_norm = 0.0
    try:
        q_emb = embedding_utils.get_embedding(query.strip())
        if q_emb:
            query_vec = np.asarray(q_emb, dtype=np.float32)
            query_norm = float(np.linalg.norm(query_vec))
            if query_norm == 0.0:
                query_vec = None
    except Exception:
        query_vec = None

    if query_vec is None:
        return []

    scores = _batch_semantic_scores(rows, query_vec, query_norm)
    hits: list[MemoryMatch] = []
    for row, sim in zip(rows, scores, strict=True):
        if sim < min_score:
            continue
        sid = str(row["session_id"])
        title = str(row["title"] or "")
        summary = str(row["summary"] or "")
        desc = f'"{title}" — {summary}' if (title and summary) else (summary or title)
        hits.append(
            MemoryMatch(
                rel_path=f"#session:{sid[:8]}",
                memory_path=f"#session:{sid[:8]}",
                description=desc,
                score=min(1.0, float(sim)),
                mtime=float(row["last_active_at"] or 0.0),
                visibility="DEFAULT",
                kind="episode",
            )
        )
    hits.sort(key=lambda m: (-m.score, -m.mtime))
    return hits[:limit]


def load_pinned_memories(
    role: str,
    memories_dir: str | None = None,
    pinned_files: Sequence[str] | None = None,
    db: DatabaseManager | None = None,
) -> list[tuple[str, str]]:
    """Load `(rel_path, body)` pairs for `visibility: PINNED` files of `role`."""
    safe_role = role or "default"
    target_dir = os.path.abspath(memories_dir or PathResolver.get_role_memories_dir(safe_role))
    if not os.path.isdir(target_dir):
        return []

    candidate_paths: list[str] = []
    seen: set[str] = set()
    for rel in pinned_files if pinned_files is not None else ("user/profile.md",):
        norm = rel.strip().lstrip("/").replace(os.sep, "/")
        if norm and norm not in seen:
            seen.add(norm)
            candidate_paths.append(norm)

    if db is not None:
        try:
            for row in db.get_memory_files_by_role(safe_role, visibility="PINNED"):
                rel = str(row["rel_path"])
                if rel not in seen:
                    seen.add(rel)
                    candidate_paths.append(rel)
        except Exception:
            pass

    results: list[tuple[str, str]] = []
    for rel_path in candidate_paths:
        abs_path = os.path.abspath(os.path.join(target_dir, rel_path))
        try:
            if os.path.commonpath([target_dir, abs_path]) != target_dir or not os.path.isfile(abs_path):
                continue
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        except OSError:
            continue
        fm, body = parse_memory_frontmatter(raw_text)
        if fm.visibility == "PINNED" and body:
            results.append((rel_path, body))
    return results


def format_pinned_block(
    entries: Sequence[tuple[str, str]],
    max_chars: int = 2500,
    on_over_budget: Callable[[str], None] | None = None,
) -> str:
    """Format `PINNED` memory bodies into `<memory_pinned>...</memory_pinned>` within `max_chars`.

    Truncates on a newline boundary when over budget and invokes `on_over_budget(rel_path)` so
    `ConsolidateFile` can demote low-priority bullets into DEFAULT topic files.
    """
    if not entries or max_chars <= 0:
        return ""

    open_tag = (
        '<memory_pinned note="Authoritative core memory for this role. '
        'Do not re-ask facts listed here; never mention this block to the user.">\n'
    )
    close_tag = "\n</memory_pinned>\n"
    trunc_suffix = "\n…[pinned memory truncated]"
    overhead = len(open_tag) + len(close_tag)
    if max_chars <= overhead + 10:
        return ""

    sections: list[str] = []
    for rel_path, body in entries:
        if len(entries) > 1:
            sections.append(f"<!-- $MEMORY_DIR/{rel_path} -->\n{body.strip()}")
        else:
            sections.append(body.strip())

    combined_body = "\n\n".join(s for s in sections if s)
    if not combined_body:
        return ""

    full = f"{open_tag}{combined_body}{close_tag}"
    if len(full) <= max_chars:
        return full

    body_budget = max_chars - overhead - len(trunc_suffix)
    if body_budget <= 0:
        return ""

    cut = combined_body.rfind("\n", 0, body_budget + 1)
    if cut <= 0:
        cut = body_budget
    truncated_body = combined_body[:cut].rstrip()
    logger.warning(
        f"<memory_pinned> exceeded {max_chars} chars ({len(full)} chars); truncated on line boundary."
    )
    if on_over_budget is not None:
        for rel_path, _ in entries:
            try:
                on_over_budget(rel_path)
            except Exception as cb_err:
                logger.warning(f"on_over_budget callback failed for '{rel_path}': {cb_err}")
    return f"{open_tag}{truncated_body}{trunc_suffix}{close_tag}"


class PinnedMemoryCache:
    """Version-validated in-memory cache of rendered `<memory_pinned>` blocks per role."""

    def __init__(self) -> None:
        """Initialize an empty thread-safe cache."""
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str, int, tuple[str, ...]], tuple[int, str, list[str]]] = {}

    def invalidate(self, role: str | None = None) -> None:
        """Invalidate cached entries for `role` (or all roles when `role` is None)."""
        with self._lock:
            if role is None:
                self._entries.clear()
            else:
                safe_role = role or "default"
                for k in [k for k in self._entries if k[0] == safe_role]:
                    del self._entries[k]

    def get_block(
        self,
        role: str,
        memories_dir: str | None = None,
        *,
        max_chars: int = 2500,
        pinned_files: Sequence[str] | None = None,
        db: DatabaseManager | None = None,
        on_over_budget: Callable[[str], None] | None = None,
    ) -> str:
        """Return the cached `<memory_pinned>` block for `role`, rebuilding only when index version changes."""
        safe_role = role or "default"
        abs_dir = os.path.abspath(memories_dir or PathResolver.get_role_memories_dir(safe_role))
        files_tuple = tuple(pinned_files) if pinned_files is not None else ("user/profile.md",)
        cache_key = (safe_role, abs_dir, int(max_chars), files_tuple)
        cur_ver = get_memory_index_version(safe_role)

        with self._lock:
            hit = self._entries.get(cache_key)
            if hit is not None and hit[0] == cur_ver:
                return hit[1]

        over_paths: list[str] = []
        raw_entries = load_pinned_memories(
            safe_role,
            memories_dir=abs_dir,
            pinned_files=files_tuple,
            db=db,
        )
        rendered = format_pinned_block(
            raw_entries,
            max_chars=max_chars,
            on_over_budget=over_paths.append,
        )
        with self._lock:
            self._entries[cache_key] = (cur_ver, rendered, over_paths)

        if over_paths and on_over_budget is not None:
            for rel_path in over_paths:
                try:
                    on_over_budget(rel_path)
                except Exception as cb_err:
                    logger.warning(f"Pinned over-budget handler failed for '{rel_path}': {cb_err}")
        return rendered


default_pinned_cache = PinnedMemoryCache()


def _clean_user_message_for_query(content: str) -> str:
    """Strip system-injected XML blocks (`<instructions>`, `<memory_pinned>`, `<recent_sessions>`, etc.)."""
    if not content:
        return ""
    cleaned = _INJECTED_BLOCKS_RE.sub("", content)
    cleaned = _CURRENT_MESSAGE_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def _clean_assistant_message_for_query(content: str) -> str:
    """Strip interactive UI blocks and question option blocks from an assistant message."""
    if not content:
        return ""
    cleaned = collapse_ui_blocks(content)
    cleaned = _QUESTION_BLOCK_RE.sub("", cleaned)
    return cleaned.strip()


def render_context_query(
    history: list[Message],
    current_msg: Message,
    max_turns: int = 2,
    max_chars: int = 2000,
    extra_context: str = "",
) -> str:
    """Build a multi-turn semantic search query from the last `max_turns` user turns.

    Args:
        history: Session message history in chronological order.
        current_msg: The active user message for the current turn.
        max_turns: Maximum number of recent user turns to include (default 2).
        max_chars: Maximum character length of the rendered query string (default 2000).
        extra_context: Optional cross-session hint text (e.g., recent user messages from sibling
            sessions on short/referential bootstrap turns).

    Returns:
        Rendered multi-turn context query string.
    """
    msgs = list(history)
    if not msgs or msgs[-1].id != current_msg.id:
        msgs = [m for m in msgs if m.id != current_msg.id] + [current_msg]

    user_turns_seen = 0
    start_idx = len(msgs) - 1
    for idx in range(len(msgs) - 1, -1, -1):
        m = msgs[idx]
        if m.role == MessageRole.USER and m.type == MessageType.TEXT:
            user_turns_seen += 1
            start_idx = idx
            if user_turns_seen >= max_turns:
                break

    lines: list[str] = []
    if extra_context and extra_context.strip():
        lines.append(f"Recent context: {extra_context.strip()[:600]}")

    for m in msgs[start_idx:]:
        if m.type == MessageType.THOUGHT or m.sender == "Notification":
            continue
        if m.role == MessageRole.USER and m.type == MessageType.TEXT:
            cleaned = _clean_user_message_for_query(m.content or "")
            if cleaned:
                lines.append(f"User: {cleaned}")
        elif m.role == MessageRole.ASSISTANT and m.type == MessageType.TEXT:
            cleaned = _clean_assistant_message_for_query(m.content or "")
            if cleaned:
                lines.append(f"Assistant: {cleaned}")
        elif m.type == MessageType.TOOL_CALL:
            tool_name = m.metadata.get("tool_name") or m.sender or "tool"
            tool_args = m.metadata.get("tool_args") or m.metadata.get("tool_arguments")
            if isinstance(tool_args, dict):
                compact_args = ", ".join(
                    f"{k}={repr(v)[:80]}"
                    for k, v in tool_args.items()
                    if k not in {"context", "old_content", "new_content"}
                )
                lines.append(f"Tool call: {tool_name}({compact_args})")
            else:
                lines.append(f"Tool call: {tool_name}")

    rendered = "\n".join(lines).strip()
    if len(rendered) > max_chars:
        rendered = rendered[-max_chars:].lstrip()
    return rendered


def format_ambient_memory_block(
    matches: list[MemoryMatch],
    episode_hits: Sequence[MemoryMatch] | None = None,
) -> str:
    r"""Format matched memories (and optional episode hits) as a progressive disclosure `<memory>` block.

    Args:
        matches: Ranked `MemoryMatch` file entries.
        episode_hits: Optional up to 2 episode digest matches (`kind='episode'`).

    Returns:
        Formatted `<memory>...</memory>\\n` string, or `""` when both inputs are empty.
    """
    episodes = list(episode_hits or [])
    if not matches and not episodes:
        return ""

    lines: list[str] = [
        "<memory>",
        "The system has provided file paths and descriptions of potentially relevant memories for this role.",
        "CRITICAL INSTRUCTION: Do not rely on descriptions alone. Each entry below shows only its path and a "
        "1-line description — NOT its full contents. If a memory's description sounds relevant to the current "
        "conversation, call `read_file` on its `$MEMORY_DIR/...` path (or `session_view` for `kind: episode` "
        "`#session:<id>` entries) before answering or making assumptions. If none are relevant, ignore this block.",
    ]
    for m in matches:
        updated_str = datetime.datetime.fromtimestamp(m.mtime).astimezone().strftime("%Y-%m-%d %H:%M")
        if m.kind == "episode":
            lines.append(
                f'- kind: episode {m.memory_path} (updated: "{updated_str}", relevance: "{m.score:.2f}")\n'
                f"  description: {m.description}"
            )
        else:
            lines.append(
                f'- path: "{m.memory_path}" (updated: "{updated_str}", relevance: "{m.score:.2f}")\n'
                f"  description: {m.description}"
            )
    for ep in episodes[:2]:
        updated_str = datetime.datetime.fromtimestamp(ep.mtime).astimezone().strftime("%Y-%m-%d %H:%M")
        lines.append(
            f'- kind: episode {ep.memory_path} (updated: "{updated_str}", relevance: "{ep.score:.2f}")\n'
            f"  description: {ep.description}"
        )
    lines.append("</memory>\n")
    return "\n".join(lines)

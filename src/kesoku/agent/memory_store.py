"""Markdown-based role memory indexing, progressive disclosure, and hybrid search for Kesoku."""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from kesoku.constants import MessageRole, MessageType
from kesoku.db.manager import SEARCH_SCORE_BATCH_SIZE, DatabaseManager, _batch_semantic_scores
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
    r"<(?:instructions|context\b[^>]*|facts|memory|background_context\b[^>]*|relevant_historical_context)>.*?"
    r"</(?:instructions|context|facts|memory|background_context|relevant_historical_context)>\s*",
    re.DOTALL,
)
_CURRENT_MESSAGE_TAG_RE = re.compile(r"</?current_message\b[^>]*>\s*")
# Tokenizes ASCII identifiers/words (>=2 chars) and CJK character runs (>=2 chars) for lexical boosting.
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_.\-/]{2,}")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]{2,}")

MAX_LEXICAL_BOOST = 0.10


@dataclass(frozen=True)
class MemoryMatch:
    """A ranked role memory file returned by semantic + lexical search."""

    rel_path: str
    memory_path: str
    description: str
    score: float
    mtime: float


def parse_memory_frontmatter(text: str) -> tuple[str, str]:
    """Extract the single-line `description` and Markdown body from a memory file.

    Parses the leading YAML frontmatter block (`--- ... ---`) if present. When the frontmatter
    is missing or does not define `description`, falls back to the first `# Heading` or the
    first non-empty line of the body so hand-written Markdown files are still discoverable.

    Args:
        text: Raw UTF-8 content of a `.md` memory file.

    Returns:
        A tuple of `(description, markdown_body)`.
    """
    if not text:
        return "", ""

    description = ""
    body = text
    match = _FRONTMATTER_RE.match(text)
    if match:
        yaml_block = match.group(1)
        body = text[match.end() :]
        description = _extract_yaml_description(yaml_block)

    if not description:
        description = _fallback_description_from_body(body)

    # Collapse whitespace so descriptions always render cleanly on one line.
    clean_desc = " ".join(description.split()).strip()
    if len(clean_desc) > 240:
        clean_desc = clean_desc[:239].rstrip() + "…"
    return clean_desc, body.strip()


def _extract_yaml_description(yaml_block: str) -> str:
    """Parse the `description:` field from a simple YAML frontmatter block without external deps."""
    lines = yaml_block.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("description:"):
            raw_val = stripped[len("description:") :].strip()
            # Multi-line YAML block scalar (`|` or `>`)
            if raw_val in {"|", ">", "|-", ">-"}:
                collected: list[str] = []
                for sub in lines[i + 1 :]:
                    if sub and not sub[0].isspace():
                        break
                    if sub.strip():
                        collected.append(sub.strip())
                return " ".join(collected)
            # Strip matching surrounding quotes
            if len(raw_val) >= 2 and raw_val[0] == raw_val[-1] and raw_val[0] in {'"', "'"}:
                raw_val = raw_val[1:-1]
            return raw_val.strip()
    return ""


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


def chunk_memory_markdown(
    rel_path: str,
    description: str,
    body: str,
    max_chunk_chars: int = 600,
) -> list[str]:
    r"""Split a Markdown memory into multi-granularity chunks with ancestor heading breadcrumbs.

    Mirrors Jetski's structural Markdown chunking:
    - Chunk 0 embeds the file path and frontmatter `description` directly (`# <rel_path>\\n<description>`).
    - Subsequent chunks split the body by Markdown sections (`# H1` .. `###### H6`), automatically
      prepending the file path/description and all active ancestor headings to each section chunk
      so no chunk loses its subject context.

    Args:
        rel_path: Path of the memory file relative to `$MEMORY_DIR`.
        description: Single-line description of the memory file.
        body: Markdown body (without frontmatter).
        max_chunk_chars: Maximum character budget per body chunk before paragraph splitting.

    Returns:
        List of non-empty text chunks ready for embedding.
    """
    header_tag = f"# {rel_path}\n{description}".strip() if description else f"# {rel_path}"
    chunks: list[str] = [header_tag]

    stripped_body = body.strip()
    if not stripped_body:
        return chunks

    file_context_line = f"# {rel_path} ({description})" if description else f"# {rel_path}"

    # Parse sections while respecting fenced code blocks so `# comments` inside code are not treated as headings.
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
                # Flush accumulated lines under the previous heading hierarchy.
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append((_current_breadcrumbs(), section_text.splitlines()))
                current_lines = []

                level = len(heading_match.group(1))
                # Clear any existing headings at the same or deeper level.
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

    Compares disk files against `memory_files` by `mtime` and SHA-256 `content_hash`:
    - Deleted files are removed from `memory_files` (cascading to `memory_chunks`).
    - Unchanged files (`mtime` or `content_hash` match) are skipped without recomputing embeddings.
    - New or modified files are parsed, chunked with breadcrumbs, embedded, and upserted.

    Args:
        db: Synchronous `DatabaseManager` instance.
        role: Role persona identifier (e.g., `'default'`, `'coder'`, `'writer'`).
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
        return 0

    # Discover all non-hidden .md files under target_dir.
    disk_files: dict[str, str] = {}
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for fname in sorted(files):
            if fname.startswith(".") or not fname.lower().endswith(".md"):
                continue
            abs_path = os.path.join(root, fname)
            rel_path = os.path.relpath(abs_path, target_dir).replace(os.sep, "/")
            disk_files[rel_path] = abs_path

    # Prune deleted files from SQLite index.
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
            # Empty file: remove from index if previously indexed.
            if prev is not None:
                db.delete_memory_files_by_ids([int(prev["id"])])
            continue

        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        if prev is not None and not force and prev.get("content_hash") == content_hash:
            # Content unchanged despite mtime touch; update mtime in-place without re-embedding.
            continue

        description, body = parse_memory_frontmatter(raw_text)
        chunk_texts = chunk_memory_markdown(rel_path, description, body)

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
            description=description,
            content_hash=content_hash,
            mtime=mtime,
            updated_at=now,
            chunks=chunk_tuples,
        )
        updated_count += 1

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
            # Add sliding 2-char bigrams for longer CJK phrases so substrings match without a tokenizer.
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


def search_role_memories(
    db: DatabaseManager,
    role: str,
    query: str,
    limit: int = 5,
    min_score: float = 0.50,
    memories_dir: str | None = None,
) -> list[MemoryMatch]:
    """Search a role's Markdown memories using Jetski's `MAX(chunk_similarity)` + lexical boost.

    Synchronizes the role's memory directory incrementally before searching so recently created,
    edited, or deleted `.md` files are always reflected immediately.

    Args:
        db: Synchronous `DatabaseManager` instance.
        role: Role persona identifier.
        query: Natural language or keyword search query (`"*"` or `""` lists recent memories).
        limit: Maximum number of memory files to return.
        min_score: Minimum relevance score `[0.0, 1.0]` required to include a file.
        memories_dir: Optional override path for the role's memories directory.

    Returns:
        List of `MemoryMatch` objects sorted by descending relevance score.
    """
    safe_role = role or "default"
    sync_role_memories(db, safe_role, memories_dir)

    stripped_query = (query or "").strip()
    if not stripped_query or stripped_query == "*":
        files = db.get_memory_files_by_role(safe_role)
        return [
            MemoryMatch(
                rel_path=row["rel_path"],
                memory_path=f"$MEMORY_DIR/{row['rel_path']}",
                description=row["description"],
                score=1.0,
                mtime=float(row["mtime"]),
            )
            for row in files[: max(1, limit)]
        ]

    chunk_rows = db.get_memory_chunks_for_role(safe_role)
    if not chunk_rows:
        return []

    query_vec: np.ndarray | None = None
    query_norm = 0.0
    try:
        query_emb = embedding_utils.get_embedding(stripped_query)
        if query_emb:
            query_vec = np.asarray(query_emb, dtype=np.float32)
            query_norm = float(np.linalg.norm(query_vec))
            if query_norm == 0.0:
                query_vec = None
    except Exception as e:
        logger.error(f"Failed to compute query embedding for memory search: {e}", exc_info=True)

    # Score all chunks in batches and aggregate per file via MAX(semantic_score).
    semantic_scores: list[float] = []
    for i in range(0, len(chunk_rows), SEARCH_SCORE_BATCH_SIZE):
        batch = chunk_rows[i : i + SEARCH_SCORE_BATCH_SIZE]
        semantic_scores.extend(_batch_semantic_scores(batch, query_vec, query_norm))

    query_tokens = _extract_query_tokens(stripped_query)

    # Group by file_id -> track max semantic score and concatenate chunk text for lexical check.
    file_agg: dict[int, dict[str, Any]] = {}
    for row, sem_score in zip(chunk_rows, semantic_scores, strict=True):
        fid = int(row["file_id"])
        entry = file_agg.get(fid)
        if entry is None:
            file_agg[fid] = {
                "rel_path": str(row["rel_path"]),
                "description": str(row["description"] or ""),
                "mtime": float(row["mtime"]),
                "max_sem": float(sem_score),
                "texts": [str(row["content"] or "")],
            }
        else:
            if sem_score > entry["max_sem"]:
                entry["max_sem"] = float(sem_score)
            entry["texts"].append(str(row["content"] or ""))

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
        # Allow exact substring hits in path/description to surface even if embedding failed.
        if total_score >= min_score or (
            query_vec is None and lex_boost > 0.0
        ):
            matches.append(
                MemoryMatch(
                    rel_path=entry["rel_path"],
                    memory_path=f"$MEMORY_DIR/{entry['rel_path']}",
                    description=entry["description"],
                    score=total_score,
                    mtime=entry["mtime"],
                )
            )

    matches.sort(key=lambda m: (-m.score, -m.mtime, m.rel_path))
    return matches[: max(1, limit)]


def _clean_user_message_for_query(content: str) -> str:
    """Strip system-injected XML blocks (`<instructions>`, `<context>`, `<memory>`, etc.) from a user message."""
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
) -> str:
    """Build a multi-turn semantic search query from the last `max_turns` user turns.

    Mirrors Jetski's `RenderContextQuery` (`hook.go`): includes recent `User:` messages,
    `Assistant:` text replies (with UI/option blocks stripped), and compact `Tool call:`
    summaries while omitting bulky tool outputs and internal thought blocks.

    Args:
        history: Session message history in chronological order.
        current_msg: The active user message for the current turn.
        max_turns: Maximum number of recent user turns to include (default 2).
        max_chars: Maximum character length of the rendered query string (default 2000).

    Returns:
        Rendered multi-turn context query string.
    """
    # Ensure current_msg is at the tail of our slice.
    msgs = list(history)
    if not msgs or msgs[-1].id != current_msg.id:
        msgs = [m for m in msgs if m.id != current_msg.id] + [current_msg]

    # Walk backwards to find the start index of the N-th most recent user message.
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
            tool_args = m.metadata.get("tool_args")
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


def format_ambient_memory_block(matches: list[MemoryMatch]) -> str:
    r"""Format matched memories as a progressive disclosure `<memory>` block for `TurnExecutor`.

    Only file paths, update timestamps, relevance scores, and 1-line descriptions are included—never
    the full Markdown bodies—so the agent calls `read_file` on demand when a memory is relevant.

    Args:
        matches: Ranked `MemoryMatch` entries.

    Returns:
        Formatted `<memory>...</memory>\\n` string, or `""` when `matches` is empty.
    """
    if not matches:
        return ""

    lines: list[str] = [
        "<memory>",
        "The system has provided file paths and descriptions of potentially relevant memories for this role.",
        "CRITICAL INSTRUCTION: Do not rely on descriptions alone. Each entry below shows only its path and a "
        "1-line description — NOT its full contents. If a memory's description sounds relevant to the current "
        "conversation, call `read_file` on its `$MEMORY_DIR/...` path before answering or making assumptions. "
        "If none are relevant, ignore this block.",
    ]
    for m in matches:
        updated_str = datetime.datetime.fromtimestamp(m.mtime).astimezone().strftime("%Y-%m-%d %H:%M")
        lines.append(
            f'- path: "{m.memory_path}" (updated: "{updated_str}", relevance: "{m.score:.2f}")\n'
            f"  description: {m.description}"
        )
    lines.append("</memory>\n")
    return "\n".join(lines)

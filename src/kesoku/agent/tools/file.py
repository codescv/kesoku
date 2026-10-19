"""File management tools for Kesoku AI Agent."""

import difflib
import os
import re

from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.utils.path import PathResolver

# `$STAGING_DIR`, `$AWD`, and `$MEMORY_DIR` (plain or braced) placeholders used in prompts/skills.
_PATH_VAR_RE = re.compile(r"\$(?:\{(STAGING_DIR|AWD|MEMORY_DIR)\}|(STAGING_DIR|AWD|MEMORY_DIR)\b)")

# `read_file` line-number prefixes as models copy them into old_content: `  12: text`, `12:` (empty line whose
# trailing space was stripped) or the `cat -n` style `    12\ttext`.
_LINE_PREFIX_RE = re.compile(r"^\s*\d+(?::(?: |$)|\t)")

# Mirrors `shell.MAX_TOOL_OUTPUT_LENGTH`; kept local because importing shell here would register its tools before
# the file tools and change the tool order in the LLM schema.
READ_FILE_MAX_CHARS = 30000
READ_FILE_DEFAULT_MAX_LINES = 500
READ_FILE_HARD_MAX_LINES = 2000
READ_FILE_MAX_LINE_CHARS = 2000

_BINARY_SNIFF_BYTES = 8192
_DIR_LISTING_MAX = 50
_SUGGEST_MAX = 3
_SUGGEST_MAX_DEPTH = 6
_SUGGEST_MAX_ENTRIES = 5000
_SUGGEST_SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "sessions"}
_HINT_LINE_MAX_CHARS = 200
_MATCH_LIST_MAX = 10
_MATCH_CONTEXT_MAX = 3
_MATCH_CONTEXT_LINES = 2


def _resolve_role_sync(context: ToolContext | None) -> str:
    """Resolve the active role synchronously from `ToolContext` (defaults to `'default'`)."""
    if not context or not context.gateway or not context.session_id:
        return "default"
    try:
        db = getattr(context.gateway, "db", None)
        sync_db = getattr(db, "_sync", getattr(db, "sync_db", db))
        if sync_db is None:
            return "default"
        sess = sync_db.get_session(context.session_id)
        if sess and sess.role_name and sess.role_name != "default":
            return sess.role_name
        mapping = sync_db.get_channel_by_session(context.session_id)
        if mapping:
            chatbot_id, channel_id = mapping
            return sync_db.get_channel_role_with_inheritance(chatbot_id, channel_id, context.session_id)
        if sess and sess.role_name:
            return sess.role_name
    except Exception:
        pass
    return "default"


def expand_path_variables(file_name: str, context: ToolContext | None = None) -> str:
    """Expand `$STAGING_DIR`, `$AWD`, and `$MEMORY_DIR` placeholders (plain or braced) in a tool file path.

    Prompts name files like `$STAGING_DIR/notes.md` or `$MEMORY_DIR/topics/foo.md`; without expansion such a path
    would silently be created literally under the AWD. `$STAGING_DIR` resolves from the session workspace in
    `context` and is left untouched when no session context is available.
    """
    if not file_name or "$" not in file_name:
        return file_name

    def _replace(match: re.Match) -> str:
        var = match.group(1) or match.group(2)
        if var == "AWD":
            return PathResolver.get_awd()
        if var == "MEMORY_DIR":
            return PathResolver.get_role_memories_dir(_resolve_role_sync(context))
        workspace = (context.session_workspace or context.session_id) if context else None
        return PathResolver.get_session_staging_dir(workspace) if workspace else match.group(0)

    return _PATH_VAR_RE.sub(_replace, file_name)


# ---------------------------------------------------------------------------------------------------------------
# Pure helpers (no filesystem access unless stated)
# ---------------------------------------------------------------------------------------------------------------


def _split_lines(text: str) -> list[str]:
    """Split text into lines on LF only, so numbering agrees with the LF-count offsets used by update_file.

    A trailing newline does not produce an extra empty line and a trailing CR (CRLF files) is dropped.
    """
    if not text:
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def _count_lines(text: str) -> int:
    """Number of lines in `text` as shown by `read_file`."""
    return len(_split_lines(text))


def _line_of(text: str, idx: int) -> int:
    """1-based line number of character offset `idx` in `text`."""
    return text.count("\n", 0, idx) + 1


def _block_span(text: str, idx: int, length: int) -> tuple[int, int]:
    """1-based inclusive line span of `text[idx:idx + length]` (a single line for an empty block)."""
    start = _line_of(text, idx)
    block = text[idx : idx + length]
    if block.endswith("\n"):
        block = block[:-1]
    return start, start + block.count("\n")


def _fmt_span(span: tuple[int, int]) -> str:
    """Render a line span as `A` or `A-B`."""
    start, end = span
    return str(start) if start == end else f"{start}-{end}"


def _find_all(content: str, needle: str) -> list[int]:
    """Start offsets of all non-overlapping occurrences of `needle` (same semantics as `str.count`)."""
    offsets: list[int] = []
    idx = content.find(needle)
    while idx != -1:
        offsets.append(idx)
        idx = content.find(needle, idx + len(needle))
    return offsets


def _strip_line_number_prefixes(text: str) -> str | None:
    """Remove `read_file` line-number prefixes from `old_content` if every non-blank line carries one.

    Returns the stripped text, or None when at least one non-blank line has no prefix (or there is nothing to
    strip). Blank lines inside the block are kept as-is.
    """
    lines = text.split("\n")
    non_blank = [line for line in lines if line.strip()]
    if not non_blank or not all(_LINE_PREFIX_RE.match(line) for line in non_blank):
        return None
    return "\n".join(_LINE_PREFIX_RE.sub("", line, count=1) if line.strip() else line for line in lines)


def _normalize_ws(line: str) -> str:
    """Whitespace-normalised form of a line: stripped, internal whitespace runs collapsed to one space."""
    return " ".join(line.split())


def _find_fuzzy_block(content_lines: list[str], old_lines: list[str]) -> tuple[int, int] | None:
    """Find `old_lines` in `content_lines` ignoring whitespace differences.

    Leading/trailing blank lines of `old_lines` are ignored. Returns the 1-based inclusive `(start, end)` span of
    the first match, or None.
    """
    probe = [_normalize_ws(line) for line in old_lines]
    while probe and not probe[0]:
        probe.pop(0)
    while probe and not probe[-1]:
        probe.pop()
    if not probe:
        return None
    normalized = [_normalize_ws(line) for line in content_lines]
    n = len(probe)
    for i in range(len(normalized) - n + 1):
        if normalized[i] == probe[0] and normalized[i : i + n] == probe:
            return i + 1, i + n
    return None


def _closest_lines(content_lines: list[str], probe_line: str, n: int = 3) -> list[tuple[int, str]]:
    """Up to `n` file lines most similar to `probe_line` (difflib ratio >= 0.6), best first, with line numbers."""
    probe = probe_line.strip()
    if not probe:
        return []
    matcher = difflib.SequenceMatcher()
    matcher.set_seq2(probe)
    scored: list[tuple[float, int, str]] = []
    for lineno, line in enumerate(content_lines, start=1):
        candidate = line.strip()
        if not candidate:
            continue
        matcher.set_seq1(candidate)
        # Same cheap-to-expensive filter cascade as difflib.get_close_matches.
        if matcher.real_quick_ratio() >= 0.6 and matcher.quick_ratio() >= 0.6:
            score = matcher.ratio()
            if score >= 0.6:
                scored.append((score, lineno, line))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(lineno, line) for _, lineno, line in scored[:n]]


def _clip(line: str, limit: int = _HINT_LINE_MAX_CHARS) -> str:
    """Shorten a line for display in a hint."""
    return line if len(line) <= limit else line[:limit] + "…"


def _numbered(lines: list[str], first_lineno: int) -> str:
    """Render lines with right-aligned `N: ` prefixes, like `read_file`."""
    width = len(str(first_lineno + len(lines) - 1))
    return "\n".join(f"{lineno:>{width}}: {_clip(line)}" for lineno, line in enumerate(lines, start=first_lineno))


def _first_non_blank(text: str) -> str:
    """First line of `text` that is not blank ('' if none)."""
    return next((line for line in text.split("\n") if line.strip()), "")


def _not_found_message(file_name: str, content: str, old_content: str, stripped: str | None) -> str:
    """Build the recovery hint for an `old_content` that does not occur in the file."""
    content_lines = _split_lines(content)
    probe_text = stripped if stripped is not None else old_content

    # 1) Same block modulo whitespace/indentation: show the exact lines so the model can copy them.
    candidates = [old_content] if stripped is None else [stripped, old_content]
    for candidate in candidates:
        span = _find_fuzzy_block(content_lines, candidate.split("\n"))
        if span:
            start, end = span
            exact = "\n".join(content_lines[start - 1 : end])
            return (
                f"Error: old_content not found in file '{file_name}'.\n"
                f"A match that differs only in whitespace/indentation exists at lines {start}-{end}:\n"
                f"```\n{exact}\n```\n"
                "Retry with exactly these lines as old_content."
            )

    # 2) Otherwise point at the lines closest to the first line of old_content.
    closest = _closest_lines(content_lines, _first_non_blank(probe_text))
    if not closest:
        return (
            f"Error: old_content not found in file '{file_name}' and no similar lines exist "
            f"(file has {len(content_lines)} lines). Use read_file(\"{file_name}\") to view the current content."
        )
    listing = "\n".join(f"  {lineno}: {_clip(line)}" for lineno, line in closest)
    best = closest[0][0]
    return (
        f"Error: old_content not found in file '{file_name}'. Closest lines to the first line of old_content:\n"
        f"{listing}\n"
        f"Use read_file(\"{file_name}\", start_line={max(1, best - 5)}, end_line={best + 15}) to copy the exact "
        "text (without the 'N: ' prefixes)."
    )


def _too_many_matches_message(file_name: str, content: str, needle: str, offsets: list[int]) -> str:
    """Build the error for an `old_content` matching more than 2 places, with line numbers and context."""
    content_lines = _split_lines(content)
    spans = [_block_span(content, idx, len(needle)) for idx in offsets]
    starts = ", ".join(str(start) for start, _ in spans[:_MATCH_LIST_MAX])
    if len(spans) > _MATCH_LIST_MAX:
        starts += ", …"
    parts = [
        f"Error: old_content matches {len(offsets)} places in '{file_name}' (lines {starts}; max allowed is 2). "
        f"Include more surrounding lines so the block is unique. "
        f"Context of the first {min(len(spans), _MATCH_CONTEXT_MAX)} matches:"
    ]
    for start, end in spans[:_MATCH_CONTEXT_MAX]:
        lo = max(1, start - _MATCH_CONTEXT_LINES)
        hi = min(len(content_lines), end + _MATCH_CONTEXT_LINES)
        parts.append(f"--- match at line {_fmt_span((start, end))}:\n{_numbered(content_lines[lo - 1 : hi], lo)}")
    return "\n".join(parts)


def _suggest_paths(resolved_missing: str, awd: str) -> list[str]:
    """Suggest up to 3 existing AWD-relative paths for a missing file (touches the filesystem, cost-bounded).

    Candidates are (a) the same relative path under each first-level AWD directory (the common missing `repo/`
    prefix) and (b) files with the same basename found by a bounded walk of the AWD.
    """
    if not awd or not os.path.isdir(awd):
        return []
    suggestions: list[str] = []

    def _add(path: str) -> None:
        rel = os.path.relpath(path, awd)
        if rel not in suggestions and os.path.abspath(path) != os.path.abspath(resolved_missing):
            suggestions.append(rel)

    # (a) Missing first-level directory prefix.
    rel_missing = os.path.relpath(resolved_missing, awd)
    if not rel_missing.startswith(".."):
        try:
            first_level = sorted(e.name for e in os.scandir(awd) if e.is_dir() and e.name not in _SUGGEST_SKIP_DIRS)
        except OSError:
            first_level = []
        for name in first_level:
            candidate = os.path.join(awd, name, rel_missing)
            if os.path.isfile(candidate):
                _add(candidate)
            if len(suggestions) >= _SUGGEST_MAX:
                return suggestions

    # (b) Same basename anywhere in the AWD (bounded depth and entry count).
    basename = os.path.basename(resolved_missing)
    entries = 0
    base_depth = awd.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(awd):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        dirs[:] = sorted(d for d in dirs if d not in _SUGGEST_SKIP_DIRS) if depth < _SUGGEST_MAX_DEPTH else []
        entries += len(dirs) + len(files)
        if basename in files:
            _add(os.path.join(root, basename))
            if len(suggestions) >= _SUGGEST_MAX:
                break
        if entries >= _SUGGEST_MAX_ENTRIES:
            break
    return suggestions[:_SUGGEST_MAX]


def _suggestion_text(resolved_missing: str) -> str:
    """` Did you mean: a, b?` for a missing path, or '' when nothing similar exists."""
    try:
        suggestions = _suggest_paths(resolved_missing, PathResolver.get_awd())
    except OSError:
        return ""
    return f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""


# ---------------------------------------------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------------------------------------------


def _create_file(file_name: str, resolved_path: str, new_content: str) -> str:
    """Create-mode branch of `update_file` (old_content omitted)."""
    if os.path.isdir(resolved_path):
        return f"Error: '{file_name}' is a directory."
    if os.path.exists(resolved_path):
        try:
            with open(resolved_path, encoding="utf-8", errors="replace") as f:
                n_lines: int | str = _count_lines(f.read())
        except OSError:
            n_lines = "?"
        return (
            f"Error: File '{file_name}' already exists ({n_lines} lines). To edit it, pass the exact lines to "
            "replace as old_content (use read_file to view them). To replace the whole file, pass its full "
            "current content as old_content."
        )
    if new_content == "":
        return "Error: new_content is empty."
    try:
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except OSError as e:
        return f"Error creating file '{file_name}': {e}"
    return f"Success: Created file '{file_name}' ({_count_lines(new_content)} lines)."


@default_registry.register
def update_file(
    file_name: str,
    old_content: str | None = None,
    new_content: str | None = None,
    context: ToolContext | None = None,
) -> str:
    """Create a file (omit `old_content`) or replace an exact, unique block of text in an existing file.

    Copy `old_content` verbatim from `read_file` output **without** the `N: ` line-number prefixes. Use a short
    block (3-15 lines) that occurs only once; if it occurs twice, both occurrences are replaced, more than twice is
    rejected. Creating a file that already exists is rejected.

    Args:
        file_name: Path to the file (relative to AWD or absolute). `$STAGING_DIR` and `$AWD` are expanded.
        old_content: The exact text to replace. Omit it only when creating a new file.
        new_content: The replacement text, or the full content of a new file. Required; pass "" to delete
            old_content. Keep each call under 4000 characters (split larger content into several calls).
        context: Optional tool execution context.

    Returns:
        A success message (with the affected line numbers) or an error message with recovery hints.
    """
    if new_content is None:
        return (
            "Error: new_content is required (if your call was cut off, split the content into ≤4000-char chunks)."
        )

    resolved_path = PathResolver.resolve(expand_path_variables(file_name, context))

    if old_content is None or old_content == "":
        return _create_file(file_name, resolved_path, new_content)

    if not os.path.exists(resolved_path):
        return (
            f"Error: File '{file_name}' does not exist.{_suggestion_text(resolved_path)} "
            "If you meant to create it, omit old_content."
        )

    try:
        with open(resolved_path, encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError) as e:
        return f"Error reading file '{file_name}': {e}"

    needle = old_content
    note = ""
    offsets = _find_all(content, needle)
    stripped = _strip_line_number_prefixes(old_content) if not offsets else None
    if not offsets and stripped:
        # The model pasted read_file output including the `N: ` prefixes; retry without them.
        stripped_offsets = _find_all(content, stripped)
        if stripped_offsets:
            needle, offsets = stripped, stripped_offsets
            note = " (note: stripped read_file line-number prefixes from old_content)"

    if not offsets:
        return _not_found_message(file_name, content, old_content, stripped)
    if len(offsets) > 2:
        return _too_many_matches_message(file_name, content, needle, offsets)

    new_file_content = content.replace(needle, new_content)
    try:
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(new_file_content)
    except OSError as e:
        return f"Error writing to file '{file_name}': {e}"

    # Report where the new text now lives (offsets shift by the length delta of earlier replacements).
    delta = len(new_content) - len(needle)
    spans = [_block_span(new_file_content, idx + i * delta, len(new_content)) for i, idx in enumerate(offsets)]
    total = _count_lines(new_file_content)
    if len(spans) == 1:
        label = "line" if spans[0][0] == spans[0][1] else "lines"
        return (
            f"Success: Updated file '{file_name}' (1 replacement at {label} {_fmt_span(spans[0])}; "
            f"file now has {total} lines).{note}"
        )
    where = f"{_fmt_span(spans[0])} and {_fmt_span(spans[1])}"
    return (
        f"Success: Updated file '{file_name}' (2 replacements; file now has {total} lines).{note} "
        f"⚠️ replaced 2 occurrences at lines {where}; if you meant one, include more context."
    )


def _directory_listing(file_name: str, resolved_path: str) -> str:
    """Error message for `read_file` on a directory, listing up to 50 sorted entries."""
    try:
        entries = sorted(e.name + ("/" if e.is_dir() else "") for e in os.scandir(resolved_path))
    except OSError as e:
        return f"Error: '{file_name}' is a directory (cannot list it: {e})."
    shown = ", ".join(entries[:_DIR_LISTING_MAX]) or "(empty)"
    if len(entries) > _DIR_LISTING_MAX:
        shown += f", … ({len(entries) - _DIR_LISTING_MAX} more)"
    return f"Error: '{file_name}' is a directory. Entries: {shown}"


def _render_line(lineno: int, width: int, line: str) -> str:
    """Render one numbered `read_file` line, truncating overly long (e.g. minified) lines."""
    if len(line) > READ_FILE_MAX_LINE_CHARS:
        line = f"{line[:READ_FILE_MAX_LINE_CHARS]}… [line truncated, {len(line)} chars]"
    return f"{lineno:>{width}}: {line}"


@default_registry.register
def read_file(
    file_name: str,
    start_line: int = 1,
    end_line: int | None = None,
    max_lines: int = READ_FILE_DEFAULT_MAX_LINES,
    context: ToolContext | None = None,
) -> str:
    """Read a text file with 1-based line numbers (use this instead of cat/head/sed -n).

    Shows lines start_line..end_line (inclusive), at most max_lines lines (hard cap 2000) and 30000 characters per
    call; the footer tells you how to continue. Paths are relative to the AWD; `$STAGING_DIR` / `$AWD` expand.
    When copying text into update_file's old_content, drop the `N: ` prefixes.

    Args:
        file_name: Path to the file (relative to AWD or absolute).
        start_line: First line to show (1-based).
        end_line: Last line to show (inclusive); defaults to start_line + max_lines - 1.
        max_lines: Maximum number of lines to show (default 500, hard cap 2000).
        context: Optional tool execution context.

    Returns:
        The numbered lines with a header and a continuation footer, or an error message.
    """
    resolved_path = PathResolver.resolve(expand_path_variables(file_name, context))
    if not os.path.exists(resolved_path):
        return f"Error: File '{file_name}' not found.{_suggestion_text(resolved_path)}"
    if os.path.isdir(resolved_path):
        return _directory_listing(file_name, resolved_path)

    try:
        with open(resolved_path, "rb") as f:
            data = f.read()
    except OSError as e:
        return f"Error reading file '{file_name}': {e}"
    if b"\x00" in data[:_BINARY_SNIFF_BYTES]:
        return (
            f"Error: '{file_name}' is a binary file ({len(data)} bytes); use run_shell_command (e.g. xxd | head) "
            "if you really need its contents."
        )

    text = data.decode("utf-8", errors="replace")
    lines = _split_lines(text)
    total = len(lines)
    if total == 0:
        return f"File '{file_name}' is empty (0 lines)."

    # Clamp the requested window.
    start = max(1, start_line)
    limit = min(max(1, max_lines), READ_FILE_HARD_MAX_LINES)
    if end_line is not None and 0 < end_line < start:
        return f"Error: end_line {end_line} is before start_line {start}."
    end = min(end_line or start + limit - 1, start + limit - 1, total)
    if start > total:
        return f"Error: start_line {start} is beyond the end of the file ({total} lines)."

    # Render whole lines until the character budget (minus room for header/footer) is exhausted.
    width = len(str(end))
    budget = READ_FILE_MAX_CHARS - 400
    rendered: list[str] = []
    used = 0
    last = start - 1
    for lineno in range(start, end + 1):
        row = _render_line(lineno, width, lines[lineno - 1])
        if rendered and used + len(row) + 1 > budget:
            break
        rendered.append(row)
        used += len(row) + 1
        last = lineno

    header = f"File: {file_name} | Showing lines {start}-{last} of {total} ({len(text)} chars total)"
    out = [header, *rendered]
    if last < end:
        out.append(
            f"... (output capped at {READ_FILE_MAX_CHARS} chars; continue from line {last + 1}). "
            f"Call read_file(\"{file_name}\", start_line={last + 1}) to continue."
        )
    elif last < total:
        out.append(
            f"... {total - last} more lines. Call read_file(\"{file_name}\", start_line={last + 1}) to continue."
        )
    return "\n".join(out)

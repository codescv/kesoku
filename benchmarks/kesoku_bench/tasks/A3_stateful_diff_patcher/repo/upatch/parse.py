"""Unified diff parser producing structured FilePatch and Hunk objects."""

import re
from dataclasses import dataclass, field
from enum import StrEnum

HUNK_HEADER_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(?:\s?(.*))?$"
)


class DiffOpKind(StrEnum):
    """Operation kind for a single line inside a unified diff hunk."""

    CONTEXT = " "
    ADD = "+"
    DELETE = "-"


class PatchParseError(ValueError):
    """Raised when a unified diff cannot be parsed."""


@dataclass(frozen=True)
class HunkLine:
    """A single context (` `), addition (`+`), or deletion (`-`) line in a hunk."""

    kind: DiffOpKind
    text: str
    no_newline_at_eof: bool = False


@dataclass(frozen=True)
class Hunk:
    """A single `@@ -old_start,old_count +new_start,new_count @@` hunk."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[HunkLine] = field(default_factory=list)
    section_header: str = ""

    @property
    def net_delta(self) -> int:
        """Return the net change in line count (`new_count - old_count`) caused by this hunk."""
        adds = sum(1 for item in self.lines if item.kind is DiffOpKind.ADD)
        deletes = sum(1 for item in self.lines if item.kind is DiffOpKind.DELETE)
        return adds - deletes

    @property
    def leading_context_count(self) -> int:
        """Return the number of contiguous leading context lines at the start of the hunk."""
        count = 0
        for item in self.lines:
            if item.kind is DiffOpKind.CONTEXT:
                count += 1
            else:
                break
        return count

    @property
    def trailing_context_count(self) -> int:
        """Return the number of contiguous trailing context lines at the end of the hunk."""
        count = 0
        for item in reversed(self.lines):
            if item.kind is DiffOpKind.CONTEXT:
                count += 1
            else:
                break
        return count


@dataclass(frozen=True)
class FilePatch:
    """Parsed unified diff for a single file (`--- old_path` / `+++ new_path`)."""

    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)


def _strip_diff_line_ending(raw_line: str) -> str:
    """Strip trailing CRLF or LF from a raw line in the patch text."""
    if raw_line.endswith("\r\n"):
        return raw_line[:-2]
    if raw_line.endswith("\n") or raw_line.endswith("\r"):
        return raw_line[:-1]
    return raw_line


def _clean_header_path(header_val: str) -> str:
    """Clean a `--- a/foo` or `+++ b/foo` path header (stripping optional timestamp)."""
    path_part = header_val.split("\t", 1)[0].strip()
    if path_part.startswith(("a/", "b/")):
        return path_part[2:]
    return path_part


def parse_unified_diff(diff_input: str | bytes) -> list[FilePatch]:
    """Parse a unified diff string or bytes into a list of `FilePatch` objects."""
    if isinstance(diff_input, bytes):
        diff_text = diff_input.decode("utf-8")
    else:
        diff_text = diff_input

    raw_lines = diff_text.splitlines(keepends=True)
    patches: list[FilePatch] = []
    idx = 0
    total = len(raw_lines)

    old_path = ""
    new_path = ""
    current_hunks: list[Hunk] = []

    while idx < total:
        line = _strip_diff_line_ending(raw_lines[idx])

        # Skip git diff preambles (`diff --git ...`, `index ...`) and blank lines outside hunks.
        if not line or line.startswith(("diff ", "index ", "new file ", "deleted file ")):
            idx += 1
            continue

        if line.startswith("--- "):
            if current_hunks:
                patches.append(FilePatch(old_path=old_path, new_path=new_path, hunks=current_hunks))
                current_hunks = []
            old_path = _clean_header_path(line[4:])
            idx += 1
            if idx < total:
                next_line = _strip_diff_line_ending(raw_lines[idx])
                if next_line.startswith("+++ "):
                    new_path = _clean_header_path(next_line[4:])
                    idx += 1
            continue

        if line.startswith("+++ "):
            new_path = _clean_header_path(line[4:])
            idx += 1
            continue

        if line.startswith("@@"):
            m = HUNK_HEADER_RE.match(line)
            if m is None:
                raise PatchParseError(f"Malformed hunk header on line {idx + 1}: {line!r}")
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            section = m.group(5) or ""

            idx += 1
            hunk_lines: list[HunkLine] = []
            old_seen = 0
            new_seen = 0

            while idx < total and (old_seen < old_count or new_seen < new_count):
                h_raw = _strip_diff_line_ending(raw_lines[idx])
                if h_raw.startswith("\\ No newline at end of file"):
                    if hunk_lines:
                        prev = hunk_lines[-1]
                        hunk_lines[-1] = HunkLine(
                            kind=prev.kind,
                            text=prev.text,
                            no_newline_at_eof=True,
                        )
                    idx += 1
                    continue

                if not h_raw:
                    # Treat an empty line inside a hunk as a single space context line.
                    hunk_lines.append(HunkLine(kind=DiffOpKind.CONTEXT, text=""))
                    old_seen += 1
                    new_seen += 1
                    idx += 1
                    continue

                prefix = h_raw[0]
                body = h_raw[1:]
                if prefix == " ":
                    hunk_lines.append(HunkLine(kind=DiffOpKind.CONTEXT, text=body))
                    old_seen += 1
                    new_seen += 1
                elif prefix == "+":
                    hunk_lines.append(HunkLine(kind=DiffOpKind.ADD, text=body))
                    new_seen += 1
                elif prefix == "-":
                    hunk_lines.append(HunkLine(kind=DiffOpKind.DELETE, text=body))
                    old_seen += 1
                else:
                    raise PatchParseError(
                        f"Unexpected hunk line prefix {prefix!r} on line {idx + 1}: {h_raw!r}"
                    )
                idx += 1

            # Check for trailing `\ No newline at end of file` immediately following the hunk.
            if idx < total:
                peek_line = _strip_diff_line_ending(raw_lines[idx])
                if peek_line.startswith("\\ No newline at end of file") and hunk_lines:
                    prev = hunk_lines[-1]
                    hunk_lines[-1] = HunkLine(
                        kind=prev.kind,
                        text=prev.text,
                        no_newline_at_eof=True,
                    )
                    idx += 1

            current_hunks.append(
                Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=hunk_lines,
                    section_header=section,
                )
            )
            continue

        raise PatchParseError(f"Unexpected line outside hunk at line {idx + 1}: {line!r}")

    if current_hunks or old_path or new_path:
        patches.append(FilePatch(old_path=old_path, new_path=new_path, hunks=current_hunks))

    return patches


def parse_single_patch(diff_input: str | bytes) -> FilePatch:
    """Parse a unified diff expected to contain exactly one `FilePatch`."""
    patches = parse_unified_diff(diff_input)
    if not patches:
        raise PatchParseError("No file patches found in diff input")
    return patches[0]

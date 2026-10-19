"""Line-ending utilities for splitting, comparing, and reassembling text lines."""

from dataclasses import dataclass

LF = "\n"
CRLF = "\r\n"


@dataclass(frozen=True)
class LineRecord:
    r"""A single line split into its body text and original line terminator (`\\n`, `\\r\\n`, or ``)."""

    content: str
    ending: str = LF

    def render(self) -> str:
        """Recombine the line content with its line ending."""
        return f"{self.content}{self.ending}"


def split_lines_preserve(text: str) -> list[str]:
    """Split `text` into lines keeping their original line endings attached."""
    return text.splitlines(keepends=True)


def split_to_records(text: str) -> list[LineRecord]:
    """Split `text` into a list of `LineRecord(content, ending)` objects."""
    records: list[LineRecord] = []
    for raw in text.splitlines(keepends=True):
        if raw.endswith(CRLF):
            records.append(LineRecord(content=raw[:-2], ending=CRLF))
        elif raw.endswith(LF):
            records.append(LineRecord(content=raw[:-1], ending=LF))
        elif raw.endswith("\r"):
            records.append(LineRecord(content=raw[:-1], ending="\r"))
        else:
            records.append(LineRecord(content=raw, ending=""))
    return records


def strip_target_newline(raw_line: str) -> str:
    """Strip the trailing newline from a target file line for comparison with a patch line."""
    # BUG: Only strips `\n`, leaving trailing `\r` on CRLF target lines!
    if raw_line.endswith("\n"):
        return raw_line[:-1]
    return raw_line


def detect_dominant_ending(lines: list[str]) -> str:
    r"""Determine the dominant line ending (`\\r\\n` vs `\\n`) across `lines`."""
    crlf_count = sum(1 for ln in lines if ln.endswith(CRLF))
    lf_count = sum(1 for ln in lines if ln.endswith(LF) and not ln.endswith(CRLF))
    if crlf_count > lf_count:
        return CRLF
    return LF


def infer_local_ending(lines: list[str], start_idx: int, count: int) -> str:
    """Infer the line ending to use for newly added lines in a hunk near `start_idx`."""
    window = lines[max(0, start_idx) : max(0, start_idx) + max(1, count)]
    for ln in window:
        if ln.endswith(CRLF):
            return CRLF
        if ln.endswith(LF):
            return LF
    return detect_dominant_ending(lines)

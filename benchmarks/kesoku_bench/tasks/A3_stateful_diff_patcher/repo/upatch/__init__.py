"""upatch: stateful unified diff parser and patch applicator."""

from upatch.apply import (
    ApplyOptions,
    HunkApplication,
    PatchApplyError,
    PatchState,
    apply_file_patch,
    apply_patch_bytes,
    apply_patch_text,
)
from upatch.io import read_patch_file, read_target_file, write_target_file
from upatch.lineend import (
    CRLF,
    LF,
    LineRecord,
    detect_dominant_ending,
    infer_local_ending,
    split_lines_preserve,
    split_to_records,
    strip_target_newline,
)
from upatch.parse import (
    DiffOpKind,
    FilePatch,
    Hunk,
    HunkLine,
    PatchParseError,
    parse_single_patch,
    parse_unified_diff,
)

__all__ = [
    "CRLF",
    "LF",
    "ApplyOptions",
    "DiffOpKind",
    "FilePatch",
    "Hunk",
    "HunkApplication",
    "HunkLine",
    "LineRecord",
    "PatchApplyError",
    "PatchParseError",
    "PatchState",
    "apply_file_patch",
    "apply_patch_bytes",
    "apply_patch_text",
    "detect_dominant_ending",
    "infer_local_ending",
    "parse_single_patch",
    "parse_unified_diff",
    "read_patch_file",
    "read_target_file",
    "split_lines_preserve",
    "split_to_records",
    "strip_target_newline",
    "write_target_file",
]

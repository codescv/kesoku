"""Stateful unified diff hunk locator and patch engine.

Supports three hunk matching strategies:
1. Exact matching (`_try_exact_strategy`): verifies context and deletion lines at the
   expected line index.
2. Offset matching (`_try_offset_strategy`): scans forward and backward from the expected
   line index within a configurable window.
3. Fuzz matching (`_try_fuzz_strategy`): allows up to `max_fuzz` outer context lines at
   the leading or trailing edge of a hunk to be ignored when locating a match.
"""

from dataclasses import dataclass, field

from upatch.lineend import (
    LF,
    detect_dominant_ending,
    split_lines_preserve,
    strip_target_newline,
)
from upatch.parse import (
    DiffOpKind,
    FilePatch,
    Hunk,
    HunkLine,
    parse_single_patch,
)


class PatchApplyError(RuntimeError):
    """Raised when one or more hunks in a unified diff cannot be applied."""


@dataclass(frozen=True)
class ApplyOptions:
    """Configuration options controlling hunk matching tolerance and line endings."""

    allow_offset: bool = False
    max_offset: int = 200
    max_fuzz: int = 0
    preserve_line_endings: bool = True


@dataclass(frozen=True)
class HunkCheckDetail:
    """Detailed diagnostic record for a single candidate position check."""

    candidate_idx: int
    matched: bool
    mismatch_offset: int = -1
    expected_text: str = ""
    actual_text: str = ""
    strategy: str = "exact"
    fuzz_level: int = 0


@dataclass(frozen=True)
class HunkApplication:
    """Result of locating and rendering a single hunk against the working line buffer."""

    hunk_index: int
    start_idx: int
    end_idx: int
    replacement_lines: list[str]
    delta: int
    applied_offset: int
    fuzz_used: int
    strategy: str
    last_mod_offset_in_replacement: int


@dataclass
class PatchState:
    """Mutable state tracked across sequential hunk applications within a single file."""

    working_lines: list[str]
    output_buffer: list[str] = field(default_factory=list)
    emitted_cursor: int = 0
    cumulative_offset: int = 0
    last_hunk_end: int = 0
    last_mod_end: int = 0
    default_ending: str = LF
    applied_hunks: list[HunkApplication] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-analysis helpers for hunks and target buffers
# ---------------------------------------------------------------------------


def extract_old_lines_from_hunk(hunk: Hunk) -> list[HunkLine]:
    """Extract all HunkLines that must match against the source buffer (CONTEXT and DELETE)."""
    old_items: list[HunkLine] = []
    for item in hunk.lines:
        if item.kind is DiffOpKind.CONTEXT:
            old_items.append(item)
        elif item.kind is DiffOpKind.DELETE:
            old_items.append(item)
        elif item.kind is DiffOpKind.ADD:
            continue
    return old_items


def extract_new_lines_from_hunk(hunk: Hunk) -> list[HunkLine]:
    """Extract all HunkLines that will appear in the patched buffer (CONTEXT and ADD)."""
    new_items: list[HunkLine] = []
    for item in hunk.lines:
        if item.kind is DiffOpKind.CONTEXT:
            new_items.append(item)
        elif item.kind is DiffOpKind.ADD:
            new_items.append(item)
        elif item.kind is DiffOpKind.DELETE:
            continue
    return new_items


def count_hunk_operations(hunk: Hunk) -> tuple[int, int, int]:
    """Count `(context_count, add_count, delete_count)` in `hunk`."""
    ctx_count = 0
    add_count = 0
    del_count = 0
    for item in hunk.lines:
        if item.kind is DiffOpKind.CONTEXT:
            ctx_count += 1
        elif item.kind is DiffOpKind.ADD:
            add_count += 1
        elif item.kind is DiffOpKind.DELETE:
            del_count += 1
    return ctx_count, add_count, del_count


def compute_last_mod_offset_in_replacement(hunk: Hunk) -> int:
    """Compute the index in the replacement slice immediately after the last ADD or DELETE."""
    out_idx = 0
    last_mod = 0
    for item in hunk.lines:
        if item.kind is DiffOpKind.CONTEXT:
            out_idx += 1
        elif item.kind is DiffOpKind.ADD:
            out_idx += 1
            last_mod = out_idx
        elif item.kind is DiffOpKind.DELETE:
            last_mod = out_idx
    return last_mod


def invert_hunk(hunk: Hunk) -> Hunk:
    """Return a reversed copy of `hunk` where ADD and DELETE operations are swapped."""
    inverted_lines: list[HunkLine] = []
    for item in hunk.lines:
        if item.kind is DiffOpKind.CONTEXT:
            inverted_lines.append(
                HunkLine(
                    kind=DiffOpKind.CONTEXT,
                    text=item.text,
                    no_newline_at_eof=item.no_newline_at_eof,
                )
            )
        elif item.kind is DiffOpKind.ADD:
            inverted_lines.append(
                HunkLine(
                    kind=DiffOpKind.DELETE,
                    text=item.text,
                    no_newline_at_eof=item.no_newline_at_eof,
                )
            )
        elif item.kind is DiffOpKind.DELETE:
            inverted_lines.append(
                HunkLine(
                    kind=DiffOpKind.ADD,
                    text=item.text,
                    no_newline_at_eof=item.no_newline_at_eof,
                )
            )
    return Hunk(
        old_start=hunk.new_start,
        old_count=hunk.new_count,
        new_start=hunk.old_start,
        new_count=hunk.old_count,
        lines=inverted_lines,
        section_header=hunk.section_header,
    )


def format_rejected_hunk(hunk: Hunk, ending: str = LF) -> str:
    """Format a failed `hunk` into a GNU-patch-compatible `.rej` block."""
    header = (
        f"@@ -{hunk.old_start},{hunk.old_count} "
        f"+{hunk.new_start},{hunk.new_count} @@"
    )
    if hunk.section_header:
        header = f"{header} {hunk.section_header}"
    out: list[str] = [f"{header}{ending}"]
    for item in hunk.lines:
        if item.kind is DiffOpKind.CONTEXT:
            out.append(f" {item.text}{ending}")
        elif item.kind is DiffOpKind.ADD:
            out.append(f"+{item.text}{ending}")
        elif item.kind is DiffOpKind.DELETE:
            out.append(f"-{item.text}{ending}")
        if item.no_newline_at_eof:
            out.append(f"\\ No newline at end of file{ending}")
    return "".join(out)


def validate_hunk_counts(hunk: Hunk) -> bool:
    """Verify that the declared old_count and new_count match the hunk body lines."""
    ctx_count, add_count, del_count = count_hunk_operations(hunk)
    expected_old = ctx_count + del_count
    expected_new = ctx_count + add_count
    return expected_old == hunk.old_count and expected_new == hunk.new_count


# ---------------------------------------------------------------------------
# Strategy 1: Exact position verification and rendering
# ---------------------------------------------------------------------------


def _compare_exact_line(target_raw: str, patch_text: str) -> bool:
    """Compare a raw target line against a patch line for exact strategy matching."""
    stripped_target = strip_target_newline(target_raw)
    return stripped_target == patch_text


def _verify_exact_window(
    working_lines: list[str],
    hunk: Hunk,
    candidate_idx: int,
) -> HunkCheckDetail:
    """Verify whether `hunk` matches `working_lines` starting at `candidate_idx`."""
    if candidate_idx < 0:
        return HunkCheckDetail(
            candidate_idx=candidate_idx,
            matched=False,
            mismatch_offset=0,
            expected_text="<non-negative index>",
            actual_text=str(candidate_idx),
            strategy="exact",
            fuzz_level=0,
        )

    src_pos = candidate_idx
    total_lines = len(working_lines)

    for op_idx, item in enumerate(hunk.lines):
        if item.kind is DiffOpKind.ADD:
            continue

        if item.kind is DiffOpKind.CONTEXT:
            if src_pos >= total_lines:
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text="<EOF>",
                    strategy="exact",
                    fuzz_level=0,
                )
            actual_raw = working_lines[src_pos]
            if not _compare_exact_line(actual_raw, item.text):
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text=strip_target_newline(actual_raw),
                    strategy="exact",
                    fuzz_level=0,
                )
            src_pos += 1
            continue

        if item.kind is DiffOpKind.DELETE:
            if src_pos >= total_lines:
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text="<EOF>",
                    strategy="exact",
                    fuzz_level=0,
                )
            actual_raw = working_lines[src_pos]
            if not _compare_exact_line(actual_raw, item.text):
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text=strip_target_newline(actual_raw),
                    strategy="exact",
                    fuzz_level=0,
                )
            src_pos += 1
            continue

    return HunkCheckDetail(
        candidate_idx=candidate_idx,
        matched=True,
        mismatch_offset=-1,
        strategy="exact",
        fuzz_level=0,
    )


def _build_exact_replacement_lines(
    working_lines: list[str],
    hunk: Hunk,
    match_idx: int,
    default_ending: str,
) -> tuple[list[str], int]:
    """Construct the replacement line list for an exact match at `match_idx`."""
    del default_ending
    rendered: list[str] = []
    src_pos = match_idx

    for item in hunk.lines:
        if item.kind is DiffOpKind.CONTEXT:
            _ = working_lines[src_pos]
            term = "" if item.no_newline_at_eof else LF
            rendered.append(f"{item.text}{term}")
            src_pos += 1
        elif item.kind is DiffOpKind.DELETE:
            _ = working_lines[src_pos]
            src_pos += 1
        elif item.kind is DiffOpKind.ADD:
            term = "" if item.no_newline_at_eof else LF
            rendered.append(f"{item.text}{term}")

    return rendered, src_pos


def _try_exact_strategy(
    state: PatchState,
    hunk: Hunk,
    hunk_index: int,
    expected_idx: int,
) -> tuple[HunkApplication | None, HunkCheckDetail]:
    """Attempt to apply `hunk` at `expected_idx` using exact matching."""
    check = _verify_exact_window(state.working_lines, hunk, expected_idx)
    if not check.matched:
        return None, check

    replacement, end_idx = _build_exact_replacement_lines(
        state.working_lines,
        hunk,
        expected_idx,
        state.default_ending,
    )
    old_span = end_idx - expected_idx
    delta = len(replacement) - old_span
    app = HunkApplication(
        hunk_index=hunk_index,
        start_idx=expected_idx,
        end_idx=end_idx,
        replacement_lines=replacement,
        delta=delta,
        applied_offset=0,
        fuzz_used=0,
        strategy="exact",
        last_mod_offset_in_replacement=compute_last_mod_offset_in_replacement(hunk),
    )
    return app, check


# ---------------------------------------------------------------------------
# Strategy 2: Offset search verification and rendering
# ---------------------------------------------------------------------------


def _compare_offset_line(target_raw: str, patch_text: str) -> bool:
    """Compare a raw target line against a patch line during offset scanning."""
    stripped_target = strip_target_newline(target_raw)
    return stripped_target == patch_text


def _verify_offset_window(
    working_lines: list[str],
    hunk: Hunk,
    candidate_idx: int,
    min_allowed_idx: int,
) -> HunkCheckDetail:
    """Verify whether `hunk` matches `working_lines` at `candidate_idx` during offset search."""
    if candidate_idx < min_allowed_idx or candidate_idx < 0:
        return HunkCheckDetail(
            candidate_idx=candidate_idx,
            matched=False,
            mismatch_offset=0,
            expected_text=f"index >= {min_allowed_idx}",
            actual_text=str(candidate_idx),
            strategy="offset",
            fuzz_level=0,
        )

    src_pos = candidate_idx
    total_lines = len(working_lines)

    for op_idx, item in enumerate(hunk.lines):
        if item.kind is DiffOpKind.ADD:
            continue

        if item.kind is DiffOpKind.CONTEXT:
            if src_pos >= total_lines:
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text="<EOF>",
                    strategy="offset",
                    fuzz_level=0,
                )
            actual_raw = working_lines[src_pos]
            if not _compare_offset_line(actual_raw, item.text):
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text=strip_target_newline(actual_raw),
                    strategy="offset",
                    fuzz_level=0,
                )
            src_pos += 1
            continue

        if item.kind is DiffOpKind.DELETE:
            if src_pos >= total_lines:
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text="<EOF>",
                    strategy="offset",
                    fuzz_level=0,
                )
            actual_raw = working_lines[src_pos]
            if not _compare_offset_line(actual_raw, item.text):
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text=strip_target_newline(actual_raw),
                    strategy="offset",
                    fuzz_level=0,
                )
            src_pos += 1
            continue

    return HunkCheckDetail(
        candidate_idx=candidate_idx,
        matched=True,
        mismatch_offset=-1,
        strategy="offset",
        fuzz_level=0,
    )


def _build_offset_replacement_lines(
    working_lines: list[str],
    hunk: Hunk,
    match_idx: int,
    default_ending: str,
) -> tuple[list[str], int]:
    """Construct the replacement line list for an offset-located hunk at `match_idx`."""
    del default_ending
    rendered: list[str] = []
    src_pos = match_idx

    for item in hunk.lines:
        if item.kind is DiffOpKind.CONTEXT:
            _ = working_lines[src_pos]
            term = "" if item.no_newline_at_eof else LF
            rendered.append(f"{item.text}{term}")
            src_pos += 1
        elif item.kind is DiffOpKind.DELETE:
            _ = working_lines[src_pos]
            src_pos += 1
        elif item.kind is DiffOpKind.ADD:
            term = "" if item.no_newline_at_eof else LF
            rendered.append(f"{item.text}{term}")

    return rendered, src_pos


def _try_offset_strategy(
    state: PatchState,
    hunk: Hunk,
    hunk_index: int,
    expected_idx: int,
    max_offset: int,
) -> tuple[HunkApplication | None, HunkCheckDetail]:
    """Search outward from `expected_idx` up to `max_offset` lines for a match."""
    min_allowed = state.last_hunk_end
    last_detail = HunkCheckDetail(
        candidate_idx=expected_idx,
        matched=False,
        strategy="offset",
    )

    for distance in range(1, max_offset + 1):
        # Try backward candidate first, then forward candidate, mirroring GNU patch.
        backward_idx = expected_idx - distance
        if backward_idx >= min_allowed:
            b_check = _verify_offset_window(state.working_lines, hunk, backward_idx, min_allowed)
            last_detail = b_check
            if b_check.matched:
                replacement, end_idx = _build_offset_replacement_lines(
                    state.working_lines,
                    hunk,
                    backward_idx,
                    state.default_ending,
                )
                old_span = end_idx - backward_idx
                return (
                    HunkApplication(
                        hunk_index=hunk_index,
                        start_idx=backward_idx,
                        end_idx=end_idx,
                        replacement_lines=replacement,
                        delta=len(replacement) - old_span,
                        applied_offset=-distance,
                        fuzz_used=0,
                        strategy="offset",
                        last_mod_offset_in_replacement=compute_last_mod_offset_in_replacement(
                            hunk
                        ),
                    ),
                    b_check,
                )

        forward_idx = expected_idx + distance
        if forward_idx <= len(state.working_lines):
            f_check = _verify_offset_window(state.working_lines, hunk, forward_idx, min_allowed)
            last_detail = f_check
            if f_check.matched:
                replacement, end_idx = _build_offset_replacement_lines(
                    state.working_lines,
                    hunk,
                    forward_idx,
                    state.default_ending,
                )
                old_span = end_idx - forward_idx
                return (
                    HunkApplication(
                        hunk_index=hunk_index,
                        start_idx=forward_idx,
                        end_idx=end_idx,
                        replacement_lines=replacement,
                        delta=len(replacement) - old_span,
                        applied_offset=distance,
                        fuzz_used=0,
                        strategy="offset",
                        last_mod_offset_in_replacement=compute_last_mod_offset_in_replacement(
                            hunk
                        ),
                    ),
                    f_check,
                )

    return None, last_detail


# ---------------------------------------------------------------------------
# Strategy 3: Fuzz-tolerant verification and rendering
# ---------------------------------------------------------------------------


def _compare_fuzz_line(target_raw: str, patch_text: str) -> bool:
    """Compare a raw target line against a patch line during fuzz matching."""
    stripped_target = strip_target_newline(target_raw)
    return stripped_target == patch_text


def _trim_hunk_for_fuzz(hunk: Hunk, fuzz: int) -> tuple[list[HunkLine], int, int]:
    """Trim up to `fuzz` leading and trailing context lines from `hunk.lines`."""
    lead_ctx = hunk.leading_context_count
    trail_ctx = hunk.trailing_context_count
    trim_lead = min(fuzz, max(0, lead_ctx))
    trim_trail = min(fuzz, max(0, trail_ctx))

    end_slice = len(hunk.lines) - trim_trail if trim_trail > 0 else len(hunk.lines)
    trimmed = hunk.lines[trim_lead:end_slice]
    return trimmed, trim_lead, trim_trail


def _verify_fuzz_window(
    working_lines: list[str],
    hunk: Hunk,
    candidate_idx: int,
    fuzz: int,
    min_allowed_idx: int,
) -> HunkCheckDetail:
    """Verify whether `hunk` matches at `candidate_idx` when ignoring `fuzz` outer context lines."""
    trimmed_ops, trim_lead, _ = _trim_hunk_for_fuzz(hunk, fuzz)
    effective_start = candidate_idx + trim_lead
    if effective_start < min_allowed_idx or effective_start < 0:
        return HunkCheckDetail(
            candidate_idx=candidate_idx,
            matched=False,
            mismatch_offset=0,
            expected_text=f"effective_start >= {min_allowed_idx}",
            actual_text=str(effective_start),
            strategy="fuzz",
            fuzz_level=fuzz,
        )

    src_pos = effective_start
    total_lines = len(working_lines)

    for op_idx, item in enumerate(trimmed_ops):
        if item.kind is DiffOpKind.ADD:
            continue

        if item.kind is DiffOpKind.CONTEXT:
            if src_pos >= total_lines:
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text="<EOF>",
                    strategy="fuzz",
                    fuzz_level=fuzz,
                )
            actual_raw = working_lines[src_pos]
            if not _compare_fuzz_line(actual_raw, item.text):
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text=strip_target_newline(actual_raw),
                    strategy="fuzz",
                    fuzz_level=fuzz,
                )
            src_pos += 1
            continue

        if item.kind is DiffOpKind.DELETE:
            if src_pos >= total_lines:
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text="<EOF>",
                    strategy="fuzz",
                    fuzz_level=fuzz,
                )
            actual_raw = working_lines[src_pos]
            if not _compare_fuzz_line(actual_raw, item.text):
                return HunkCheckDetail(
                    candidate_idx=candidate_idx,
                    matched=False,
                    mismatch_offset=op_idx,
                    expected_text=item.text,
                    actual_text=strip_target_newline(actual_raw),
                    strategy="fuzz",
                    fuzz_level=fuzz,
                )
            src_pos += 1
            continue

    return HunkCheckDetail(
        candidate_idx=candidate_idx,
        matched=True,
        mismatch_offset=-1,
        strategy="fuzz",
        fuzz_level=fuzz,
    )


def _build_fuzz_replacement_lines(
    working_lines: list[str],
    hunk: Hunk,
    match_idx: int,
    fuzz: int,
    default_ending: str,
) -> tuple[list[str], int, int]:
    """Construct the replacement slice `(lines, start_idx, end_idx)` for a fuzz match."""
    del default_ending
    trimmed_ops, trim_lead, _ = _trim_hunk_for_fuzz(hunk, fuzz)
    effective_start = match_idx + trim_lead
    src_pos = effective_start
    rendered: list[str] = []

    for item in trimmed_ops:
        if item.kind is DiffOpKind.CONTEXT:
            _ = working_lines[src_pos]
            term = "" if item.no_newline_at_eof else LF
            rendered.append(f"{item.text}{term}")
            src_pos += 1
        elif item.kind is DiffOpKind.DELETE:
            _ = working_lines[src_pos]
            src_pos += 1
        elif item.kind is DiffOpKind.ADD:
            term = "" if item.no_newline_at_eof else LF
            rendered.append(f"{item.text}{term}")

    return rendered, effective_start, src_pos


def _try_fuzz_strategy(
    state: PatchState,
    hunk: Hunk,
    hunk_index: int,
    expected_idx: int,
    max_offset: int,
    max_fuzz: int,
) -> tuple[HunkApplication | None, HunkCheckDetail]:
    """Attempt fuzz matching for `fuzz` from 1 up to `max_fuzz` across `max_offset`."""
    min_allowed = state.last_hunk_end
    last_detail = HunkCheckDetail(
        candidate_idx=expected_idx,
        matched=False,
        strategy="fuzz",
    )

    for fuzz in range(1, max_fuzz + 1):
        for distance in range(0, max_offset + 1):
            candidates = [expected_idx] if distance == 0 else [
                expected_idx - distance,
                expected_idx + distance,
            ]
            for cand_idx in candidates:
                f_check = _verify_fuzz_window(
                    state.working_lines,
                    hunk,
                    cand_idx,
                    fuzz,
                    min_allowed,
                )
                last_detail = f_check
                if f_check.matched:
                    replacement, eff_start, eff_end = _build_fuzz_replacement_lines(
                        state.working_lines,
                        hunk,
                        cand_idx,
                        fuzz,
                        state.default_ending,
                    )
                    old_span = eff_end - eff_start
                    return (
                        HunkApplication(
                            hunk_index=hunk_index,
                            start_idx=eff_start,
                            end_idx=eff_end,
                            replacement_lines=replacement,
                            delta=len(replacement) - old_span,
                            applied_offset=cand_idx - expected_idx,
                            fuzz_used=fuzz,
                            strategy="fuzz",
                            last_mod_offset_in_replacement=len(replacement),
                        ),
                        f_check,
                    )

    return None, last_detail


# ---------------------------------------------------------------------------
# Multi-hunk orchestration and buffer splicing
# ---------------------------------------------------------------------------


def _locate_and_apply_hunk(
    state: PatchState,
    hunk: Hunk,
    hunk_index: int,
    options: ApplyOptions,
) -> HunkApplication:
    """Locate `hunk` in `state.working_lines` using configured strategies and return its application."""
    # BUG 1: Uses original 1-based hunk.old_start without adding state.cumulative_offset!
    expected_idx = max(0, hunk.old_start - 1)

    app, exact_detail = _try_exact_strategy(state, hunk, hunk_index, expected_idx)
    if app is not None:
        return app

    if options.allow_offset and options.max_offset > 0:
        app, _ = _try_offset_strategy(
            state,
            hunk,
            hunk_index,
            expected_idx,
            options.max_offset,
        )
        if app is not None:
            return app

    if options.max_fuzz > 0:
        app, _ = _try_fuzz_strategy(
            state,
            hunk,
            hunk_index,
            expected_idx,
            options.max_offset if options.allow_offset else 0,
            options.max_fuzz,
        )
        if app is not None:
            return app

    raise PatchApplyError(
        f"Context mismatch in hunk #{hunk_index + 1} at line {expected_idx + 1}: "
        f"expected {exact_detail.expected_text!r}, got {exact_detail.actual_text!r}"
    )


def _splice_hunk_into_state(state: PatchState, app: HunkApplication) -> None:
    """Apply `app` to `state.working_lines` and append emitted lines to `state.output_buffer`."""
    # BUG 3: When hunk k+1's leading context overlaps hunk k's trailing context,
    # `app.start_idx < state.emitted_cursor`. In Python, `working_lines[emitted_cursor:start_idx]`
    # evaluates to `[]`, and `output_buffer.extend(app.replacement_lines)` appends the
    # overlapping leading context lines a second time!
    unchanged_between = state.working_lines[state.emitted_cursor : app.start_idx]
    state.output_buffer.extend(unchanged_between)
    state.output_buffer.extend(app.replacement_lines)

    state.working_lines[app.start_idx : app.end_idx] = app.replacement_lines
    new_end = app.start_idx + len(app.replacement_lines)
    state.emitted_cursor = new_end
    state.last_hunk_end = new_end
    state.last_mod_end = app.start_idx + app.last_mod_offset_in_replacement
    state.cumulative_offset += app.delta
    state.applied_hunks.append(app)


def apply_file_patch(
    original_text: str,
    file_patch: FilePatch,
    options: ApplyOptions | None = None,
) -> str:
    """Apply all hunks in `file_patch` to `original_text` and return the patched text."""
    opts = options if options is not None else ApplyOptions()
    initial_lines = split_lines_preserve(original_text)
    dominant_ending = detect_dominant_ending(initial_lines)

    state = PatchState(
        working_lines=list(initial_lines),
        default_ending=dominant_ending,
    )

    for idx, hunk in enumerate(file_patch.hunks):
        app = _locate_and_apply_hunk(state, hunk, idx, opts)
        _splice_hunk_into_state(state, app)

    if state.emitted_cursor < len(state.working_lines):
        state.output_buffer.extend(state.working_lines[state.emitted_cursor :])

    return "".join(state.output_buffer)


def apply_patch_text(
    original_text: str,
    patch_text: str,
    options: ApplyOptions | None = None,
) -> str:
    """Parse `patch_text` and apply it to `original_text`."""
    file_patch = parse_single_patch(patch_text)
    return apply_file_patch(original_text, file_patch, options=options)


def apply_patch_bytes(
    original_bytes: bytes,
    patch_bytes: bytes | str,
    options: ApplyOptions | None = None,
) -> bytes:
    """Decode `original_bytes`, apply `patch_bytes`, and return UTF-8 encoded patched bytes."""
    original_text = original_bytes.decode("utf-8")
    patch_text = patch_bytes.decode("utf-8") if isinstance(patch_bytes, bytes) else patch_bytes
    patched_text = apply_patch_text(original_text, patch_text, options=options)
    return patched_text.encode("utf-8")

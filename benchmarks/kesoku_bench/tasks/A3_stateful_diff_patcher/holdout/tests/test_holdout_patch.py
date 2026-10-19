"""Holdout verification tests for A3_stateful_diff_patcher."""

from pathlib import Path

from upatch.apply import apply_patch_bytes, apply_patch_text
from upatch.cli import main

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def test_multi_hunk_delta_plus2_then_minus1() -> None:
    """Verify multi-hunk patch where hunk 1 has Δ=+2 and hunk 2 has Δ=-1 on repeated context."""
    original = (
        "header = v1\n"
        "block = start\n"
        "key = old_a\n"
        "block = end\n"
        "spacer = 1\n"
        "block = start\n"
        "key = old_b\n"
        "deprecated = yes\n"
        "block = end\n"
        "spacer = 2\n"
        "block = start\n"
        "key = old_c\n"
        "block = end\n"
    )
    patch = (
        "--- a/doc.txt\n"
        "+++ b/doc.txt\n"
        "@@ -2,3 +2,5 @@\n"
        " block = start\n"
        "-key = old_a\n"
        "+key = new_a1\n"
        "+key_extra_1 = 10\n"
        "+key_extra_2 = 20\n"
        " block = end\n"
        "@@ -6,4 +8,3 @@\n"
        " block = start\n"
        " key = old_b\n"
        "-deprecated = yes\n"
        " block = end\n"
        "@@ -11,3 +12,3 @@\n"
        " block = start\n"
        "-key = old_c\n"
        "+key = new_c\n"
        " block = end\n"
    )
    patched = apply_patch_text(original, patch)
    expected = (
        "header = v1\n"
        "block = start\n"
        "key = new_a1\n"
        "key_extra_1 = 10\n"
        "key_extra_2 = 20\n"
        "block = end\n"
        "spacer = 1\n"
        "block = start\n"
        "key = old_b\n"
        "block = end\n"
        "spacer = 2\n"
        "block = start\n"
        "key = new_c\n"
        "block = end\n"
    )
    assert patched == expected


def test_multi_fixture_matches_gnu_patch(tmp_path: Path) -> None:
    """Verify `upatch apply fixtures/multi.patch fixtures/multi.txt` matches GNU patch bytes."""
    out_file = tmp_path / "multi.out"
    rc = main(
        [
            "apply",
            str(FIXTURES_DIR / "multi.patch"),
            str(FIXTURES_DIR / "multi.txt"),
            "-o",
            str(out_file),
        ]
    )
    assert rc == 0
    expected = (
        b"[section:alpha]\n"
        b"enabled = true\n"
        b"workers = 4\n"
        b"queue_size = 64\n"
        b"prefetch = 2\n"
        b"timeout = 10\n"
        b"retries = 1\n"
        b"\n"
        b"[section:beta]\n"
        b"enabled = true\n"
        b"workers = 4\n"
        b"timeout = 20\n"
        b"retries = 2\n"
        b"\n"
        b"[section:gamma]\n"
        b"enabled = true\n"
        b"workers = 8\n"
        b"timeout = 45\n"
        b"retries = 3\n"
    )
    assert out_file.read_bytes() == expected


def test_crlf_byte_exact_round_trip(tmp_path: Path) -> None:
    """Verify CRLF file (`fixtures/win.txt`) patches cleanly and preserves CRLF on every line."""
    out_file = tmp_path / "win.out"
    rc = main(
        [
            "apply",
            str(FIXTURES_DIR / "win.patch"),
            str(FIXTURES_DIR / "win.txt"),
            "-o",
            str(out_file),
        ]
    )
    assert rc == 0
    expected = (
        b"header = 1\r\n"
        b"mode = strict\r\n"
        b"verify_tls = true\r\n"
        b"timeout = 30\r\n"
        b"retries = 3\r\n"
        b"footer = end\r\n"
    )
    assert out_file.read_bytes() == expected


def test_mixed_crlf_lf_preserved_per_line() -> None:
    """Verify mixed CRLF/LF file preserves the exact line ending of every existing line."""
    original_bytes = (
        b"line1 = crlf\r\n"
        b"line2 = lf\n"
        b"line3 = old_val\r\n"
        b"line4 = lf_keep\n"
        b"line5 = crlf_keep\r\n"
    )
    patch_bytes = (
        b"--- a/mixed.txt\n"
        b"+++ b/mixed.txt\n"
        b"@@ -2,4 +2,4 @@\n"
        b" line2 = lf\n"
        b"-line3 = old_val\n"
        b"+line3 = new_val\n"
        b" line4 = lf_keep\n"
        b" line5 = crlf_keep\n"
    )
    result = apply_patch_bytes(original_bytes, patch_bytes)
    lines = result.splitlines(keepends=True)
    assert lines[0] == b"line1 = crlf\r\n"
    assert lines[1] == b"line2 = lf\n"
    assert lines[2] in (b"line3 = new_val\r\n", b"line3 = new_val\n")
    assert lines[3] == b"line4 = lf_keep\n"
    assert lines[4] == b"line5 = crlf_keep\r\n"


def test_overlapping_context_pair() -> None:
    """Verify two hunks whose trailing and leading context overlap do not duplicate lines."""
    target_bytes = (FIXTURES_DIR / "overlap.txt").read_bytes()
    patch_bytes = (FIXTURES_DIR / "overlap.patch").read_bytes()

    result = apply_patch_bytes(target_bytes, patch_bytes)
    expected = (
        b"line_1\n"
        b"line_2\n"
        b"line_3_a\n"
        b"line_3_b\n"
        b"line_4\n"
        b"line_5\n"
        b"line_6\n"
        b"line_7_a\n"
        b"line_7_b\n"
        b"line_8\n"
        b"line_9\n"
        b"line_10\n"
    )
    assert result == expected

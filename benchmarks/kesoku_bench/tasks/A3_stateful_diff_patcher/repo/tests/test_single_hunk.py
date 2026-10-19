"""Visible unit tests for single-hunk LF patch application in upatch."""

from pathlib import Path

from upatch.apply import ApplyOptions, apply_patch_bytes, apply_patch_text
from upatch.cli import main

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def test_apply_single_hunk_lf_fixture() -> None:
    """Verify applying `fixtures/single.patch` to `fixtures/single.txt`."""
    target_bytes = (FIXTURES_DIR / "single.txt").read_bytes()
    patch_bytes = (FIXTURES_DIR / "single.patch").read_bytes()

    result = apply_patch_bytes(target_bytes, patch_bytes)
    expected = (
        b"host = 127.0.0.1\n"
        b"port = 9090\n"
        b"tls = true\n"
        b"debug = false\n"
        b"workers = 4\n"
    )
    assert result == expected


def test_apply_single_hunk_with_offset_option() -> None:
    """Verify single-hunk application when shifted by 2 lines with `allow_offset=True`."""
    original = (
        "# preamble 1\n"
        "# preamble 2\n"
        "host = 127.0.0.1\n"
        "port = 8080\n"
        "debug = false\n"
        "workers = 4\n"
    )
    patch = (FIXTURES_DIR / "single.patch").read_text(encoding="utf-8")
    patched = apply_patch_text(original, patch, options=ApplyOptions(allow_offset=True))
    assert "port = 9090\ntls = true\n" in patched


def test_cli_apply_single_hunk_to_output_file(tmp_path: Path) -> None:
    """Verify `upatch apply` writes the patched result to `-o <output>`."""
    out_file = tmp_path / "single.out.txt"
    rc = main(
        [
            "apply",
            str(FIXTURES_DIR / "single.patch"),
            str(FIXTURES_DIR / "single.txt"),
            "-o",
            str(out_file),
        ]
    )
    assert rc == 0
    assert out_file.read_bytes() == (
        b"host = 127.0.0.1\n"
        b"port = 9090\n"
        b"tls = true\n"
        b"debug = false\n"
        b"workers = 4\n"
    )

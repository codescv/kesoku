"""Visible unit tests for unified diff parsing in upatch."""

from upatch.parse import DiffOpKind, parse_single_patch, parse_unified_diff


def test_parse_single_hunk_header_and_operations() -> None:
    """Verify parsing a standard single-hunk unified diff."""
    diff = (
        "--- a/config.txt\n"
        "+++ b/config.txt\n"
        "@@ -1,4 +1,5 @@\n"
        " host = 127.0.0.1\n"
        "-port = 8080\n"
        "+port = 9090\n"
        "+tls = true\n"
        " debug = false\n"
        " workers = 4\n"
    )
    patch = parse_single_patch(diff)
    assert patch.old_path == "config.txt"
    assert patch.new_path == "config.txt"
    assert len(patch.hunks) == 1

    hunk = patch.hunks[0]
    assert hunk.old_start == 1
    assert hunk.old_count == 4
    assert hunk.new_start == 1
    assert hunk.new_count == 5
    assert hunk.net_delta == 1
    assert [ln.kind for ln in hunk.lines] == [
        DiffOpKind.CONTEXT,
        DiffOpKind.DELETE,
        DiffOpKind.ADD,
        DiffOpKind.ADD,
        DiffOpKind.CONTEXT,
        DiffOpKind.CONTEXT,
    ]


def test_parse_multiple_hunks_and_leading_trailing_context_counts() -> None:
    """Verify leading and trailing context counts across multiple hunks."""
    diff = (
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -2,4 +2,4 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 20\n"
        " c = 3\n"
        " d = 4\n"
        "@@ -10,3 +10,4 @@\n"
        " x = 10\n"
        "+y = 11\n"
        " z = 12\n"
        " w = 13\n"
    )
    patches = parse_unified_diff(diff)
    assert len(patches) == 1
    assert len(patches[0].hunks) == 2
    assert patches[0].hunks[0].leading_context_count == 1
    assert patches[0].hunks[0].trailing_context_count == 2
    assert patches[0].hunks[1].net_delta == 1

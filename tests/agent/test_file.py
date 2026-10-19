"""Unit tests for file management tools."""

import os
from pathlib import Path

import pytest

from kesoku.agent.tools import file as file_mod
from kesoku.agent.tools.file import (
    READ_FILE_MAX_CHARS,
    _closest_lines,
    _find_fuzzy_block,
    _strip_line_number_prefixes,
    _suggest_paths,
    read_file,
    update_file,
)
from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.config import get_config


@pytest.fixture
def test_awd(tmp_path):
    """Fixture to set up a temporary agent working directory."""
    cfg = get_config()
    old_awd = cfg.agent_working_dir
    cfg.agent_working_dir = str(tmp_path)
    yield tmp_path
    cfg.agent_working_dir = old_awd


def _write(path: Path, content: str) -> Path:
    """Write `content` to `path` (creating parents) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------------------------
# update_file: create mode
# ---------------------------------------------------------------------------------------------------------------


def test_update_file_create_success(test_awd) -> None:
    """Test creating a new file when old_content is None."""
    file_name = "new_file.txt"
    new_content = "Hello World\n"

    res = update_file(file_name=file_name, old_content=None, new_content=new_content)
    assert res.startswith("Success: Created file")
    assert "(1 lines)" in res

    resolved_path = os.path.join(str(test_awd), file_name)
    assert os.path.exists(resolved_path)
    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == new_content


def test_update_file_create_by_omitting_old_content(test_awd) -> None:
    """Omitting old_content entirely creates the file (parents included)."""
    res = update_file(file_name="sub/dir/notes.md", new_content="a\nb\nc")
    assert res == "Success: Created file 'sub/dir/notes.md' (3 lines)."
    assert (test_awd / "sub" / "dir" / "notes.md").read_text(encoding="utf-8") == "a\nb\nc"


def test_update_file_empty_old_content_is_create(test_awd) -> None:
    """old_content == '' is treated like an omitted old_content (create mode)."""
    res = update_file(file_name="x.txt", old_content="", new_content="hi")
    assert res.startswith("Success: Created file")
    assert (test_awd / "x.txt").read_text(encoding="utf-8") == "hi"


def test_update_file_create_rejects_empty_new_content(test_awd) -> None:
    """Creating a file with empty content is rejected."""
    res = update_file(file_name="x.txt", new_content="")
    assert res == "Error: new_content is empty."
    assert not (test_awd / "x.txt").exists()


def test_update_file_new_content_required(test_awd) -> None:
    """A missing new_content (e.g. a cut-off call) never deletes the old block."""
    _write(test_awd / "f.txt", "keep me\n")
    res = update_file(file_name="f.txt", old_content="keep me\n")
    assert res.startswith("Error: new_content is required")
    assert "4000-char chunks" in res
    assert (test_awd / "f.txt").read_text(encoding="utf-8") == "keep me\n"

    res2 = update_file(file_name="new.txt")
    assert res2.startswith("Error: new_content is required")
    assert not (test_awd / "new.txt").exists()


def test_update_file_modify_fail_old_content_none(test_awd) -> None:
    """Create mode on an existing file is rejected with guidance and leaves the file untouched."""
    file_name = "test.txt"
    initial_content = "line 1\nline 2\n"
    resolved_path = os.path.join(str(test_awd), file_name)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(initial_content)

    res = update_file(
        file_name=file_name,
        old_content=None,
        new_content="new content",
    )
    assert res.startswith("Error: File 'test.txt' already exists (2 lines).")
    assert "pass the exact lines to replace as old_content" in res
    assert "read_file" in res

    # Content should not change
    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == initial_content


def test_update_file_create_on_directory_rejected(test_awd) -> None:
    """Create mode on an existing directory gives a clear error."""
    (test_awd / "d").mkdir()
    res = update_file(file_name="d", new_content="x")
    assert res == "Error: 'd' is a directory."


# ---------------------------------------------------------------------------------------------------------------
# update_file: edit mode
# ---------------------------------------------------------------------------------------------------------------


def test_update_file_create_fail_old_content_provided(test_awd) -> None:
    """Test that editing fails if old_content is provided but the file doesn't exist."""
    file_name = "new_file.txt"
    new_content = "Hello World\n"

    res = update_file(file_name=file_name, old_content="something", new_content=new_content)
    assert res.startswith("Error: File 'new_file.txt' does not exist.")
    assert "If you meant to create it, omit old_content." in res
    assert "Did you mean" not in res

    resolved_path = os.path.join(str(test_awd), file_name)
    assert not os.path.exists(resolved_path)


def test_update_file_missing_file_suggests_repo_prefix(test_awd) -> None:
    """A path missing the first-level `repo/` directory gets a suggestion."""
    _write(test_awd / "repo" / "pkg" / "mod.py", "x = 1\n")
    res = update_file(file_name="pkg/mod.py", old_content="x = 1", new_content="x = 2")
    assert "does not exist" in res
    assert "Did you mean: repo/pkg/mod.py?" in res
    assert not (test_awd / "pkg").exists()


def test_update_file_modify_success_single_match(test_awd) -> None:
    """Test modifying a file with a single match of old_content."""
    file_name = "test.txt"
    initial_content = "line 1\nline 2\nline 3\n"
    resolved_path = os.path.join(str(test_awd), file_name)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(initial_content)

    res = update_file(
        file_name=file_name,
        old_content="line 2\n",
        new_content="line 2 modified\n",
    )
    assert res == "Success: Updated file 'test.txt' (1 replacement at line 2; file now has 3 lines)."

    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == "line 1\nline 2 modified\nline 3\n"


def test_update_file_success_reports_new_line_range(test_awd) -> None:
    """The success message reports the line range of the new text in the updated file."""
    _write(test_awd / "t.py", "a\nb\nc\nd\n")
    res = update_file(file_name="t.py", old_content="b\nc\n", new_content="B1\nB2\nB3\n")
    assert "1 replacement at lines 2-4; file now has 5 lines" in res
    assert (test_awd / "t.py").read_text(encoding="utf-8") == "a\nB1\nB2\nB3\nd\n"


def test_update_file_empty_new_content_deletes_block(test_awd) -> None:
    """An explicit '' new_content in edit mode deletes the block."""
    _write(test_awd / "t.txt", "a\nb\nc\n")
    res = update_file(file_name="t.txt", old_content="b\n", new_content="")
    assert res.startswith("Success: Updated file")
    assert "file now has 2 lines" in res
    assert (test_awd / "t.txt").read_text(encoding="utf-8") == "a\nc\n"


def test_update_file_modify_success_double_match(test_awd) -> None:
    """Two matches are both replaced and the message warns about it with both line ranges."""
    file_name = "test.txt"
    initial_content = "target\nline 1\ntarget\nline 2\n"
    resolved_path = os.path.join(str(test_awd), file_name)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(initial_content)

    res = update_file(
        file_name=file_name,
        old_content="target\n",
        new_content="replacement\nmore\n",
    )
    assert res.startswith("Success: Updated file")
    assert "⚠️ replaced 2 occurrences at lines 1-2 and 4-5; if you meant one, include more context" in res

    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == "replacement\nmore\nline 1\nreplacement\nmore\nline 2\n"


def test_update_file_modify_fail_no_match(test_awd) -> None:
    """Test that modifying fails if old_content is not found."""
    file_name = "test.txt"
    initial_content = "line 1\nline 2\n"
    resolved_path = os.path.join(str(test_awd), file_name)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(initial_content)

    res = update_file(
        file_name=file_name,
        old_content="nonexistent",
        new_content="replacement",
    )
    assert res.startswith("Error: old_content not found")

    # Content should not change
    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == initial_content


def test_update_file_modify_fail_too_many_matches(test_awd) -> None:
    """Test that modifying fails if old_content matches more than 2 places."""
    file_name = "test.txt"
    initial_content = "target\ntarget\ntarget\n"
    resolved_path = os.path.join(str(test_awd), file_name)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(initial_content)

    res = update_file(
        file_name=file_name,
        old_content="target\n",
        new_content="replacement\n",
    )
    assert res.startswith("Error: old_content matches 3 places")
    assert "max allowed is 2" in res

    # Content should not change
    with open(resolved_path, encoding="utf-8") as f:
        assert f.read() == initial_content


def test_update_file_too_many_matches_lists_lines_and_context(test_awd) -> None:
    """>2 matches: the error lists the match lines and shows numbered context for the first 3."""
    body = "".join(f"x{i}\nreturn None\n" for i in range(1, 6))  # matches at lines 2, 4, 6, 8, 10
    _write(test_awd / "m.py", body)
    res = update_file(file_name="m.py", old_content="return None", new_content="return 1")
    assert "matches 5 places in 'm.py' (lines 2, 4, 6, 8, 10; max allowed is 2)" in res
    assert "Include more surrounding lines" in res
    assert "Context of the first 3 matches:" in res
    assert "--- match at line 2:" in res and "--- match at line 6:" in res
    assert "--- match at line 8:" not in res
    assert "\n1: x1\n" in res and "\n3: x2\n" in res
    assert (test_awd / "m.py").read_text(encoding="utf-8") == body


def test_update_file_strips_line_number_prefixes(test_awd) -> None:
    """old_content pasted from read_file with `N: ` prefixes (incl. a bare `N:` empty line) still applies."""
    _write(test_awd / "s.py", "def f():\n    a = 1\n\n    return a\n")
    old = " 1: def f():\n 2:     a = 1\n 3:\n 4:     return a\n"
    res = update_file(file_name="s.py", old_content=old, new_content="def f():\n    return 2\n")
    assert res.startswith("Success: Updated file"), res
    assert "(note: stripped read_file line-number prefixes from old_content)" in res
    assert (test_awd / "s.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_update_file_whitespace_fuzzy_hint(test_awd) -> None:
    """A block differing only in indentation yields the exact file lines and their range."""
    _write(test_awd / "w.py", "class A:\n    def g(self):\n        return  1\n\nx = 2\n")
    res = update_file(file_name="w.py", old_content="def g(self):\n    return 1\n", new_content="pass\n")
    assert res.startswith("Error: old_content not found in file 'w.py'.")
    assert "differs only in whitespace/indentation exists at lines 2-3" in res
    assert "```\n    def g(self):\n        return  1\n```" in res
    assert "Retry with exactly these lines as old_content." in res


def test_update_file_closest_lines_hint(test_awd) -> None:
    """Without a whitespace-insensitive match, the closest lines and a read_file call are suggested."""
    lines = [f"filler line number {i}" for i in range(1, 40)]
    lines[29] = "def compute_total(items, tax_rate):"
    _write(test_awd / "c.py", "\n".join(lines) + "\n")
    res = update_file(
        file_name="c.py",
        old_content="def compute_totals(items, tax):\n    pass\n",
        new_content="x",
    )
    assert "Closest lines to the first line of old_content:" in res
    assert "  30: def compute_total(items, tax_rate):" in res
    assert 'read_file("c.py", start_line=25, end_line=45)' in res


def test_update_file_no_similar_lines(test_awd) -> None:
    """With nothing similar, the error points to read_file."""
    _write(test_awd / "n.txt", "aaaa\nbbbb\n")
    res = update_file(file_name="n.txt", old_content="zzzzzzzzzzzz qqqq", new_content="x")
    assert "no similar lines exist" in res
    assert 'read_file("n.txt")' in res


# ---------------------------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1: a\n2: b", "a\nb"),
        ("  9: x\n 10:     y\n", "x\n    y\n"),
        ("3: a\n4:\n5: b", "a\n\nb"),  # empty source line whose trailing space was stripped
        ("3: a\n\n5: b", "a\n\nb"),  # a fully blank line is kept
        ("     7\tfoo\n     8\tbar", "foo\nbar"),  # cat -n style
        ("1: a\nb", None),  # one line without a prefix -> no stripping
        ("plain text", None),
        ("", None),
        ("12:30 meeting", None),  # a colon not followed by space/EOL is not a prefix
    ],
)
def test_strip_line_number_prefixes(text: str, expected: str | None) -> None:
    """Prefixes are stripped only when every non-blank line has one."""
    assert _strip_line_number_prefixes(text) == expected


def test_find_fuzzy_block() -> None:
    """Whitespace-normalised block search returns the first 1-based inclusive span."""
    content = ["x", "  if a:", "\treturn   b", "if a:", "return b"]
    assert _find_fuzzy_block(content, ["if a:", "  return b", ""]) == (2, 3)
    assert _find_fuzzy_block(content, ["", "nope"]) is None
    assert _find_fuzzy_block(content, ["", "  "]) is None


def test_closest_lines() -> None:
    """Closest lines are ranked by similarity and carry their line numbers."""
    content = ["import os", "def load_config(path):", "", "def save_config(path, data):"]
    res = _closest_lines(content, "def load_configs(path):")
    assert res[0] == (2, "def load_config(path):")
    assert _closest_lines(content, "   ") == []
    assert _closest_lines(content, "completely unrelated zebra") == []


def test_suggest_paths_basename_walk(tmp_path) -> None:
    """Same-basename files deep in the AWD are suggested; skipped dirs are not walked."""
    _write(tmp_path / "a" / "b" / "c" / "target.py", "")
    _write(tmp_path / ".venv" / "target.py", "")
    _write(tmp_path / "node_modules" / "x" / "target.py", "")
    res = _suggest_paths(str(tmp_path / "target.py"), str(tmp_path))
    assert res == [os.path.join("a", "b", "c", "target.py")]


def test_suggest_paths_caps_and_missing_awd(tmp_path) -> None:
    """At most 3 suggestions; a non-existent AWD yields none."""
    for d in ("d1", "d2", "d3", "d4"):
        _write(tmp_path / d / "f.txt", "")
    assert len(_suggest_paths(str(tmp_path / "f.txt"), str(tmp_path))) == 3
    assert _suggest_paths(str(tmp_path / "f.txt"), str(tmp_path / "nope")) == []


# ---------------------------------------------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------------------------------------------


def _make_lines(path: Path, n: int) -> Path:
    """Write an n-line file `line 1` .. `line n`."""
    return _write(path, "".join(f"line {i}\n" for i in range(1, n + 1)))


def test_read_file_registered_after_update_file() -> None:
    """read_file is a registered tool, right after update_file."""
    names = list(default_registry._tools)
    assert names.index("read_file") == names.index("update_file") + 1


def test_read_file_whole_small_file(test_awd) -> None:
    """A small file is shown fully with a header and no footer."""
    _write(test_awd / "a.txt", "alpha\nbeta\n")
    res = read_file("a.txt")
    assert res == "File: a.txt | Showing lines 1-2 of 2 (11 chars total)\n1: alpha\n2: beta"


def test_read_file_range_and_right_aligned_numbers(test_awd) -> None:
    """start/end select an inclusive range; numbers are right-aligned to the width of the last line."""
    _make_lines(test_awd / "r.txt", 120)
    res = read_file("r.txt", start_line=98, end_line=101)
    lines = res.split("\n")
    assert lines[0] == "File: r.txt | Showing lines 98-101 of 120 (" + str(len("".join(
        f"line {i}\n" for i in range(1, 121)))) + " chars total)"
    assert lines[1:5] == [" 98: line 98", " 99: line 99", "100: line 100", "101: line 101"]
    assert lines[5] == '... 19 more lines. Call read_file("r.txt", start_line=102) to continue.'


def test_read_file_pagination_default_max_lines(test_awd) -> None:
    """Default max_lines is 500 and the footer says where to continue."""
    _make_lines(test_awd / "big.txt", 1200)
    res = read_file("big.txt")
    assert "Showing lines 1-500 of 1200" in res
    assert res.endswith('... 700 more lines. Call read_file("big.txt", start_line=501) to continue.')
    res2 = read_file("big.txt", start_line=1001)
    assert "Showing lines 1001-1200 of 1200" in res2
    assert "more lines" not in res2


def test_read_file_max_lines_hard_cap_and_floor(test_awd) -> None:
    """max_lines is capped at 2000 and floored at 1."""
    _write(test_awd / "big.txt", "x\n" * 3000)
    assert "Showing lines 1-2000 of 3000" in read_file("big.txt", max_lines=5000)
    assert "Showing lines 1-1 of 3000" in read_file("big.txt", max_lines=0)
    assert "Showing lines 5-14 of 3000" in read_file("big.txt", start_line=5, max_lines=10)


def test_read_file_end_beyond_eof_and_clamping(test_awd) -> None:
    """end_line beyond EOF is clamped; start_line < 1 is treated as 1."""
    _make_lines(test_awd / "s.txt", 10)
    assert "Showing lines 8-10 of 10" in read_file("s.txt", start_line=8, end_line=50)
    assert "Showing lines 1-3 of 10" in read_file("s.txt", start_line=-5, end_line=3)


def test_read_file_start_beyond_eof(test_awd) -> None:
    """start_line beyond the file is an error naming the total."""
    _make_lines(test_awd / "s.txt", 10)
    assert read_file("s.txt", start_line=11) == "Error: start_line 11 is beyond the end of the file (10 lines)."


def test_read_file_end_before_start(test_awd) -> None:
    """end_line before start_line is an error."""
    _make_lines(test_awd / "s.txt", 10)
    assert read_file("s.txt", start_line=5, end_line=2) == "Error: end_line 2 is before start_line 5."


def test_read_file_char_cap(test_awd) -> None:
    """Output stops at the last whole line under the 30000-char cap and says where to continue."""
    _write(test_awd / "wide.txt", "".join(f"{i:04d}" + "x" * 996 + "\n" for i in range(1, 101)))
    res = read_file("wide.txt")
    assert len(res) <= READ_FILE_MAX_CHARS
    shown = [line for line in res.split("\n")[1:] if not line.startswith("...")]
    last = int(shown[-1].split(":")[0])
    assert 20 < last < 100
    assert all(len(line) > 1000 for line in shown)  # whole lines only
    assert f"output capped at {READ_FILE_MAX_CHARS} chars; continue from line {last + 1}" in res
    assert f'read_file("wide.txt", start_line={last + 1})' in res
    assert f"Showing lines 1-{last} of 100" in res


def test_read_file_long_line_truncated(test_awd) -> None:
    """A single line over 2000 chars is truncated with a marker giving its length."""
    _write(test_awd / "min.js", "a" * 5000 + "\nshort\n")
    res = read_file("min.js")
    assert "1: " + "a" * 2000 + "… [line truncated, 5000 chars]" in res
    assert "2: short" in res


def test_read_file_keeps_tabs(test_awd) -> None:
    """Tabs inside lines are kept verbatim."""
    _write(test_awd / "t.mk", "all:\n\techo hi\n")
    assert "2: \techo hi" in read_file("t.mk")


def test_read_file_missing_with_suggestions(test_awd) -> None:
    """Missing files get repo-prefix and basename-walk suggestions."""
    _write(test_awd / "repo" / "dagrun" / "scheduler.py", "x\n")
    res = read_file("dagrun/scheduler.py")
    assert res == "Error: File 'dagrun/scheduler.py' not found. Did you mean: repo/dagrun/scheduler.py?"
    res2 = read_file("scheduler.py")
    assert "Did you mean: repo/dagrun/scheduler.py?" in res2
    assert read_file("nothing_like_it.py") == "Error: File 'nothing_like_it.py' not found."


def test_read_file_directory_listing(test_awd) -> None:
    """A directory yields a sorted listing with trailing '/' on subdirectories."""
    _write(test_awd / "pkg" / "b.py", "")
    _write(test_awd / "pkg" / "a.py", "")
    (test_awd / "pkg" / "sub").mkdir()
    assert read_file("pkg") == "Error: 'pkg' is a directory. Entries: a.py, b.py, sub/"


def test_read_file_directory_listing_capped(test_awd) -> None:
    """Directory listings show at most 50 entries."""
    for i in range(55):
        _write(test_awd / "many" / f"f{i:02d}.txt", "")
    res = read_file("many")
    assert "f49.txt" in res and "f50.txt" not in res
    assert "… (5 more)" in res


def test_read_file_binary(test_awd) -> None:
    """Files with NUL bytes in the first 8 KiB are reported as binary."""
    (test_awd / "img.bin").write_bytes(b"\x89PNG\x00\x01\x02" + b"z" * 10)
    res = read_file("img.bin")
    assert res.startswith("Error: 'img.bin' is a binary file (17 bytes)")
    assert "run_shell_command" in res


def test_read_file_empty(test_awd) -> None:
    """An empty file is reported, not an error."""
    _write(test_awd / "e.txt", "")
    assert read_file("e.txt") == "File 'e.txt' is empty (0 lines)."


def test_read_file_invalid_utf8_replaced(test_awd) -> None:
    """Invalid UTF-8 is decoded with replacement characters instead of failing."""
    (test_awd / "l1.txt").write_bytes(b"caf\xe9\n")
    assert "1: caf\ufffd" in read_file("l1.txt")


def test_read_file_crlf(test_awd) -> None:
    """CRLF line endings do not leak '\\r' into the output."""
    (test_awd / "w.txt").write_bytes(b"a\r\nb\r\n")
    res = read_file("w.txt")
    assert "1: a\n2: b" in res and "\r" not in res


def test_read_file_output_roundtrips_into_update_file(test_awd) -> None:
    """Lines copied from read_file (with prefixes) can be fed back to update_file."""
    _write(test_awd / "rt.py", "".join(f"v{i} = {i}\n" for i in range(1, 13)))
    out = read_file("rt.py", start_line=9, end_line=11)
    block = "\n".join(out.split("\n")[1:4]) + "\n"
    assert block.startswith(" 9: v9 = 9")
    res = update_file("rt.py", block, "v9 = 0\n")
    assert res.startswith("Success"), res
    assert "v10" not in (test_awd / "rt.py").read_text(encoding="utf-8")


def test_read_file_never_raises_on_os_error(test_awd, monkeypatch) -> None:
    """OS errors become error strings."""
    _write(test_awd / "x.txt", "x\n")

    def _boom(*args: object, **kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(file_mod, "open", _boom, raising=False)
    assert read_file("x.txt").startswith("Error reading file 'x.txt': denied")


# ---------------------------------------------------------------------------------------------------------------
# $STAGING_DIR / session notes
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def staging_cfg(tmp_path):
    """AWD + sessions_dir under tmp_path, restored afterwards."""
    cfg = get_config()
    old_awd, old_sessions = cfg.agent_working_dir, cfg.workspace.sessions_dir
    cfg.agent_working_dir = str(tmp_path / "awd")
    cfg.workspace.sessions_dir = str(tmp_path / "sessions")
    os.makedirs(cfg.agent_working_dir, exist_ok=True)
    yield tmp_path
    cfg.agent_working_dir, cfg.workspace.sessions_dir = old_awd, old_sessions


@pytest.mark.parametrize("path", ["$STAGING_DIR/notes.md", "${STAGING_DIR}/notes.md"])
def test_update_file_expands_staging_dir_for_session_notes(staging_cfg, path: str) -> None:
    """The literal $STAGING_DIR path from the prompt lands in the session staging dir, not under AWD."""
    ctx = ToolContext(session_id="sess_x", session_workspace="sess_x")
    res = update_file(path, None, "- rule A\n- scene: park", context=ctx)
    assert res.startswith("Success"), res
    notes = staging_cfg / "sessions" / "sess_x" / "notes.md"
    assert notes.read_text(encoding="utf-8") == "- rule A\n- scene: park"
    assert not (staging_cfg / "awd" / "$STAGING_DIR").exists()
    assert not (staging_cfg / "awd" / "${STAGING_DIR}").exists()

    # Incremental edit: only the changed line is replaced, the other rule survives.
    res2 = update_file(path, "- scene: park", "- scene: museum", context=ctx)
    assert res2.startswith("Success"), res2
    assert notes.read_text(encoding="utf-8") == "- rule A\n- scene: museum"


def test_read_file_expands_staging_dir(staging_cfg) -> None:
    """read_file resolves $STAGING_DIR from the session context."""
    ctx = ToolContext(session_id="sess_r", session_workspace="sess_r")
    _write(staging_cfg / "sessions" / "sess_r" / "cmd_output_1.txt", "out 1\nout 2\n")
    res = read_file("$STAGING_DIR/cmd_output_1.txt", start_line=2, context=ctx)
    assert res.startswith("File: $STAGING_DIR/cmd_output_1.txt | Showing lines 2-2 of 2")
    assert res.endswith("2: out 2")


def test_update_file_notes_edit_requires_existing_old_content(staging_cfg) -> None:
    """Editing notes with stale old_content fails loudly instead of silently rewriting the file."""
    ctx = ToolContext(session_id="sess_y", session_workspace="sess_y")
    update_file("$STAGING_DIR/notes.md", None, "- rule A\n- rule B", context=ctx)
    res = update_file("$STAGING_DIR/notes.md", "- rule Z", "- rule C", context=ctx)
    assert "not found" in res
    notes = staging_cfg / "sessions" / "sess_y" / "notes.md"
    assert notes.read_text(encoding="utf-8") == "- rule A\n- rule B"


def test_update_file_notes_create_twice_rejected(staging_cfg) -> None:
    """old_content=None on an existing notes.md is rejected (no accidental full overwrite)."""
    ctx = ToolContext(session_id="sess_z", session_workspace="sess_z")
    update_file("$STAGING_DIR/notes.md", None, "- rule A", context=ctx)
    res = update_file("$STAGING_DIR/notes.md", None, "- replaced everything", context=ctx)
    assert res.startswith("Error")
    notes = staging_cfg / "sessions" / "sess_z" / "notes.md"
    assert notes.read_text(encoding="utf-8") == "- rule A"


@pytest.mark.parametrize("path", ["$MEMORY_DIR/topics/python.md", "${MEMORY_DIR}/topics/python.md"])
def test_read_and_update_file_expands_memory_dir(tmp_path, path: str) -> None:
    """$MEMORY_DIR expands to the active role's memories directory for both update_file and read_file."""
    from unittest.mock import MagicMock

    cfg = get_config()
    old_awd, old_roles = cfg.agent_working_dir, cfg.workspace.roles_dir
    cfg.agent_working_dir = str(tmp_path / "awd")
    cfg.workspace.roles_dir = str(tmp_path / "roles")
    os.makedirs(cfg.agent_working_dir, exist_ok=True)
    try:
        mock_gw = MagicMock()
        mock_sess = MagicMock()
        mock_sess.role_name = "coder"
        mock_gw.db._sync.get_session.return_value = mock_sess

        ctx = ToolContext(session_id="sess_mem", session_workspace="sess_mem", gateway=mock_gw)
        content = "---\ndescription: Python conventions\n---\n- Use uv run\n"
        res = update_file(path, None, content, context=ctx)
        assert res.startswith("Success"), res

        target = tmp_path / "roles" / "coder" / "memories" / "topics" / "python.md"
        assert target.exists()
        assert target.read_text(encoding="utf-8") == content

        read_out = read_file(path, context=ctx)
        assert "description: Python conventions" in read_out
        assert "- Use uv run" in read_out
    finally:
        cfg.agent_working_dir, cfg.workspace.roles_dir = old_awd, old_roles


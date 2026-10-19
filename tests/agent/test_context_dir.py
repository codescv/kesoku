"""Unit tests for per-turn context directory injection (kesoku.agent.context_dir)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from kesoku.agent import context_dir
from kesoku.agent.context_dir import (
    CONTEXT_FILE_MAX_CHARS,
    TRUNCATED_MARKER,
    ContextEntry,
    apply_total_budget,
    build_context_prefix,
    collect_context_entries,
    context_block_name,
    insert_extra_entries,
    render_context_blocks,
    truncate_context_content,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("20-scene.md", "scene"),
        ("99-test.md", "test"),
        ("05_todo.md", "todo"),
        ("notes.md", "notes"),
        ("plain.md", "plain"),
        ("10-multi-part-name.md", "multi-part-name"),
        ("2024.md", "2024"),
        ("42-.md", "42-"),
    ],
)
def test_context_block_name(filename: str, expected: str) -> None:
    """Leading numeric prefixes and the .md extension are stripped."""
    assert context_block_name(filename) == expected


def test_context_block_name_escapes_quotes() -> None:
    """Double quotes are replaced so the name attribute stays well-formed."""
    assert '"' not in context_block_name('10-a"b.md')


def test_truncate_context_content() -> None:
    """Content over the per-file cap is cut at the tail and marked."""
    assert truncate_context_content("short") == "short"
    long = "x" * (CONTEXT_FILE_MAX_CHARS + 100)
    out = truncate_context_content(long)
    assert out.endswith(TRUNCATED_MARKER)
    assert out.startswith("x" * CONTEXT_FILE_MAX_CHARS)
    assert len(out) == CONTEXT_FILE_MAX_CHARS + 1 + len(TRUNCATED_MARKER)


@pytest.mark.asyncio
async def test_collect_empty_or_missing(tmp_path: Path) -> None:
    """No notes and no context dir (or an empty one) yield no entries."""
    assert await collect_context_entries(str(tmp_path)) == []
    (tmp_path / "context").mkdir()
    assert await collect_context_entries(str(tmp_path)) == []
    assert await build_context_prefix(str(tmp_path)) == ""
    assert await collect_context_entries("") == []


@pytest.mark.asyncio
async def test_collect_sorted_with_notes_first(tmp_path: Path) -> None:
    """notes.md comes first, then context/*.md sorted by filename; non-md and empty files are skipped."""
    (tmp_path / "notes.md").write_text("  - rule A  \n")
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "30-todo.md").write_text("todo list")
    (ctx / "10-rules.md").write_text("rule text")
    (ctx / "20-scene.md").write_text("scene text")
    (ctx / "40-empty.md").write_text("   \n")
    (ctx / "data.json").write_text("{}")
    (ctx / "sub.md").mkdir()

    entries = await collect_context_entries(str(tmp_path))
    assert [e.name for e in entries] == ["notes", "rules", "scene", "todo"]
    assert entries[0].content == "- rule A"


@pytest.mark.asyncio
async def test_collect_truncates_each_file(tmp_path: Path) -> None:
    """Each file is individually capped."""
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "big.md").write_text("y" * (CONTEXT_FILE_MAX_CHARS * 2))
    entries = await collect_context_entries(str(tmp_path))
    assert entries[0].content.endswith(TRUNCATED_MARKER)
    assert len(entries[0].content) < CONTEXT_FILE_MAX_CHARS + 20


def test_apply_total_budget_drops_from_end() -> None:
    """Entries at the end of the sorted list are dropped first, with a warning."""
    entries = [ContextEntry("notes", "a" * 10), ContextEntry("b", "b" * 10), ContextEntry("c", "c" * 10)]
    with patch.object(context_dir.logger, "warning") as warn:
        kept = apply_total_budget(entries, 25)
    assert [e.name for e in kept] == ["notes", "b"]
    warn.assert_called_once()
    assert "dropped" in warn.call_args[0][0]
    assert apply_total_budget(entries, 30) == entries
    assert apply_total_budget(entries, 5) == []


def test_render_context_blocks() -> None:
    """Entries render as consecutive <context name="..."> blocks."""
    out = render_context_blocks([ContextEntry("notes", "n"), ContextEntry("test", "今天的暗号是🍊")])
    assert out == '<context name="notes">\nn\n</context>\n<context name="test">\n今天的暗号是🍊\n</context>\n'
    assert render_context_blocks([]) == ""


@pytest.mark.asyncio
async def test_build_context_prefix_end_to_end(tmp_path: Path) -> None:
    """Full pipeline: notes compatibility, sorting and the total budget."""
    (tmp_path / "notes.md").write_text("note")
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "99-test.md").write_text("今天的暗号是🍊")
    (ctx / "50-big.md").write_text("z" * 5000)

    prefix = await build_context_prefix(str(tmp_path), 20000)
    assert prefix.index('<context name="notes">') < prefix.index('<context name="big">')
    assert prefix.index('<context name="big">') < prefix.index('<context name="test">')
    assert "<session_notes>" not in prefix

    # A tight budget drops the last file in sort order first (test), then big.
    prefix_small = await build_context_prefix(str(tmp_path), 5010)
    assert '<context name="test">' not in prefix_small
    assert '<context name="big">' in prefix_small
    prefix_tiny = await build_context_prefix(str(tmp_path), 10)
    assert prefix_tiny == '<context name="notes">\nnote\n</context>\n'


@pytest.mark.asyncio
async def test_context_file_named_notes_is_renamed(tmp_path: Path) -> None:
    """A tool-written context/notes.md (or NN-notes.md) never renders as a second notes block."""
    (tmp_path / "notes.md").write_text("pinned")
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "10-notes.md").write_text("tool data")
    (ctx / "notes.md").write_text("more tool data")
    with patch.object(context_dir.logger, "warning") as warn:
        entries = await collect_context_entries(str(tmp_path))
    assert [e.name for e in entries] == ["notes", "notes-1", "notes-1"]
    assert entries[0].content == "pinned"
    assert warn.call_count == 2
    prefix = render_context_blocks(entries)
    assert prefix.count('<context name="notes">') == 1


@pytest.mark.asyncio
async def test_build_context_prefix_extra_entries_after_notes(tmp_path: Path) -> None:
    """Extra (system snapshot) entries go right after notes and before context/*.md, in the given order."""
    (tmp_path / "notes.md").write_text("note")
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "10-tool.md").write_text("tool data")
    extras = [ContextEntry("running_jobs", "jobs"), ContextEntry("staging_artifacts", "files")]

    prefix = await build_context_prefix(str(tmp_path), 20000, extra_entries=extras)
    order = [
        prefix.index('<context name="notes">'),
        prefix.index('<context name="running_jobs">'),
        prefix.index('<context name="staging_artifacts">'),
        prefix.index('<context name="tool">'),
    ]
    assert order == sorted(order)


@pytest.mark.asyncio
async def test_build_context_prefix_extra_entries_without_notes(tmp_path: Path) -> None:
    """Without notes, extra entries come first; empty extra entries are skipped; works without a context dir."""
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "10-tool.md").write_text("tool data")
    extras = [ContextEntry("running_jobs", "jobs"), ContextEntry("staging_artifacts", "")]
    prefix = await build_context_prefix(str(tmp_path), extra_entries=extras)
    assert prefix.startswith('<context name="running_jobs">\njobs\n</context>\n')
    assert "staging_artifacts" not in prefix
    assert '<context name="tool">' in prefix

    # Only extras, empty staging dir.
    empty = tmp_path / "empty"
    empty.mkdir()
    only = await build_context_prefix(str(empty), extra_entries=[ContextEntry("running_jobs", "jobs")])
    assert only == '<context name="running_jobs">\njobs\n</context>\n'
    # Default (no extras) is unchanged behaviour.
    assert await build_context_prefix(str(empty)) == ""


@pytest.mark.asyncio
async def test_build_context_prefix_budget_drops_tool_files_before_extras(tmp_path: Path) -> None:
    """The drop-from-the-end budget sacrifices context/*.md files before the system snapshots."""
    (tmp_path / "notes.md").write_text("n" * 10)
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "10-tool.md").write_text("t" * 100)
    extras = [ContextEntry("running_jobs", "j" * 50)]

    prefix = await build_context_prefix(str(tmp_path), 70, extra_entries=extras)
    assert '<context name="notes">' in prefix
    assert '<context name="running_jobs">' in prefix
    assert '<context name="tool">' not in prefix


def test_insert_extra_entries() -> None:
    """insert_extra_entries does not mutate its input and places extras after a leading notes entry only."""
    entries = [ContextEntry("notes", "n"), ContextEntry("a", "a")]
    out = insert_extra_entries(entries, [ContextEntry("x", "x")])
    assert [e.name for e in out] == ["notes", "x", "a"]
    assert [e.name for e in entries] == ["notes", "a"]
    assert [e.name for e in insert_extra_entries([ContextEntry("a", "a")], [ContextEntry("x", "x")])] == ["x", "a"]
    assert insert_extra_entries(entries, ()) == entries

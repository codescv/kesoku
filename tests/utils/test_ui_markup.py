"""Unit tests for kesoku.utils.ui_markup (```ui blocks, <details>, multiselect answers)."""

import pytest

from kesoku.utils.ui_markup import (
    UiQuestion,
    collapse_ui_blocks,
    extract_ui_options,
    find_first_ui_block,
    format_multi_selection,
    parse_multi_selection,
    parse_ui_block_body,
    render_details_as_text,
    render_question_as_text,
    resolve_answer,
)


def test_parse_select_body() -> None:
    """A strict-YAML select body parses with max=1."""
    q = parse_ui_block_body("type: select\nprompt: P?\noptions:\n  - a\n  - b [x]\n")
    assert q == UiQuestion(prompt="P?", options=["a", "b [x]"], multi=False, max=1)


def test_parse_select_ignores_max() -> None:
    """max is only meaningful for multiselect."""
    q = parse_ui_block_body("type: select\nprompt: P\noptions: [a, b]\nmax: 2\n")
    assert q is not None and q.max == 1


def test_parse_multiselect_body() -> None:
    """Multiselect max defaults to len(options), is clamped, and accepts numeric strings."""
    assert parse_ui_block_body("type: multiselect\nprompt: P\noptions: [a, b, c]\n").max == 3
    assert parse_ui_block_body("type: multiselect\nprompt: P\noptions: [a, b, c]\nmax: 0\n").max == 1
    assert parse_ui_block_body("type: MultiSelect\nprompt: P\noptions: [a, b, c]\nmax: '2'\n").max == 2


def test_parse_with_comments_like_prompt_example() -> None:
    """YAML comments (as in the prompt example) are ignored."""
    body = "type: multiselect   # select | multiselect\nprompt: P\noptions:\n  - a\n  - b\nmax: 1   # only multi\n"
    q = parse_ui_block_body(body)
    assert q is not None and q.multi and q.max == 1 and q.options == ["a", "b"]


def test_lenient_fallback_keeps_options_verbatim() -> None:
    """Options that YAML would turn into lists/bools/numbers are read verbatim from their lines."""
    q = parse_ui_block_body('type: select\nprompt: "Q: x"\noptions:\n  - [a] first\n  - yes\n  - 2\n  - "quoted"\n')
    assert q is not None
    assert q.options == ["[a] first", "yes", "2", "quoted"]


def test_parse_drops_empty_options_and_rejects_invalid() -> None:
    """Blank options are dropped; missing pieces return None."""
    assert parse_ui_block_body("type: select\nprompt: P\noptions:\n  - ''\n  - a\n").options == ["a"]
    assert parse_ui_block_body("type: select\nprompt: P\noptions: []\n") is None
    assert parse_ui_block_body("type: select\nprompt: ''\noptions: [a]\n") is None
    assert parse_ui_block_body("prompt: P\noptions: [a]\n") is None
    assert parse_ui_block_body("- just\n- a list\n") is None


def test_find_first_ui_block() -> None:
    """Only the first fenced block is returned; text without blocks returns (None, None)."""
    text = "x\n```ui\ntype: select\nprompt: A\noptions: [1]\n```\n```ui\ntype: select\nprompt: B\noptions: [2]\n```"
    match, q = find_first_ui_block(text)
    assert match is not None and q is not None and q.prompt == "A"
    assert find_first_ui_block("no blocks") == (None, None)
    assert find_first_ui_block(None) == (None, None)
    # Inline mention without a fenced line is not a block.
    assert find_first_ui_block("use ```ui blocks") == (None, None)


def test_find_first_ui_block_invalid_returns_match_without_question() -> None:
    """An invalid first block yields a match but no question."""
    match, q = find_first_ui_block("```ui\nnot: valid\n```")
    assert match is not None and q is None


def test_extract_ui_options() -> None:
    """Options and multi flag of the interactive block."""
    assert extract_ui_options("```ui\ntype: multiselect\nprompt: P\noptions: [a, b]\n```") == (["a", "b"], True)
    assert extract_ui_options("plain") == ([], False)


def test_multi_selection_round_trip() -> None:
    """Multiselect answers are formatted as 选择：A；B and parsed back."""
    text = format_multi_selection(["A", "B C"])
    assert text == "选择：A；B C"
    assert parse_multi_selection(text) == ["A", "B C"]
    assert parse_multi_selection("  选择：A；；B ") == ["A", "B"]
    assert parse_multi_selection("A；B") is None
    assert parse_multi_selection("选择：") is None
    assert parse_multi_selection(None) is None


@pytest.mark.parametrize(
    ("multi", "reply", "expected"),
    [
        (False, "a", "a"),
        (False, " b ", "b"),
        (False, "选择：a；b", None),
        (False, "other", None),
        (True, "选择：a；c", "a；c"),
        (True, "a", "a"),
        (True, "选择：a；z", None),
        (True, "", None),
    ],
)
def test_resolve_answer(multi: bool, reply: str, expected: str | None) -> None:
    """Answers resolve only when they exactly match options."""
    q = UiQuestion(prompt="P", options=["a", "b", "c"], multi=multi, max=3 if multi else 1)
    assert resolve_answer(q, reply) == expected


def test_collapse_ui_blocks() -> None:
    """Valid blocks collapse to prompt (+answer for the first block only); invalid blocks stay."""
    first = "```ui\ntype: select\nprompt: A?\noptions: [x, y]\n```"
    second = "```ui\ntype: select\nprompt: B?\noptions: [x, z]\n```"
    invalid = "```ui\nbroken\n```"
    out = collapse_ui_blocks(f"t\n{first}\n{second}\n{invalid}", "x")
    assert out == f"t\n```ui\nprompt: A?\nanswered: x\n```\n```ui\nprompt: B?\n```\n{invalid}"
    assert collapse_ui_blocks(out, "x") == out  # idempotent on collapsed output
    assert collapse_ui_blocks("plain", "x") == "plain"
    assert collapse_ui_blocks(first) == "```ui\nprompt: A?\n```"


def test_render_fallbacks() -> None:
    """Plain-text renderings for platforms without interactive components."""
    assert render_details_as_text(" T ", "\nB\n") == "**T**\n\nB"
    assert render_details_as_text("", "B") == "B"
    assert render_details_as_text("T", "") == "**T**"
    assert render_question_as_text("Q?", ["a", "b"]) == "Q?\n1. a\n2. b"
    assert render_question_as_text("Q?", ["a", "b"], multi=True, max_choices=2) == "Q?\n(select up to 2)\n1. a\n2. b"

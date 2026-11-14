"""Unit tests for kesoku.utils.ui_markup (```ui blocks, <details>, multiselect answers)."""

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
    """Blank options are dropped; missing pieces return None; omitted type defaults to select."""
    assert parse_ui_block_body("type: select\nprompt: P\noptions:\n  - ''\n  - a\n").options == ["a"]
    assert parse_ui_block_body("type: select\nprompt: P\noptions: []\n") is None
    assert parse_ui_block_body("type: select\nprompt: ''\noptions: [a]\n") is None
    assert parse_ui_block_body("prompt: P\noptions: [a]\n") == UiQuestion(prompt="P", options=["a"], multi=False, max=1)
    assert parse_ui_block_body("type: unknown\nprompt: P\noptions: [a]\n") is None
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


def test_collapse_ui_blocks() -> None:
    """Valid blocks collapse to plain prompt text (stripping fences and options); invalid blocks stay."""
    first = "```ui\ntype: select\nprompt: A?\noptions: [x, y]\n```"
    second = "```ui\nprompt: B?\noptions: [x, z]\n```"
    invalid = "```ui\nbroken\n```"
    out = collapse_ui_blocks(f"t\n{first}\n{second}\n{invalid}")
    assert out == f"t\nA?\nB?\n{invalid}"
    assert collapse_ui_blocks(out) == out  # idempotent on collapsed output
    assert collapse_ui_blocks("plain") == "plain"
    assert collapse_ui_blocks(first) == "A?"


def test_render_fallbacks() -> None:
    """Plain-text renderings for platforms without interactive components."""
    assert render_details_as_text(" T ", "\nB\n") == "**T**\n\nB"
    assert render_details_as_text("", "B") == "B"
    assert render_details_as_text("T", "") == "**T**"
    assert render_question_as_text("Q?", ["a", "b"]) == "Q?\n1. a\n2. b"
    assert render_question_as_text("Q?", ["a", "b"], multi=True, max_choices=2) == "Q?\n(select up to 2)\n1. a\n2. b"


def test_inline_bilingual_with_spoiler_and_voice() -> None:
    from kesoku.gateway.chatbot.base import parse_message_content

    raw = (
        "舞台灯光亮起。\n\n"
        "> 今夜(こんや)はどの物語(ものがたり)にしようか、小張(シャオジャン)？\n"
        "<spoiler>今晚要听哪一个故事呀，小张？</spoiler>\n"
        "[voice: /tmp/theater_open.wav]\n\n"
        "选一个方向吧。"
    )
    segs = parse_message_content(raw, rich_blocks=True)
    bilinguals = [s for s in segs if s["type"] == "bilingual"]
    voices = [s for s in segs if s["type"] == "voice"]
    assert len(bilinguals) == 1
    assert len(voices) == 0  # folded into bilingual segment
    b = bilinguals[0]
    assert b["ja_plain"] == "今夜はどの物語にしようか、小張？"
    assert b["zh"] == "今晚要听哪一个故事呀，小张？"
    assert b["voice"] == {"ref": "/tmp/theater_open.wav"}


def test_inline_bilingual_fullwidth_ruby_and_plain_zh() -> None:
    from kesoku.gateway.chatbot.base import parse_message_content

    raw = (
        "> 「……ふふ、そんなに熱心（ねっしん）に見（み）つめられたら、ちょっと照（て）れちゃうじゃない。」\n"
        "“……呵呵，被你这么热切地盯着看，我也会有点害羞的呀。”"
    )
    segs = parse_message_content(raw, rich_blocks=True)
    assert len(segs) == 1
    b = segs[0]
    assert b["type"] == "bilingual"
    assert "熱心" in b["ja_plain"]
    assert any(p.get("rt") == "ねっしん" for p in b["ruby_parts"])
    assert b["zh"] == "“……呵呵，被你这么热切地盯着看，我也会有点害羞的呀。”"

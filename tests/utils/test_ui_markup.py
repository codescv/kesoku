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


_MULTI_LINE_QUOTE = (
    "### ① 原文\n"
    "> 森(もり)さんは　毎晩(まいばん)　テレビを　見(み)る。\n"
    "> 甲(こう)：明日(あした)　ボーリングに　行(い)かない？ / "
    "乙(おつ)：ごめん。　明日(あした)は　仕事(しごと)が　あるから…\n"
    "<spoiler>森先生每天晚上看电视。\n"
    "甲：明天去打保龄球怎么样？ / 乙：抱歉，明天还有工作……</spoiler>\n\n"
    "### ② 下一题"
)


def test_inline_bilingual_multi_line_quote_and_multi_line_spoiler() -> None:
    """Consecutive `> ja` lines + a `<spoiler>` spanning lines become ONE bilingual segment keeping `\\n`."""
    from kesoku.gateway.chatbot.base import parse_message_content

    segs = parse_message_content(_MULTI_LINE_QUOTE, rich_blocks=True)
    assert [s["type"] for s in segs] == ["text", "bilingual", "text"]
    b = segs[1]
    assert b["ja_plain"] == (
        "森さんは　毎晩　テレビを　見る。\n甲：明日　ボーリングに　行かない？ / 乙：ごめん。　明日は　仕事が　あるから…"
    )
    assert b["zh"] == "森先生每天晚上看电视。\n甲：明天去打保龄球怎么样？ / 乙：抱歉，明天还有工作……"
    assert "\n" in b["ja_html"] and "<ruby>森<rt>もり</rt></ruby>" in b["ja_html"]
    assert any(p["base"] == "仕事" and p.get("rt") == "しごと" for p in b["ruby_parts"])
    assert "".join(p["base"] for p in b["ruby_parts"]) == b["ja_plain"]
    assert "speaker" not in b  # multi-line quotes never guess a speaker
    assert all("<spoiler>" not in (s.get("content") or "") for s in segs)
    assert segs[2]["content"].strip() == "### ② 下一题"


def test_inline_bilingual_multi_line_variants() -> None:
    """Single-line spoiler after multiple quote lines, `||bar||`, voice tag, and the kana requirement."""
    from kesoku.gateway.chatbot.base import parse_message_content
    from kesoku.utils.ui_markup import INLINE_BILINGUAL_RE, parse_inline_bilingual_match

    raw = "> 今日(きょう)は\n> いい天気(てんき)ですね。\n<spoiler>今天天气真好。</spoiler>\n[voice: /tmp/a.wav]"
    segs = parse_message_content(raw, rich_blocks=True)
    assert len(segs) == 1 and segs[0]["type"] == "bilingual"
    assert segs[0]["ja_plain"] == "今日は\nいい天気ですね。"
    assert segs[0]["zh"] == "今天天气真好。"
    assert segs[0]["voice"] == {"ref": "/tmp/a.wav"}

    bar = parse_message_content("> 行(い)こう。\n> 早(はや)く！\n||走吧。\n快点！||", rich_blocks=True)
    assert bar[0]["type"] == "bilingual" and bar[0]["zh"] == "走吧。\n快点！"

    # Kana only on the second quote line still counts.
    m = INLINE_BILINGUAL_RE.search("> 東京\n> へ行きます\n<spoiler>去东京</spoiler>")
    assert m is not None
    seg = parse_inline_bilingual_match(m)
    assert seg is not None and seg["ja_plain"] == "東京\nへ行きます"

    # A quote with no kana at all is never a bilingual pair.
    assert INLINE_BILINGUAL_RE.search("> 中文引用\n> 第二行\n<spoiler>翻译</spoiler>") is None
    # A spoiler body is capped (no runaway match across a code fence).
    fenced = "> 行(い)こう。\n<spoiler>走吧\n```\ncode\n```\n</spoiler>"
    assert all(s["type"] != "bilingual" for s in parse_message_content(fenced, rich_blocks=True))


def test_inline_bilingual_single_line_behaviour_unchanged() -> None:
    """The classic 2-line form keeps speaker extraction."""
    from kesoku.gateway.chatbot.base import parse_message_content

    segs = parse_message_content("> ハナ：おはよう！\n<spoiler>花：早上好！</spoiler>", rich_blocks=True)
    assert segs[0]["type"] == "bilingual"
    assert segs[0]["speaker"] == "花" and segs[0]["ja_plain"] == "おはよう！" and segs[0]["zh"] == "早上好！"


def test_parse_structured_option_items_and_layout() -> None:
    """Dict options derive canonical string values while metadata goes to `option_items`."""
    body = (
        "type: select\n"
        "prompt: Pick a partner\n"
        "layout: Cards\n"
        "options:\n"
        "  - {name: Alpha, subtitle: Lead, avatar: '[file: chars/alpha.png]'}\n"
        "  - {label: Beta}\n"
        "  - {value: gamma-id, name: Gamma, desc: Support}\n"
        "  - Plain option\n"
        "  - {subtitle: no name}\n"
    )
    q = parse_ui_block_body(body)
    assert q is not None
    assert q.options == ["Alpha（Lead）", "Beta", "gamma-id", "Plain option"]
    assert all(isinstance(o, str) for o in q.options)
    assert q.layout == "cards"
    assert q.option_items == (
        {"value": "Alpha（Lead）", "name": "Alpha", "subtitle": "Lead", "avatar": "chars/alpha.png"},
        {"value": "Beta", "name": "Beta"},
        {"value": "gamma-id", "name": "Gamma", "subtitle": "Support"},
        {"value": "Plain option"},
    )
    seg = q.to_segment(rich=True)
    assert seg["layout"] == "cards" and seg["option_items"][0]["avatar"] == "chars/alpha.png"
    plain = q.to_segment()
    assert "layout" not in plain and "option_items" not in plain
    assert plain["choices"] == q.options


def test_plain_options_keep_legacy_shape() -> None:
    """Without dict options `option_items` stays empty; unknown layouts are dropped; numbers stay verbatim."""
    q = parse_ui_block_body("prompt: P\nlayout: grid\noptions:\n  - 1.10\n  - b\n")
    assert q is not None and q.options == ["1.10", "b"]
    assert q.layout is None and q.option_items == ()
    assert q.to_segment(rich=True) == q.to_segment()
    listed = parse_ui_block_body("prompt: P\nlayout: list\noptions: [a, b]\n")
    assert listed is not None and listed.layout == "list" and listed.to_segment(rich=True)["layout"] == "list"


def test_structured_options_stay_compatible_with_plain_helpers() -> None:
    """`extract_ui_options` / `collapse_ui_blocks` keep working with dict options."""
    text = "Intro\n```ui\nprompt: Who?\noptions:\n  - {name: Alpha, subtitle: Lead}\n  - Other\n```\n"
    assert extract_ui_options(text) == (["Alpha（Lead）", "Other"], False)
    assert collapse_ui_blocks(text).strip() == "Intro\nWho?"


def test_accidental_yaml_mapping_options_stay_verbatim() -> None:
    """`- Beta [file: x]` (a YAML mapping by accident) and unknown-key dicts keep the legacy verbatim parsing."""
    q = parse_ui_block_body("prompt: P\noptions:\n  - Alpha\n  - Beta [file: /tmp/x.png]\n  - 'Note: later'\n")
    assert q is not None and q.options == ["Alpha", "Beta [file: /tmp/x.png]", "Note: later"]
    assert q.option_items == ()
    mixed = parse_ui_block_body("prompt: P\noptions:\n  - {Name: Alpha, Avatar: a.png}\n  - Other\n")
    assert mixed is not None and mixed.option_items[0] == {"value": "Alpha", "name": "Alpha", "avatar": "a.png"}


def test_parse_multiselect_option_groups_and_subtitles() -> None:
    """Structured options preserve `group` and `subtitle` in `option_items` and `to_segment(rich=True)`."""
    body = (
        "type: multiselect\n"
        "prompt: Choose upgrades\n"
        "options:\n"
        "  - {name: Slash, subtitle: 'Replace Strike (+2 dmg)', group: Alice}\n"
        "  - {name: Guard, desc: 'Keep current kit', Group: Alice}\n"
        "  - {name: Fireball, subtitle: 'AoE', group: Bob}\n"
        "  - {name: Extra Potion, group: '   '}\n"
        "  - Plain bonus\n"
    )
    q = parse_ui_block_body(body)
    assert q is not None
    assert q.multi is True
    assert q.max == 5
    assert q.layout is None
    assert q.options == [
        "Slash（Replace Strike (+2 dmg)）",
        "Guard（Keep current kit）",
        "Fireball（AoE）",
        "Extra Potion",
        "Plain bonus",
    ]
    assert q.option_items == (
        {
            "value": "Slash（Replace Strike (+2 dmg)）",
            "name": "Slash",
            "subtitle": "Replace Strike (+2 dmg)",
            "group": "Alice",
        },
        {
            "value": "Guard（Keep current kit）",
            "name": "Guard",
            "subtitle": "Keep current kit",
            "group": "Alice",
        },
        {
            "value": "Fireball（AoE）",
            "name": "Fireball",
            "subtitle": "AoE",
            "group": "Bob",
        },
        {"value": "Extra Potion", "name": "Extra Potion"},
        {"value": "Plain bonus"},
    )
    rich_seg = q.to_segment(rich=True)
    assert "layout" not in rich_seg
    assert rich_seg["option_items"][0]["group"] == "Alice"
    assert rich_seg["option_items"][2]["group"] == "Bob"
    assert "group" not in rich_seg["option_items"][3]
    plain_seg = q.to_segment()
    assert "option_items" not in plain_seg

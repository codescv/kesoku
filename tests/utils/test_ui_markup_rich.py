"""Unit tests for the rich fenced blocks in kesoku.utils.ui_markup (```bilingual / ```card) and their
integration with kesoku.gateway.chatbot.base.parse_message_content."""

import json

import pytest

from kesoku.gateway.chatbot.base import parse_message_content
from kesoku.utils.ui_markup import (
    CARD_FALLBACK_MAX_LINES,
    CARD_FALLBACK_VALUE_MAX_CHARS,
    CARD_KIND_RE,
    RICH_BLOCK_RE,
    card_media_refs,
    flatten_rich_blocks,
    parse_bilingual_block_body,
    parse_card_block_body,
    parse_rich_block,
    render_bilingual_as_text,
    render_card_as_text,
    render_rich_block_as_text,
)


def _block(kind: str, body: str) -> str:
    """Wrap a YAML body in a fenced rich block."""
    return f"```{kind}\n{body.rstrip()}\n```"


def _types(segments: list[dict]) -> list[str]:
    return [s["type"] for s in segments]


# ---------------------------------------------------------------------------------------------------------------
# ```bilingual
# ---------------------------------------------------------------------------------------------------------------


def test_bilingual_inline_paren_ruby() -> None:
    """`漢字(かな)` / full-width parens annotate the preceding kanji run; ja_plain drops the readings."""
    seg = parse_bilingual_block_body("ja: 今日(きょう)は天気（てんき）がいい\nzh: 今天天气很好\n")
    assert seg is not None
    assert seg["type"] == "bilingual"
    assert seg["ja"] == "今日(きょう)は天気（てんき）がいい"
    assert seg["ja_plain"] == "今日は天気がいい"
    assert seg["zh"] == "今天天气很好"
    assert seg["ruby_parts"] == [
        {"base": "今日", "rt": "きょう"},
        {"base": "は"},
        {"base": "天気", "rt": "てんき"},
        {"base": "がいい"},
    ]
    assert seg["ja_html"] == "<ruby>今日<rt>きょう</rt></ruby>は<ruby>天気<rt>てんき</rt></ruby>がいい"
    assert "ruby" not in seg


def test_bilingual_inline_brace_ruby() -> None:
    """`{base|rt}` works for any base text, including kana and latin."""
    seg = parse_bilingual_block_body("ja: '{明日|あした}も{ABC|えーびーしー}'\n")
    assert seg is not None
    assert seg["ja_plain"] == "明日もABC"
    assert seg["ruby_parts"] == [
        {"base": "明日", "rt": "あした"},
        {"base": "も"},
        {"base": "ABC", "rt": "えーびーしー"},
    ]
    assert seg["zh"] == ""


def test_bilingual_plain_text_without_ruby() -> None:
    """Plain ja yields a single unannotated part."""
    seg = parse_bilingual_block_body("ja: こんにちは\nzh: 你好\n")
    assert seg is not None
    assert seg["ruby_parts"] == [{"base": "こんにちは"}]
    assert seg["ja_html"] == "こんにちは"


def test_bilingual_ruby_string_matching_ja() -> None:
    """A ruby string whose plain text equals ja supplies the readings."""
    seg = parse_bilingual_block_body("ja: 今日は雨\nruby: 今日(きょう)は雨(あめ)\n")
    assert seg is not None
    assert seg["ja_plain"] == "今日は雨"
    assert seg["ruby_parts"] == [{"base": "今日", "rt": "きょう"}, {"base": "は"}, {"base": "雨", "rt": "あめ"}]
    assert seg["ruby"] == "今日(きょう)は雨(あめ)"


def test_bilingual_ruby_string_disagreeing_with_ja_keeps_ja_text() -> None:
    """When the ruby string's text differs from ja, ja wins and the readings are applied where they fit."""
    seg = parse_bilingual_block_body("ja: 今日は雨です\nruby: 今日(きょう)は雪(ゆき)\n")
    assert seg is not None
    assert seg["ja_plain"] == "今日は雨です"
    assert seg["ruby_parts"] == [{"base": "今日", "rt": "きょう"}, {"base": "は雨です"}]


def test_bilingual_ruby_string_only() -> None:
    """With no `ja`, a ruby string is used as the sentence."""
    seg = parse_bilingual_block_body("ruby: 水(みず)を飲む\n")
    assert seg is not None
    assert seg["ja_plain"] == "水を飲む"
    assert seg["ruby_parts"][0] == {"base": "水", "rt": "みず"}


def test_bilingual_ruby_list_of_dicts_and_pairs() -> None:
    """A ruby list accepts `{base, rt}` dicts and `[base, rt]` pairs, applied sequentially; junk is skipped."""
    body = (
        "ja: 山と川と山\n"
        "ruby:\n"
        "  - {base: 山, rt: やま}\n"
        "  - [川, かわ]\n"
        "  - {base: 山, rt: さん}\n"
        "  - {base: 海, rt: うみ}\n"
        "  - 42\n"
        "  - [only-one]\n"
    )
    seg = parse_bilingual_block_body(body)
    assert seg is not None
    assert seg["ruby_parts"] == [
        {"base": "山", "rt": "やま"},
        {"base": "と"},
        {"base": "川", "rt": "かわ"},
        {"base": "と"},
        {"base": "山", "rt": "さん"},
    ]
    assert isinstance(seg["ruby"], list)


def test_bilingual_ruby_dict_annotates_every_occurrence_longest_first() -> None:
    """A ruby mapping annotates every occurrence, preferring the longest key at each position."""
    seg = parse_bilingual_block_body("ja: 日本語と日本\nruby:\n  日本: にほん\n  日本語: にほんご\n")
    assert seg is not None
    assert seg["ruby_parts"] == [
        {"base": "日本語", "rt": "にほんご"},
        {"base": "と"},
        {"base": "日本", "rt": "にほん"},
    ]
    assert seg["ruby"] == {"日本": "にほん", "日本語": "にほんご"}


def test_bilingual_ja_html_escapes_markup() -> None:
    """ja_html only contains <ruby>/<rt> markup; everything else, including readings, is escaped."""
    body = "ja: '<script>alert(1)</script>&{\"x\"|<b>}'\n"
    seg = parse_bilingual_block_body(body)
    assert seg is not None
    html = seg["ja_html"]
    assert "<script>" not in html and "<b>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;&amp;" in html
    assert "<ruby>&quot;x&quot;<rt>&lt;b&gt;</rt></ruby>" in html
    assert seg["ja_plain"].startswith("<script>")


def test_bilingual_voice_as_tag_bare_path_and_dict() -> None:
    """voice may be a `[voice: p]` tag, a bare path, or a `{ref|path, duration_ms}` mapping."""
    tag = parse_bilingual_block_body("ja: はい\nvoice: '[voice: /tmp/a.ogg]'\n")
    assert tag is not None and tag["voice"] == {"ref": "/tmp/a.ogg"}

    bare = parse_bilingual_block_body("ja: はい\nvoice: /tmp/b.ogg\n")
    assert bare is not None and bare["voice"] == {"ref": "/tmp/b.ogg"}

    as_ref = parse_bilingual_block_body("ja: はい\nvoice:\n  ref: '[voice: /tmp/c.ogg]'\n  duration_ms: 1234.7\n")
    assert as_ref is not None and as_ref["voice"] == {"ref": "/tmp/c.ogg", "duration_ms": 1234}

    as_path = parse_bilingual_block_body("ja: はい\nvoice:\n  path: /tmp/d.ogg\n  duration_ms: -5\n")
    assert as_path is not None and as_path["voice"] == {"ref": "/tmp/d.ogg"}


@pytest.mark.parametrize("voice_yaml", ["voice: ''", "voice: 3", "voice:\n  duration_ms: 10", "voice: [a, b]"])
def test_bilingual_invalid_voice_is_dropped(voice_yaml: str) -> None:
    """Unusable voice values are omitted rather than invalidating the block."""
    seg = parse_bilingual_block_body(f"ja: はい\n{voice_yaml}\n")
    assert seg is not None
    assert "voice" not in seg


def test_bilingual_tokens_filter() -> None:
    """Only token dicts with a string `surface` are kept; an all-invalid list drops the field."""
    body = (
        "ja: 本を読む\n"
        "tokens:\n"
        "  - {surface: 本, reading: ほん, pos: noun}\n"
        "  - {surface: 3}\n"
        "  - plain\n"
        "  - {reading: を}\n"
        "  - {surface: 読む}\n"
    )
    seg = parse_bilingual_block_body(body)
    assert seg is not None
    assert seg["tokens"] == [{"surface": "本", "reading": "ほん", "pos": "noun"}, {"surface": "読む"}]

    none_valid = parse_bilingual_block_body("ja: 本\ntokens:\n  - x\n  - {surface: 1}\n")
    assert none_valid is not None and "tokens" not in none_valid

    not_list = parse_bilingual_block_body("ja: 本\ntokens: {surface: 本}\n")
    assert not_list is not None and "tokens" not in not_list


def test_bilingual_speaker_and_name_fallback() -> None:
    """`speaker` is stripped; `name` is accepted as an alias; blank speakers are omitted."""
    seg = parse_bilingual_block_body("ja: はい\nspeaker: '  Narrator  '\n")
    assert seg is not None and seg["speaker"] == "Narrator"

    alias = parse_bilingual_block_body("ja: はい\nname: Guide\n")
    assert alias is not None and alias["speaker"] == "Guide"

    blank = parse_bilingual_block_body("ja: はい\nspeaker: '   '\n")
    assert blank is not None and "speaker" not in blank


@pytest.mark.parametrize(
    "body",
    [
        "ja: [unclosed\n",  # invalid YAML
        "- just\n- a list\n",  # not a mapping
        "zh: 只有中文\n",  # no ja and no ruby string
        "ja: '   '\n",  # blank ja
        "ja: 123\n",  # non-string ja
        "ja: ''\nruby: {日: ひ}\n",  # blank ja; a ruby mapping cannot stand in for the text
    ],
)
def test_bilingual_invalid_bodies_return_none(body: str) -> None:
    assert parse_bilingual_block_body(body) is None


def test_bilingual_segment_is_json_serializable() -> None:
    seg = parse_bilingual_block_body("ja: 今日(きょう)\nzh: 今天\nvoice: /tmp/v.ogg\ntokens:\n  - {surface: 今日}\n")
    assert seg is not None
    assert json.loads(json.dumps(seg, ensure_ascii=False)) == seg


# ---------------------------------------------------------------------------------------------------------------
# ```card
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["weather", "vocab.word", "note-1", "A_B", "x" * 64])
def test_card_kind_regex_accepts(kind: str) -> None:
    assert CARD_KIND_RE.match(kind)
    seg = parse_card_block_body(f"kind: '{kind}'\n")
    assert seg is not None and seg["kind"] == kind


@pytest.mark.parametrize("kind", ["", "has space", "slash/kind", "x" * 65, "<b>", "名前"])
def test_card_kind_regex_rejects(kind: str) -> None:
    assert not CARD_KIND_RE.match(kind)
    assert parse_card_block_body(f"kind: '{kind}'\ntitle: T\n") is None


def test_card_kind_is_stripped() -> None:
    seg = parse_card_block_body("kind: '  weather  '\n")
    assert seg is not None and seg["kind"] == "weather"


def test_card_data_passthrough_and_jsonable() -> None:
    """Every non-kind field passes through; YAML-only types (dates, sets) become JSON-safe."""
    body = (
        "kind: summary\n"
        "title: {en: Daily summary, zh: 每日小结}\n"
        "count: 3\n"
        "ratio: 0.5\n"
        "ok: true\n"
        "empty: null\n"
        "when: 2024-01-02\n"
        "tags: !!set {a: null}\n"
        "items:\n"
        "  - {label: one, at: 2024-03-04}\n"
        "  - two\n"
        "1: numeric-key\n"
    )
    seg = parse_card_block_body(body)
    assert seg is not None
    assert seg["type"] == "card"
    assert seg["kind"] == "summary"
    data = seg["data"]
    assert "kind" not in data
    assert data["title"] == {"en": "Daily summary", "zh": "每日小结"}
    assert data["count"] == 3 and data["ratio"] == 0.5 and data["ok"] is True and data["empty"] is None
    assert data["when"] == "2024-01-02"
    assert data["tags"] == ["a"]
    assert data["items"] == [{"label": "one", "at": "2024-03-04"}, "two"]
    assert data["1"] == "numeric-key"
    assert json.loads(json.dumps(seg, ensure_ascii=False)) == seg


@pytest.mark.parametrize(
    "body",
    [
        "kind: [unclosed\n",  # invalid YAML
        "title: no kind\n",  # missing kind
        "kind: 5\n",  # non-string kind
        "just a string\n",  # not a mapping
        "",  # empty
    ],
)
def test_card_invalid_bodies_return_none(body: str) -> None:
    assert parse_card_block_body(body) is None


def test_card_media_refs() -> None:
    """Only top-level string fields that are exactly one media tag are reported."""
    data = {
        "image": "[file: /tmp/pic.png]",
        "audio": "  [voice: /tmp/a.ogg ]  ",
        "caption": "see [file: /tmp/pic.png]",
        "nested": {"image": "[file: /tmp/x.png]"},
        "other": "[link: x]",
    }
    assert card_media_refs(data) == {
        "image": {"kind": "file", "ref": "/tmp/pic.png"},
        "audio": {"kind": "voice", "ref": "/tmp/a.ogg"},
    }


# ---------------------------------------------------------------------------------------------------------------
# parse_rich_block / render_*_as_text / flatten_rich_blocks
# ---------------------------------------------------------------------------------------------------------------


def test_parse_rich_block_dispatch() -> None:
    assert parse_rich_block("bilingual", "ja: はい\n")["type"] == "bilingual"
    assert parse_rich_block("card", "kind: k\n")["type"] == "card"
    assert parse_rich_block("ui", "kind: k\n") is None


def test_rich_block_re_requires_line_fences() -> None:
    """Fences must open and close on their own lines; other fence kinds are not matched."""
    text = "a\n  ```card\nkind: k\n  ```\n```python\nkind: k\n```\ninline ```card kind: k```"
    matches = list(RICH_BLOCK_RE.finditer(text))
    assert len(matches) == 1
    assert matches[0].group("kind") == "card"
    assert matches[0].group("body") == "kind: k\n"


def test_render_bilingual_as_text() -> None:
    seg = parse_bilingual_block_body("ja: 今日(きょう)\nzh: 今天\nspeaker: Host\nvoice: '[voice: /tmp/v.ogg]'\n")
    assert render_bilingual_as_text(seg) == "Host：今日\n今天\n[voice: /tmp/v.ogg]"

    minimal = parse_bilingual_block_body("ja: はい\n")
    assert render_bilingual_as_text(minimal) == "はい"
    assert render_rich_block_as_text(minimal) == "はい"


def test_render_card_as_text_title_points_and_media() -> None:
    seg = parse_card_block_body(
        "kind: place\n"
        "name: {ja: 公園, en: Park}\n"
        "area: North\n"
        "rating: 4.5\n"
        "open: true\n"
        "features: [quiet, {en: shady}, 7, {other: x}]\n"
        "glossary: [skip, me]\n"
        "nested: {deep: {x: 1}}\n"
        "photo: '[file: /tmp/park.png]'\n"
    )
    text = render_card_as_text(seg)
    assert text.splitlines() == [
        "**公園**",
        "- area: North",
        "- rating: 4.5",
        "- features: quiet、shady、7",
        "[file: /tmp/park.png]",
    ]
    assert render_rich_block_as_text(seg) == text


def test_render_card_title_key_priority_and_kind_fallback() -> None:
    """`title` beats `name`; with no usable title the kind is used."""
    seg = parse_card_block_body("kind: k\nname: N\ntitle: T\n")
    assert render_card_as_text(seg).splitlines() == ["**T**", "- name: N"]

    no_title = parse_card_block_body("kind: fallback-kind\nbody: text\n")
    assert render_card_as_text(no_title).splitlines()[0] == "**fallback-kind**"

    assert render_card_as_text({"type": "card"}) == "**card**"
    assert render_card_as_text({"type": "card", "kind": "k", "data": "not-a-dict"}) == "**k**"


def test_render_card_limits_lines_and_value_length() -> None:
    body = "kind: many\n" + "".join(f"k{i}: v{i}\n" for i in range(30))
    body += "long: " + "x" * (CARD_FALLBACK_VALUE_MAX_CHARS + 50) + "\n"
    lines = render_card_as_text(parse_card_block_body(body)).splitlines()
    assert len(lines) == 1 + CARD_FALLBACK_MAX_LINES
    assert lines[-1] == f"- k{CARD_FALLBACK_MAX_LINES - 1}: v{CARD_FALLBACK_MAX_LINES - 1}"

    long_only = parse_card_block_body("kind: k\nlong: " + "y" * (CARD_FALLBACK_VALUE_MAX_CHARS + 50) + "\n")
    value = render_card_as_text(long_only).splitlines()[1][len("- long: ") :]
    assert len(value) == CARD_FALLBACK_VALUE_MAX_CHARS
    assert value.endswith("…")


def test_render_rich_block_as_text_unknown_type() -> None:
    assert render_rich_block_as_text({"type": "question"}) == ""


def test_flatten_rich_blocks_replaces_valid_blocks() -> None:
    text = (
        "Before\n"
        + _block("bilingual", "ja: 雨(あめ)\nzh: 下雨")
        + "\nMiddle\n"
        + _block("card", "kind: k\ntitle: Card title\nnote: n1")
        + "\nAfter"
    )
    flat = flatten_rich_blocks(text)
    assert "```" not in flat
    assert flat == "Before\n雨\n下雨\nMiddle\n**Card title**\n- note: n1\nAfter"


def test_flatten_rich_blocks_leaves_invalid_and_other_blocks() -> None:
    bad_yaml = _block("card", "kind: [oops")
    no_kind = _block("card", "title: T")
    bad_bilingual = _block("bilingual", "zh: only")
    other = _block("python", "print(1)")
    text = "\n".join([bad_yaml, no_kind, bad_bilingual, other])
    assert flatten_rich_blocks(text) == text


@pytest.mark.parametrize("text", ["", "no fences here", "inline `code` only"])
def test_flatten_rich_blocks_passthrough(text: str) -> None:
    assert flatten_rich_blocks(text) == text


# ---------------------------------------------------------------------------------------------------------------
# parse_message_content
# ---------------------------------------------------------------------------------------------------------------

_MIXED = (
    "Intro\n"
    + _block("bilingual", "ja: 水(みず)\nzh: 水\nvoice: '[voice: /tmp/water.ogg]'")
    + "\n"
    + _block("card", "kind: info\ntitle: Facts\nimage: '[file: /tmp/fact.png]'")
    + "\nOutro [file: /tmp/outside.txt]"
)


def test_default_mode_never_leaks_rich_fences_and_keeps_media() -> None:
    segments = parse_message_content(_MIXED)
    texts = "".join(s["content"] for s in segments if s["type"] == "text")
    assert "```" not in texts
    assert "bilingual" not in texts and "kind:" not in texts
    assert "Intro" in texts and "水" in texts and "**Facts**" in texts and "Outro" in texts
    assert [s for s in segments if s["type"] in ("voice", "file")] == [
        {"type": "voice", "path": "/tmp/water.ogg"},
        {"type": "file", "path": "/tmp/fact.png"},
        {"type": "file", "path": "/tmp/outside.txt"},
    ]
    assert not any(s["type"] in ("bilingual", "card") for s in segments)


def test_default_mode_invalid_rich_block_stays_raw_text() -> None:
    content = "x\n" + _block("card", "title: no kind") + "\ny"
    assert parse_message_content(content) == [{"type": "text", "content": content}]


def test_rich_mode_returns_structured_segments_and_masks_inner_tags() -> None:
    segments = parse_message_content(_MIXED, rich_blocks=True)
    assert _types(segments) == ["text", "bilingual", "text", "card", "text", "file"]
    assert segments[0] == {"type": "text", "content": "Intro\n"}
    bilingual = segments[1]
    assert bilingual["ja_plain"] == "水" and bilingual["voice"] == {"ref": "/tmp/water.ogg"}
    card = segments[3]
    assert card["kind"] == "info"
    assert card["data"] == {"title": "Facts", "image": "[file: /tmp/fact.png]"}
    # Tags inside blocks are part of the block, not separate media segments.
    assert segments[-1] == {"type": "file", "path": "/tmp/outside.txt"}
    assert not any(s.get("path") in ("/tmp/water.ogg", "/tmp/fact.png") for s in segments)


def test_rich_mode_invalid_block_is_text() -> None:
    content = _block("card", "kind: 'bad kind'\nimage: '[file: /tmp/z.png]'")
    segments = parse_message_content(content, rich_blocks=True)
    assert "card" not in _types(segments)
    assert "```card" in "".join(s.get("content", "") for s in segments)


def test_rich_mode_without_rich_blocks_matches_default() -> None:
    content = "Hi [file: /tmp/a.png] <details><summary>S</summary>B</details> [voice: /tmp/b.ogg]"
    assert parse_message_content(content, rich_blocks=True) == parse_message_content(content)


_UI = "```ui\ntype: select\nprompt: Pick one\noptions:\n  - Alpha\n  - Beta [file: /tmp/in-ui.png]\n```"


@pytest.mark.parametrize("rich", [False, True])
def test_ui_block_unchanged_when_mixed_with_rich_blocks(rich: bool) -> None:
    content = _block("card", "kind: k\ntitle: T") + "\n" + _UI + "\n" + _block("bilingual", "ja: はい")
    segments = parse_message_content(content, rich_blocks=rich)
    questions = [s for s in segments if s["type"] == "question"]
    assert questions == [
        {
            "type": "question",
            "question": "Pick one",
            "choices": ["Alpha", "Beta [file: /tmp/in-ui.png]"],
            "multi": False,
            "max": 1,
        }
    ]
    # The tag inside the ```ui block is masked in both modes.
    assert not any(s["type"] == "file" for s in segments)
    if rich:
        assert _types(segments) == ["card", "text", "question", "text", "bilingual"]
    else:
        assert _types(segments) == ["text", "question", "text"]
        assert segments[0]["content"] == "**T**\n"
        assert segments[2]["content"] == "\nはい"

    # The ui segment is identical to what the same block yields on its own.
    assert questions == [s for s in parse_message_content(_UI) if s["type"] == "question"]


def test_rich_block_inside_details_body_stays_in_details() -> None:
    """A rich block inside <details> is part of the details body in rich mode (first span wins)."""
    inner = _block("card", "kind: k\ntitle: T")
    content = f"<details><summary>More</summary>\n{inner}\n</details>"
    segments = parse_message_content(content, rich_blocks=True)
    assert _types(segments) == ["details"]
    assert "```card" in segments[0]["body"]


# ---------------------------------------------------------------------------------------------------------------
# Unquoted media tags (YAML reads `[file: p]` as a list of one mapping) are coerced back to tags
# ---------------------------------------------------------------------------------------------------------------


def test_card_unquoted_media_tag_is_kept() -> None:
    seg = parse_card_block_body("kind: k\nimage: [file: /tmp/x.png]\n")
    assert seg is not None
    assert card_media_refs(seg["data"]) == {"image": {"kind": "file", "ref": "/tmp/x.png"}}


def test_bilingual_unquoted_voice_tag_is_kept() -> None:
    seg = parse_bilingual_block_body("ja: はい\nvoice: [voice: /tmp/a.ogg]\n")
    assert seg is not None and seg.get("voice") == {"ref": "/tmp/a.ogg"}

"""Unit tests for the chatbot message parser utility."""

from kesoku.gateway.chatbot.base import parse_message_content


def test_parse_no_file_blocks() -> None:
    """Verify that a standard string without file blocks parses as a single text segment."""
    content = "Hello world, this is a standard message."
    segments = parse_message_content(content)
    assert segments == [{"type": "text", "content": "Hello world, this is a standard message."}]


def test_parse_single_file_block_middle() -> None:
    """Verify that a single file block in the middle splits into text, file, and text segments."""
    content = "Hello [file: /path/to/image.png] welcome!"
    segments = parse_message_content(content)
    assert segments == [
        {"type": "text", "content": "Hello "},
        {"type": "file", "path": "/path/to/image.png"},
        {"type": "text", "content": " welcome!"},
    ]


def test_parse_single_file_block_start() -> None:
    """Verify a file block at the very beginning of the message works correctly."""
    content = "[file: /path/to/audio.mp3] has finished rendering."
    segments = parse_message_content(content)
    assert segments == [
        {"type": "file", "path": "/path/to/audio.mp3"},
        {"type": "text", "content": " has finished rendering."},
    ]


def test_parse_single_file_block_end() -> None:
    """Verify a file block at the very end of the message works correctly."""
    content = "Here is the generated report: [file: /path/report.pdf]"
    segments = parse_message_content(content)
    assert segments == [
        {"type": "text", "content": "Here is the generated report: "},
        {"type": "file", "path": "/path/report.pdf"},
    ]


def test_parse_multiple_file_blocks() -> None:
    """Verify that multiple file blocks are processed in the correct order."""
    content = "A: [file: /a.png] B: [file: /b.jpg] C"
    segments = parse_message_content(content)
    assert segments == [
        {"type": "text", "content": "A: "},
        {"type": "file", "path": "/a.png"},
        {"type": "text", "content": " B: "},
        {"type": "file", "path": "/b.jpg"},
        {"type": "text", "content": " C"},
    ]


def test_parse_whitespace_and_formatting() -> None:
    """Verify that file paths are correctly trimmed of leading/trailing whitespace inside the block."""
    content = "Check this out: [file:    /path/to/some_file.zip     ]"
    segments = parse_message_content(content)
    assert segments == [
        {"type": "text", "content": "Check this out: "},
        {"type": "file", "path": "/path/to/some_file.zip"},
    ]


def test_parse_voice_blocks() -> None:
    """Verify that a voice block is correctly parsed and recognized."""
    content = "Listen: [voice: /path/to/audio.ogg] thanks!"
    segments = parse_message_content(content)
    assert segments == [
        {"type": "text", "content": "Listen: "},
        {"type": "voice", "path": "/path/to/audio.ogg"},
        {"type": "text", "content": " thanks!"},
    ]


def test_parse_mixed_file_and_voice_blocks() -> None:
    """Verify that mixed file and voice blocks are parsed in the correct order."""
    content = "Here is a doc [file: /doc.txt] and voice [voice: /speech.ogg] end."
    segments = parse_message_content(content)
    assert segments == [
        {"type": "text", "content": "Here is a doc "},
        {"type": "file", "path": "/doc.txt"},
        {"type": "text", "content": " and voice "},
        {"type": "voice", "path": "/speech.ogg"},
        {"type": "text", "content": " end."},
    ]


def test_parse_voice_whitespace_and_formatting() -> None:
    """Verify that voice paths are correctly trimmed of leading/trailing whitespace."""
    content = "Listen: [voice:    /path/voice.wav     ]"
    segments = parse_message_content(content)
    assert segments == [
        {"type": "text", "content": "Listen: "},
        {"type": "voice", "path": "/path/voice.wav"},
    ]


def test_parse_single_question_block() -> None:
    """Verify that a question block is correctly parsed and recognized using || syntax."""
    content = "Please choose:\n[question: What is your favorite color? || Red | Blue | Green]\nThank you!"
    segments = parse_message_content(content)
    assert segments == [
        {"type": "text", "content": "Please choose:\n"},
        {
            "type": "question",
            "question": "What is your favorite color?",
            "choices": ["Red", "Blue", "Green"],
            "multi": False,
            "max": 1,
        },
        {"type": "text", "content": "\nThank you!"},
    ]


def test_parse_question_block_spaces() -> None:
    """Verify that spaces in question and choices are stripped correctly using || syntax."""
    content = "[question:   Are you ready?    ||   Yes  |  No   ]"
    segments = parse_message_content(content)
    assert segments == [
        {
            "type": "question",
            "question": "Are you ready?",
            "choices": ["Yes", "No"],
            "multi": False,
            "max": 1,
        }
    ]


def test_parse_mixed_question_file_voice() -> None:
    """Verify mixed parsing of question (|| syntax), file, and voice blocks."""
    content = "[file: /a.png] and [voice: /b.wav] next [question: Continue? || Yes | No]"
    segments = parse_message_content(content)
    assert segments == [
        {"type": "file", "path": "/a.png"},
        {"type": "text", "content": " and "},
        {"type": "voice", "path": "/b.wav"},
        {"type": "text", "content": " next "},
        {
            "type": "question",
            "question": "Continue?",
            "choices": ["Yes", "No"],
            "multi": False,
            "max": 1,
        },
    ]


def test_parse_question_block_fallback() -> None:
    """Verify that standard single bar split fallback works correctly for backwards compatibility."""
    content = "[question: Legacy Question | Opt A | Opt B]"
    segments = parse_message_content(content)
    assert segments == [
        {
            "type": "question",
            "question": "Legacy Question",
            "choices": ["Opt A", "Opt B"],
            "multi": False,
            "max": 1,
        }
    ]


# ---------------------------------------------------------------------------
# ```ui blocks and <details>
# ---------------------------------------------------------------------------

UI_SELECT = """Pick one:
```ui
type: select
prompt: 下一步怎么做？
options:
  - 正面强攻 [风险高]
  - 绕到侧翼
  - 先问问意见
```
Done."""


def test_parse_ui_select_block() -> None:
    """A ```ui select block becomes a question segment; bracketed option text stays intact."""
    segments = parse_message_content(UI_SELECT)
    assert segments == [
        {"type": "text", "content": "Pick one:\n"},
        {
            "type": "question",
            "question": "下一步怎么做？",
            "choices": ["正面强攻 [风险高]", "绕到侧翼", "先问问意见"],
            "multi": False,
            "max": 1,
        },
        {"type": "text", "content": "\nDone."},
    ]


def test_parse_ui_multiselect_with_max() -> None:
    """Multiselect carries multi=True and the given max."""
    content = "```ui\ntype: multiselect\nprompt: Pick\noptions: [A, B, C]\nmax: 2\n```"
    assert parse_message_content(content) == [
        {"type": "question", "question": "Pick", "choices": ["A", "B", "C"], "multi": True, "max": 2}
    ]


def test_parse_ui_multiselect_default_and_clamped_max() -> None:
    """max defaults to len(options) and is clamped to [1, len(options)]."""
    default = parse_message_content("```ui\ntype: multiselect\nprompt: P\noptions:\n  - a\n  - b\n```")
    assert default[0]["max"] == 2
    clamped = parse_message_content("```ui\ntype: multiselect\nprompt: P\noptions: [a, b]\nmax: 9\n```")
    assert clamped[0]["max"] == 2


def test_parse_ui_option_starting_with_bracket_and_colon_prompt() -> None:
    """Options starting with '[' and prompts containing ': ' fall back to verbatim line parsing."""
    content = "```ui\ntype: select\nprompt: Note: choose\noptions:\n  - [risky] charge\n  - wait: and see\n```"
    seg = parse_message_content(content)[0]
    assert seg["type"] == "question"
    assert seg["question"] == "Note: choose"
    assert seg["choices"] == ["[risky] charge", "wait: and see"]


def test_parse_only_first_ui_block_is_interactive() -> None:
    """A second ```ui block is left as plain text."""
    second = "```ui\ntype: select\nprompt: Second\noptions: [x, y]\n```"
    content = "```ui\ntype: select\nprompt: First\noptions: [a, b]\n```\nmid\n" + second
    segments = parse_message_content(content)
    assert segments[0]["type"] == "question" and segments[0]["question"] == "First"
    assert segments[1] == {"type": "text", "content": "\nmid\n" + second}
    assert len(segments) == 2


def test_parse_invalid_ui_block_stays_text() -> None:
    """Invalid YAML / missing fields leave the block untouched as text (no crash)."""
    for body in [
        "type: select\nprompt: P\n",  # no options
        "type: radio\nprompt: P\noptions: [a]\n",  # bad type
        "type: select\noptions: [a]\n",  # no prompt
        "type: multiselect\nprompt: P\noptions: [a]\nmax: lots\n",  # bad max
        ": : :\n- [\n",  # garbage
        "",
    ]:
        content = f"before\n```ui\n{body}```\nafter"
        assert parse_message_content(content) == [{"type": "text", "content": content}], body


def test_parse_ui_block_masks_tags_inside() -> None:
    """A [file:] tag inside a non-interactive ```ui block is not extracted."""
    content = "```ui\ntype: select\nprompt: P\n```\n```ui\ntype: select\nprompt: Q [file: /x.png]\noptions: [a]\n```"
    segments = parse_message_content(content)
    assert all(s["type"] == "text" for s in segments)


def test_parse_details_block() -> None:
    """<details><summary> becomes a details segment."""
    content = "Intro\n<details><summary>Explanation</summary>\nLong body\nline 2\n</details>\nOutro"
    assert parse_message_content(content) == [
        {"type": "text", "content": "Intro\n"},
        {"type": "details", "title": "Explanation", "body": "Long body\nline 2"},
        {"type": "text", "content": "\nOutro"},
    ]


def test_parse_details_with_attributes_and_multiple() -> None:
    """`<details open>` is accepted and several details blocks are all parsed."""
    content = "<details open><summary>A</summary>a</details><DETAILS><summary>B</summary>b</DETAILS>"
    segments = parse_message_content(content)
    assert segments == [
        {"type": "details", "title": "A", "body": "a"},
        {"type": "details", "title": "B", "body": "b"},
    ]


def test_parse_details_body_keeps_inner_tags_as_text() -> None:
    """Tags inside a details body are part of the body."""
    content = "<details><summary>T</summary>see [file: /a.png]</details>"
    assert parse_message_content(content) == [{"type": "details", "title": "T", "body": "see [file: /a.png]"}]


def test_parse_mixed_ui_details_and_file() -> None:
    """All syntaxes combine in document order."""
    content = (
        "[file: /a.png]\n<details><summary>S</summary>B</details>\n"
        "```ui\ntype: select\nprompt: P\noptions: [x]\n```"
    )
    types = [s["type"] for s in parse_message_content(content)]
    assert types == ["file", "text", "details", "text", "question"]

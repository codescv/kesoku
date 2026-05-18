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

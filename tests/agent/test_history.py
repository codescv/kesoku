"""Unit tests for history preparation (truncation, determinism, historical question-option stripping)."""

from kesoku.agent.history import prepare_history_for_llm, strip_question_options
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message


def test_truncate_tool_call_arguments_only_for_file_tools():
    """Long string args of update_session_notes/update_file are truncated; other tools are untouched."""
    from kesoku.agent.history import MAX_HISTORICAL_TOOL_ARG_CHARS, truncate_tool_call_arguments

    long = "n" * 2000
    args = {"content": long}
    out = truncate_tool_call_arguments("update_session_notes", args)
    assert out is not args
    assert out["content"].startswith("n" * MAX_HISTORICAL_TOOL_ARG_CHARS)
    assert "2000 chars total" in out["content"]
    assert args["content"] == long  # original not mutated
    img_args = {"command": "uv run generate_image.py --prompt " + long}
    assert truncate_tool_call_arguments("run_shell_command", img_args) is img_args
    short = {"content": "short"}
    assert truncate_tool_call_arguments("update_file", short) is short


def test_prepare_history_truncates_completed_notes_tool_call_args():
    """Completed turns get truncated notes arguments; the active turn keeps them in full."""
    from kesoku.agent.history import prepare_history_for_llm
    from kesoku.constants import MessageRole, MessageStatus, MessageType
    from kesoku.db import Message

    def call(i: int) -> list[Message]:
        return [
            Message(session_id="s", chatbot_id="cb", channel_id="ch", sender="u", role=MessageRole.USER,
                    type=MessageType.TEXT, content=f"turn {i}", status=MessageStatus.RESPONDED, timestamp=i * 10),
            Message(session_id="s", chatbot_id="cb", channel_id="ch", sender="update_session_notes",
                    role=MessageRole.TOOL, type=MessageType.TOOL_CALL, content="Calling tool",
                    status=MessageStatus.RESPONDED, timestamp=i * 10 + 1,
                    metadata={"tool_name": "update_session_notes", "tool_arguments": {"content": "n" * 2000}}),
        ]

    prepared = prepare_history_for_llm(call(1) + call(2))
    calls = [m for m in prepared if m.type == MessageType.TOOL_CALL]
    assert len(calls[0].metadata["tool_arguments"]["content"]) < 400
    assert len(calls[1].metadata["tool_arguments"]["content"]) == 2000


def test_prepare_history_truncation_is_deterministic(tmp_path):
    """Truncated historical tool output must be byte-identical across calls (required for implicit caching)."""
    import time as _time

    from kesoku.agent.history import prepare_history_for_llm
    from kesoku.constants import MessageRole, MessageStatus, MessageType
    from kesoku.db import Message

    def turn(i: int) -> list[Message]:
        return [
            Message(id=f"u{i}", session_id="s", chatbot_id="cb", channel_id="ch", sender="u", role=MessageRole.USER,
                    type=MessageType.TEXT, content=f"q{i}", status=MessageStatus.RESPONDED, timestamp=i * 10),
            Message(id=f"r{i}", session_id="s", chatbot_id="cb", channel_id="ch", sender="chat_search",
                    role=MessageRole.TOOL, type=MessageType.TOOL_RESULT, content="x" * 2000,
                    status=MessageStatus.RESPONDED, timestamp=i * 10 + 1,
                    metadata={"tool_name": "chat_search", "tool_result": "x" * 2000}),
        ]

    history = turn(1) + turn(2)
    first = prepare_history_for_llm(history, staging_dir=str(tmp_path))
    _time.sleep(1.1)
    second = prepare_history_for_llm(history, staging_dir=str(tmp_path))
    assert [m.content for m in first] == [m.content for m in second]
    assert "hist_tool_chat_search_r1.txt" in first[1].content


# ---------------------------------------------------------------------------
# Historical [question: ...] option stripping (generic engine syntax from prompt.py)
# ---------------------------------------------------------------------------

Q_BLOCK = "[question: 下一步怎么做？ || 像香槟软木塞一样弹射出去 | 冷静地递上一杯热可可 | 拉着队友逃跑]"
STRIPPED = "[question: ...]"


def _hmsg(role: MessageRole, content: str, ts: float, msg_id: str, msg_type: MessageType = MessageType.TEXT) -> Message:
    """Build a minimal history Message."""
    return Message(
        id=msg_id,
        session_id="s",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1" if role == MessageRole.USER else "Kesoku",
        role=role,
        type=msg_type,
        content=content,
        status=MessageStatus.RESPONDED,
        timestamp=ts,
    )


def _question_history(num_completed: int, with_active: bool = True) -> list[Message]:
    """Build `num_completed` user/assistant turns whose assistant reply ends with a [question:] block."""
    history: list[Message] = []
    for i in range(num_completed):
        history.append(_hmsg(MessageRole.USER, f"用户第{i}轮", i * 100.0, f"u{i}"))
        block = Q_BLOCK.replace("下一步", f"第{i}步")
        history.append(_hmsg(MessageRole.ASSISTANT, f"第{i}轮的正文。\n{block}", i * 100.0 + 10, f"a{i}"))
    if with_active:
        history.append(_hmsg(MessageRole.USER, "冷静地递上一杯热可可", num_completed * 100.0, "u_active"))
    return history


def _assistant_texts(prepared: list[Message]) -> list[str]:
    """Contents of assistant messages, in order."""
    return [m.content for m in prepared if m.role == MessageRole.ASSISTANT]


def test_strip_question_options_basic() -> None:
    """Stem and options collapse into the placeholder; surrounding text is preserved."""
    out = strip_question_options(f"正文。\n{Q_BLOCK}")
    assert out == f"正文。\n{STRIPPED}"
    assert "||" not in out and "香槟" not in out


def test_strip_question_options_multiple_idempotent_and_plain() -> None:
    """All blocks are stripped, stripping is idempotent, and text without blocks is untouched."""
    out = strip_question_options(f"A {Q_BLOCK} B [question: 选哪个？ || x | y] C")
    assert out == f"A {STRIPPED} B {STRIPPED} C"
    assert strip_question_options(out) == out
    plain = "普通回复 [file: a.png] [voice: b.mp3]"
    assert strip_question_options(plain) == plain
    assert strip_question_options("") == ""


def test_strip_question_options_keeps_attachment_markup() -> None:
    """[file:]/[voice:] markup adjacent to a question block survives."""
    out = strip_question_options(f"[file: images/x.png]\n台词\n{Q_BLOCK}\n[voice: audio/a.mp3]")
    assert "[file: images/x.png]" in out and "[voice: audio/a.mp3]" in out and STRIPPED in out


def test_prepare_history_strips_turns_older_than_last_completed() -> None:
    """Turns with idx < num_turns - 2 are stripped; the last completed turn and the active turn are intact."""
    prepared = prepare_history_for_llm(_question_history(4), strip_old_questions=True)
    texts = _assistant_texts(prepared)
    assert len(texts) == 4
    for t in texts[:3]:
        assert STRIPPED in t and "||" not in t and "香槟" not in t
        assert "轮的正文" in t
    assert "第3步" in texts[3] and "||" in texts[3] and "像香槟软木塞一样弹射出去" in texts[3]


def test_prepare_history_without_active_turn() -> None:
    """Without an active user turn the final turn is 'latest' and the one before it the last completed."""
    texts = _assistant_texts(prepare_history_for_llm(_question_history(3, with_active=False), strip_old_questions=True))
    assert STRIPPED in texts[0] and "||" not in texts[0]
    assert "||" in texts[1] and "||" in texts[2]


def test_prepare_history_short_history_untouched() -> None:
    """One completed turn plus the active turn: nothing is stripped."""
    texts = _assistant_texts(prepare_history_for_llm(_question_history(1), strip_old_questions=True))
    assert "||" in texts[0]


def test_prepare_history_question_flag_off() -> None:
    """strip_old_questions=False keeps every option."""
    texts = _assistant_texts(prepare_history_for_llm(_question_history(4), strip_old_questions=False))
    assert all("||" in t for t in texts)


def test_prepare_history_question_strip_does_not_mutate_input() -> None:
    """Originals are untouched and user messages keep their text."""
    history = _question_history(4)
    before = [m.content for m in history]
    prepared = prepare_history_for_llm(history, strip_old_questions=True)
    assert [m.content for m in history] == before
    users = [m.content for m in prepared if m.role == MessageRole.USER]
    assert any("用户第0轮" in u for u in users)
    assert any("冷静地递上一杯热可可" in u for u in users)


def test_prepare_history_question_strip_prefix_stable() -> None:
    """Already-stripped turns render byte-identically as new turns arrive (implicit prefix caching)."""
    p1 = _assistant_texts(prepare_history_for_llm(_question_history(4), strip_old_questions=True))
    p2 = _assistant_texts(prepare_history_for_llm(_question_history(5), strip_old_questions=True))
    assert p1[:3] == p2[:3]


def test_prepare_history_question_strip_skips_tool_messages() -> None:
    """Tool results are not rewritten by the question stripper (existing truncation logic owns them)."""
    history = _question_history(3)
    history.insert(
        1,
        Message(
            id="t0", session_id="s", chatbot_id="cli", channel_id="ch1", sender="chat_search",
            role=MessageRole.TOOL, type=MessageType.TOOL_RESULT, content=f"Tool returned: {Q_BLOCK}",
            status=MessageStatus.RESPONDED, timestamp=5.0,
            metadata={"tool_name": "chat_search", "tool_result": Q_BLOCK},
        ),
    )
    prepared = prepare_history_for_llm(history, strip_old_questions=True)
    tool_msgs = [m for m in prepared if m.role == MessageRole.TOOL]
    assert tool_msgs and "||" in tool_msgs[0].content


def test_prepare_history_truncates_completed_update_file_args() -> None:
    """Completed-turn update_file calls (notes edits) get old/new content truncated; the active turn is kept."""

    def call(i: int) -> list[Message]:
        return [
            Message(session_id="s", chatbot_id="cb", channel_id="ch", sender="u", role=MessageRole.USER,
                    type=MessageType.TEXT, content=f"turn {i}", status=MessageStatus.RESPONDED, timestamp=i * 10),
            Message(session_id="s", chatbot_id="cb", channel_id="ch", sender="update_file",
                    role=MessageRole.TOOL, type=MessageType.TOOL_CALL, content="Calling tool",
                    status=MessageStatus.RESPONDED, timestamp=i * 10 + 1,
                    metadata={"tool_name": "update_file", "tool_arguments": {
                        "file_name": "/tmp/s/notes.md", "old_content": "o" * 2000, "new_content": "n" * 2000}}),
        ]

    prepared = prepare_history_for_llm(call(1) + call(2))
    calls = [m for m in prepared if m.type == MessageType.TOOL_CALL]
    old_args, new_args = calls[0].metadata["tool_arguments"], calls[1].metadata["tool_arguments"]
    assert len(old_args["old_content"]) < 400 and len(old_args["new_content"]) < 400
    assert old_args["file_name"] == "/tmp/s/notes.md"
    assert len(new_args["old_content"]) == 2000 and len(new_args["new_content"]) == 2000

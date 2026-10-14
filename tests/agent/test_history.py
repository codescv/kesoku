

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

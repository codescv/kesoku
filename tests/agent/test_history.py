

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

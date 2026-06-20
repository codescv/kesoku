"""Unit tests for the LcmHtmlReporter utility."""

import os

from kesoku.db import Message, Session, SummaryNode
from kesoku.gateway.chatbot.lcm_reporter import LcmHtmlReporter


def test_render_to_temp_file() -> None:
    """Test that LcmHtmlReporter renders the context successfully into an HTML file."""
    session = Session(id="session_abc123", title="Test Session")

    # Summary Nodes
    node1 = SummaryNode(
        id="node-1-uuid",
        session_id="session_abc123",
        level=0,
        summary="Summary of node 1",
        start_timestamp=1000.0,
        end_timestamp=1010.0,
        token_count=100,
        source_token_count=300,
        parent_id=None,
    )

    node2 = SummaryNode(
        id="node-2-uuid",
        session_id="session_abc123",
        level=1,
        summary="Summary of node 2",
        start_timestamp=1000.0,
        end_timestamp=1020.0,
        token_count=150,
        source_token_count=450,
        parent_id="parent-uuid-some-other",  # Parented/consolidated (inactive)
    )

    all_summaries = [node1, node2]
    root_summaries = [node1]

    protected_head = [
        Message(
            session_id="session_abc123",
            chatbot_id="cli",
            channel_id="cli_channel",
            sender="user",
            role="user",
            content="Hello",
        )
    ]

    buffer = [
        Message(
            session_id="session_abc123",
            chatbot_id="cli",
            channel_id="cli_channel",
            sender="user",
            role="user",
            content="Backlog item",
        )
    ]

    protected_tail = [
        Message(
            session_id="session_abc123",
            chatbot_id="cli",
            channel_id="cli_channel",
            sender="assistant",
            role="assistant",
            content="Hi there!",
        ),
        Message(
            session_id="session_abc123",
            chatbot_id="cli",
            channel_id="cli_channel",
            sender="run_shell_command",
            role="tool",
            content="Command stdout results",
            metadata={"tool_name": "run_shell_command"},
        ),
    ]

    sys_msg = "You are a helpful assistant."

    temp_file_path = LcmHtmlReporter.render_to_temp_file(
        session=session,
        root_summaries=root_summaries,
        all_summaries=all_summaries,
        protected_head=protected_head,
        buffer=buffer,
        protected_tail=protected_tail,
        sys_msg=sys_msg,
        last_metrics={"context_tokens": 78000, "cached_tokens": 138000},
    )

    try:
        assert os.path.exists(temp_file_path)
        assert temp_file_path.endswith("_lcm_context.html")

        with open(temp_file_path, encoding="utf-8") as f:
            content = f.read()

        assert "LCM Active Context" in content
        assert "session_abc123" in content
        assert "Summary of node 1" in content
        assert "Summary of node 2" in content
        assert "Parented / Consolidated" in content  # Inactive node suffix
        assert "Command stdout results" in content
        assert "run_shell_command" in content
        assert "You are a helpful assistant." in content
        assert "Backlog item" in content
        assert "Active Buffer" in content
        assert "Actual LLM Context (Last Turn)" in content
        assert "78K active + 138K cached" in content
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

"""Unit tests for the LcmHtmlReporter utility."""

import os
from unittest.mock import MagicMock

from openlcm.core.dag import SummaryNode

from kesoku.db import Session
from kesoku.gateway.chatbot.lcm_reporter import LcmHtmlReporter


def test_render_to_temp_file() -> None:
    """Test that LcmHtmlReporter renders the context successfully into an HTML file."""
    session = Session(id="session_abc123", title="Test Session")

    # Mock DAG Nodes
    node1 = MagicMock(spec=SummaryNode)
    node1.node_id = 1
    node1.depth = 0
    node1.summary = "Summary of node 1"
    node1.token_count = 100
    node1.source_token_count = 300
    node1.created_at = 1000.0
    node1.expand_hint = None

    node2 = MagicMock(spec=SummaryNode)
    node2.node_id = 2
    node2.depth = 1
    node2.summary = "Summary of node 2"
    node2.token_count = 150
    node2.source_token_count = 450
    node2.created_at = 1001.0
    node2.expand_hint = "lcm_expand(node_id=2)"

    all_nodes = [node1, node2]
    active_node_ids = {1}

    fresh_msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "tool", "name": "run_shell_command", "content": "Command stdout results"},
    ]
    sys_msg = "You are a helpful assistant."
    assembled_context = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    backlog_msgs = [
        {"role": "user", "content": "Backlog item"},
    ]

    temp_file_path = LcmHtmlReporter.render_to_temp_file(
        session=session,
        all_nodes=all_nodes,
        active_node_ids=active_node_ids,
        fresh_msgs=fresh_msgs,
        backlog_msgs=backlog_msgs,
        sys_msg=sys_msg,
        assembled_context=assembled_context,
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
        assert "Condensed" in content  # Inactive node suffix
        assert "Command stdout results" in content
        assert "run_shell_command" in content
        assert "You are a helpful assistant." in content
        assert "Backlog item" in content
        assert "Uncompacted Backlog" in content
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

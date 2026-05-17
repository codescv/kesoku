"""Unit tests for Kesoku agent tools and skill registry."""

from unittest.mock import MagicMock
import pytest

from kesoku.agent.tools import ToolContext, WebSearchTool, web_search


def test_web_search_tool_success() -> None:
    """Test WebSearchTool successfully executes search and formats grounding sources."""
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.text = "The weather in Tokyo is sunny."

    mock_candidate = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.web.uri = "https://weather.com/tokyo"
    mock_chunk1.web.title = "Tokyo Weather"

    # Duplicate URI to test deduplication logic
    mock_chunk2 = MagicMock()
    mock_chunk2.web.uri = "https://weather.com/tokyo"
    mock_chunk2.web.title = "Tokyo Weather Update"

    mock_candidate.grounding_metadata.grounding_chunks = [mock_chunk1, mock_chunk2]
    mock_res.candidates = [mock_candidate]

    mock_client.models.generate_content.return_value = mock_res

    # Explicitly inject mock client
    tool = WebSearchTool(client=mock_client)
    ctx = ToolContext(session_id="s1", session_workspace="workspace1")
    result = tool.web_search("What is the weather in Tokyo?", ctx)

    assert "The weather in Tokyo is sunny." in result
    assert "Sources:" in result
    assert "- Tokyo Weather: https://weather.com/tokyo" in result
    # Verify that duplicate sources are successfully deduplicated
    assert result.count("https://weather.com/tokyo") == 1


def test_web_search_tool_api_failure() -> None:
    """Test WebSearchTool handles API call exceptions gracefully."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API Quota Exceeded")

    tool = WebSearchTool(client=mock_client)
    result = tool.web_search("Test query", None)

    assert "Web search failed: API Quota Exceeded" in result

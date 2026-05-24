"""Unit tests for Kesoku LLM module and Anthropic Claude support."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kesoku.agent.llm import (
    ClaudeLLM,
    GeminiLLM,
    MockLLM,
    function_to_anthropic_tool,
    get_llm,
)
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    TYPE_TEXT,
    TYPE_THOUGHT,
    TYPE_TOOL_CALL,
    TYPE_TOOL_RESULT,
)
from kesoku.db import Message


def sample_tool_func(query: str, max_results: int = 10, context: Any = None) -> str:
    """Search for a query.

    Args:
        query: The search query.
        max_results: Maximum results to return.
        context: Context info.

    Returns:
        A string result.
    """
    return f"results for {query}"


def test_function_to_anthropic_tool() -> None:
    """Verify function_to_anthropic_tool correctly generates the schema for Anthropic."""
    schema = function_to_anthropic_tool(sample_tool_func)

    assert schema["name"] == "sample_tool_func"
    assert schema["description"] == "Search for a query."
    assert "context" not in schema["input_schema"]["properties"]
    assert "query" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["properties"]["query"]["type"] == "string"
    assert schema["input_schema"]["properties"]["max_results"]["type"] == "integer"
    assert "query" in schema["input_schema"]["required"]
    assert "max_results" not in schema["input_schema"]["required"]


def test_get_llm_providers() -> None:
    """Verify get_llm retrieves the correct LLM provider instance."""
    with patch("kesoku.agent.llm.GeminiLLM", return_value=MagicMock(spec=GeminiLLM)):
        gemini_llm = get_llm("gemini")
        assert gemini_llm is not None

    with patch("kesoku.agent.llm.ClaudeLLM", return_value=MagicMock(spec=ClaudeLLM)):
        claude_llm = get_llm("claude")
        assert claude_llm is not None

    mock_llm = get_llm("mock")
    assert isinstance(mock_llm, MockLLM)

    with pytest.raises(ValueError) as exc_info:
        get_llm("invalid_provider")
    assert "Unsupported LLM provider" in str(exc_info.value)


@pytest.mark.asyncio
async def test_claude_llm_generate_history_conversion() -> None:
    """Verify ClaudeLLM converts conversational history to Anthropic format correctly."""
    # Setup fake database messages for history
    msg_user = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="User",
        role=ROLE_USER,
        type=TYPE_TEXT,
        content="Hello! Do a calculation.",
    )

    msg_thought = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=ROLE_ASSISTANT,
        type=TYPE_THOUGHT,
        content="Thinking...",
    )

    msg_tool_call = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=ROLE_TOOL,
        type=TYPE_TOOL_CALL,
        content="Calling tool calculator",
        metadata={"tool_name": "calculator", "tool_arguments": {"expr": "2+2"}},
    )

    msg_tool_res = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="calculator",
        role=ROLE_TOOL,
        type=TYPE_TOOL_RESULT,
        content="Tool returned 4",
        metadata={"tool_name": "calculator", "tool_result": "4"},
        parent_id=msg_tool_call.id,
    )

    # Mock Anthropic Vertex client calls
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.content = [MagicMock(type="text", text="The answer is 4.")]
    mock_res.usage.input_tokens = 10
    mock_res.usage.output_tokens = 5
    mock_client.messages.create.return_value = mock_res

    with patch("anthropic.AnthropicVertex", return_value=mock_client):
        claude = ClaudeLLM()
        response = await claude.generate(
            prompt="What is 2+2?",
            system_prompt="You are a helpful math assistant.",
            history=[msg_user, msg_thought, msg_tool_call, msg_tool_res],
        )

        assert response.content == "The answer is 4."
        assert response.prompt_tokens == 10
        assert response.candidates_tokens == 5
        assert response.total_tokens == 15

        # Verify the messages argument formatted for Anthropic
        mock_client.messages.create.assert_called_once()
        called_kwargs = mock_client.messages.create.call_args[1]
        assert called_kwargs["system"] == "You are a helpful math assistant."

        messages = called_kwargs["messages"]
        # Expect alternating messages (user -> assistant -> user)
        assert len(messages) >= 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        # Verify tool use was represented correctly in assistant role
        tool_use_block = next(
            b for b in messages[1]["content"] if b["type"] == "tool_use"
        )
        assert tool_use_block["name"] == "calculator"
        assert tool_use_block["input"] == {"expr": "2+2"}
        assert tool_use_block["id"] == f"toolu_{msg_tool_call.id}"

        # Verify tool result was represented correctly in user role
        tool_res_block = next(
            b for m in messages for b in m["content"] if b["type"] == "tool_result"
        )
        assert tool_res_block["content"] == "4"
        assert tool_res_block["tool_use_id"] == f"toolu_{msg_tool_call.id}"


@pytest.mark.asyncio
async def test_gemini_llm_with_attachments() -> None:
    """Verify GeminiLLM extracts and formats attachments correctly using types.Part.from_bytes."""
    msg = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="User",
        role=ROLE_USER,
        type=TYPE_TEXT,
        content="Look at this!",
        metadata={
            "attachments": [
                {
                    "path": "/tmp/nonexistent_test_image.png",
                    "mime_type": "image/png",
                    "filename": "test_image.png",
                }
            ]
        },
    )

    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.parts = [MagicMock(text="Beautiful photo!", thought=False, function_call=None)]
    mock_res.usage_metadata = MagicMock(
        prompt_token_count=10, candidates_token_count=5, total_token_count=15
    )
    mock_client.models.generate_content.return_value = mock_res

    from unittest.mock import mock_open
    with (
        patch("google.genai.Client", return_value=mock_client),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=b"fake_bytes")),
    ):
        gemini = GeminiLLM()
        response = await gemini.generate(history=[msg])

        assert response.content == "Beautiful photo!"

        called_contents = mock_client.models.generate_content.call_args[1]["contents"]
        # Verify that user Content has two parts: the text, and the image bytes part
        assert len(called_contents[0].parts) == 2
        # Verify the text part
        assert called_contents[0].parts[0].text == "Look at this!"
        # Verify the bytes part loaded correctly
        assert called_contents[0].parts[1].inline_data.data == b"fake_bytes"
        assert called_contents[0].parts[1].inline_data.mime_type == "image/png"


@pytest.mark.asyncio
async def test_claude_llm_with_attachments() -> None:
    """Verify ClaudeLLM extracts and formats attachments correctly using base64 source blocks."""
    msg = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="User",
        role=ROLE_USER,
        type=TYPE_TEXT,
        content="Look at this PDF",
        metadata={
            "attachments": [
                {
                    "path": "/tmp/nonexistent_test.pdf",
                    "mime_type": "application/pdf",
                    "filename": "test.pdf",
                }
            ]
        },
    )

    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.content = [MagicMock(type="text", text="Extracted PDF text")]
    mock_res.usage.input_tokens = 12
    mock_res.usage.output_tokens = 6
    mock_client.messages.create.return_value = mock_res

    # Use patches to mock file presence and reading
    from unittest.mock import mock_open
    with (
        patch("anthropic.AnthropicVertex", return_value=mock_client),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=b"fake_pdf_bytes")),
    ):
        claude = ClaudeLLM()
        response = await claude.generate(history=[msg])

        assert response.content == "Extracted PDF text"

        called_kwargs = mock_client.messages.create.call_args[1]
        messages = called_kwargs["messages"]

        # Verify the PDF block was included correctly
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][1]["type"] == "document"
        assert messages[0]["content"][1]["source"]["media_type"] == "application/pdf"
        assert messages[0]["content"][1]["source"]["data"] == "ZmFrZV9wZGZfYnl0ZXM="


@pytest.mark.asyncio
async def test_image_mime_type_correction() -> None:
    """Verify GeminiLLM and ClaudeLLM correct the mime type based on magic bytes if they mismatch."""
    png_data = b"\x89PNG\r\n\x1a\n_fake_png_data"
    msg = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="User",
        role=ROLE_USER,
        type=TYPE_TEXT,
        content="Mismatched Image",
        metadata={
            "attachments": [
                {
                    "path": "/tmp/mismatched.jpg",
                    "mime_type": "image/jpeg",  # Incorrect metadata mime_type
                    "filename": "mismatched.jpg",
                }
            ]
        },
    )

    # 1. Test Claude LLM correction
    mock_claude_client = MagicMock()
    mock_claude_res = MagicMock()
    mock_claude_res.content = [MagicMock(type="text", text="Processed image")]
    mock_claude_client.messages.create.return_value = mock_claude_res

    from unittest.mock import mock_open
    with (
        patch("anthropic.AnthropicVertex", return_value=mock_claude_client),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=png_data)),
    ):
        claude = ClaudeLLM()
        await claude.generate(history=[msg])

        called_kwargs = mock_claude_client.messages.create.call_args[1]
        messages = called_kwargs["messages"]
        # Verify image was corrected to image/png in Claude source block
        assert messages[0]["content"][1]["source"]["media_type"] == "image/png"

    # 2. Test Gemini LLM correction
    mock_gemini_client = MagicMock()
    mock_gemini_res = MagicMock()
    mock_gemini_res.parts = [MagicMock(text="Processed image", thought=False, function_call=None)]
    mock_gemini_client.models.generate_content.return_value = mock_gemini_res

    with (
        patch("google.genai.Client", return_value=mock_gemini_client),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=png_data)),
    ):
        gemini = GeminiLLM()
        await gemini.generate(history=[msg])

        called_contents = mock_gemini_client.models.generate_content.call_args[1]["contents"]
        # Verify image was corrected to image/png in Gemini Part inline_data
        assert called_contents[0].parts[1].inline_data.mime_type == "image/png"


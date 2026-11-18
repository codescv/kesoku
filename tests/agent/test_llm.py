"""Unit tests for Kesoku LLM module and Anthropic Claude support."""

import asyncio
import io
import json
import os
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
import requests
from google.genai import _api_client, types
from PIL import Image

from kesoku.agent.agent import Agent, SessionWorker
from kesoku.agent.compressor import HistoryCompressor
from kesoku.agent.history import build_history
from kesoku.agent.llm import (
    _EXPLICIT_CACHE_UNSUPPORTED_LOCK,
    _EXPLICIT_CACHE_UNSUPPORTED_MODELS,
    _IMAGE_ATTACHMENT_LRU,
    _IMAGE_ATTACHMENT_LRU_LOCK,
    _IMAGE_ATTACHMENT_LRU_MAXSIZE,
    BaseLLM,
    ClaudeLLM,
    GeminiLLM,
    LLMResponse,
    MockLLM,
    ToolCallRequest,
    _drain_blocking_stream,
    _load_attachments,
    _patch_requests_response_utf8_lines,
    _summarize_gemini_parts,
    function_to_anthropic_tool,
    get_llm,
    history_to_turns,
    is_explicit_cache_unsupported_model,
    prewarm_image_attachments,
)
from kesoku.agent.tools.registry import ToolContext
from kesoku.agent.turn_executor import (
    TurnExecutor,
    discard_session_cache_state,
)
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import (
    ClaudeConfig,
    DiscordChannelOverride,
    DiscordConfig,
    GeminiConfig,
    KesokuConfig,
    WorkspaceConfig,
    resolve_gemini_cache_ttl,
)
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import AsyncDatabaseManager, DatabaseManager, Message
from kesoku.gateway.chatbot.base import (
    Chatbot,
)
from kesoku.gateway.chatbot.discord.adapter import DiscordChatbot
from kesoku.gateway.gateway import Gateway


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


def test_function_to_anthropic_tool_annotated() -> None:
    """Verify function_to_anthropic_tool correctly handles typing.Annotated type hints."""
    from typing import Annotated

    def annotated_tool_func(
        query: Annotated[str, "Query string"],
        max_results: Annotated[int | None, "Max results"] = 10,
    ) -> str:
        """Annotated tool.

        Args:
            query: Query string.
            max_results: Max results.
        """
        return query

    schema = function_to_anthropic_tool(annotated_tool_func)

    assert schema["name"] == "annotated_tool_func"
    assert schema["input_schema"]["properties"]["query"]["type"] == "string"
    assert schema["input_schema"]["properties"]["max_results"]["type"] == "integer"


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


def test_get_llm_with_context_override() -> None:
    """Verify get_llm applies use_context_compression overrides for provider and model configuration."""
    from kesoku.config import KesokuConfig

    cfg = KesokuConfig()
    cfg.agent.llm = "gemini"
    cfg.agent.context_llm = "claude"
    cfg.gemini.model_name = "gemini-3.5-pro"
    cfg.gemini.context_model_name = "gemini-2.0-flash-lite"
    cfg.claude.model_name = "claude-3-5-sonnet"
    cfg.claude.context_model_name = "claude-3-5-haiku"

    # Case 1: no use_context_compression (resolves to gemini model_name)
    with patch("kesoku.agent.llm.GeminiLLM") as mock_gemini:
        get_llm(config=cfg)
        mock_gemini.assert_called_once()
        called_config = mock_gemini.call_args[1]["config"]
        assert called_config.model_name == "gemini-3.5-pro"

    # Case 2: use_context_compression=True, which overrides provider to claude and uses claude's context_model_name
    with patch("kesoku.agent.llm.ClaudeLLM") as mock_claude:
        get_llm(config=cfg, use_context_compression=True)
        mock_claude.assert_called_once()
        called_config = mock_claude.call_args[1]["config"]
        assert called_config.model_name == "claude-3-5-haiku"

    # Case 3: use_context_compression=True, but provider explicitly overridden to gemini
    # (uses gemini's context_model_name)
    with patch("kesoku.agent.llm.GeminiLLM") as mock_gemini:
        get_llm(provider="gemini", config=cfg, use_context_compression=True)
        mock_gemini.assert_called_once()
        called_config = mock_gemini.call_args[1]["config"]
        assert called_config.model_name == "gemini-2.0-flash-lite"

    # Case 4: explicit model_name override for claude and gemini
    with patch("kesoku.agent.llm.ClaudeLLM") as mock_claude:
        get_llm(provider="claude", config=cfg, model_name="claude-opus-5-5")
        mock_claude.assert_called_once()
        called_config = mock_claude.call_args[1]["config"]
        assert called_config.model_name == "claude-opus-5-5"

    with patch("kesoku.agent.llm.GeminiLLM") as mock_gemini:
        get_llm(provider="gemini", config=cfg, model_name="gemini-3.8-pro")
        mock_gemini.assert_called_once()
        called_config = mock_gemini.call_args[1]["config"]
        assert called_config.model_name == "gemini-3.8-pro"


@pytest.mark.asyncio
async def test_claude_llm_generate_history_conversion() -> None:
    """Verify ClaudeLLM converts conversational history to Anthropic format correctly."""
    # Setup fake database messages for history
    msg_user = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello! Do a calculation.",
    )

    msg_thought = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        type=MessageType.THOUGHT,
        content="Thinking...",
    )

    msg_tool_call = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_CALL,
        content="Calling tool calculator",
        metadata={"tool_name": "calculator", "tool_arguments": {"expr": "2+2"}},
    )

    msg_tool_res = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="calculator",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_RESULT,
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
    mock_res.usage.cache_read_input_tokens = None
    mock_client.messages.create.return_value = mock_res

    with patch("kesoku.agent.llm.AnthropicVertex", return_value=mock_client):
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
        tool_use_block = next(b for b in messages[1]["content"] if b["type"] == "tool_use")
        assert tool_use_block["name"] == "calculator"
        assert tool_use_block["input"] == {"expr": "2+2"}
        assert tool_use_block["id"] == f"toolu_{msg_tool_call.id}"

        # Verify tool result was represented correctly in user role
        tool_res_block = next(b for m in messages for b in m["content"] if b["type"] == "tool_result")
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
        role=MessageRole.USER,
        type=MessageType.TEXT,
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
    mock_res.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5, total_token_count=15)
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
        role=MessageRole.USER,
        type=MessageType.TEXT,
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
    mock_res.usage.cache_read_input_tokens = None
    mock_client.messages.create.return_value = mock_res

    # Use patches to mock file presence and reading
    from unittest.mock import mock_open

    with (
        patch("kesoku.agent.llm.AnthropicVertex", return_value=mock_client),
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
        role=MessageRole.USER,
        type=MessageType.TEXT,
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
        patch("kesoku.agent.llm.AnthropicVertex", return_value=mock_claude_client),
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


def test_history_to_turns_conversion() -> None:
    """Verify history_to_turns correctly converts database messages to provider-neutral IR."""
    msg_system_1 = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="System",
        role=MessageRole.SYSTEM,
        type=MessageType.TEXT,
        content="System Prompt 1",
    )
    msg_system_2 = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="System",
        role=MessageRole.SYSTEM,
        type=MessageType.TEXT,
        content="System Prompt 2",
    )
    msg_user = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello!",
    )
    msg_thought = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        type=MessageType.THOUGHT,
        content="Hmm...",
    )
    msg_tool_call = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_CALL,
        content="Call calculator",
        metadata={"tool_name": "calculator", "tool_arguments": {"expr": "1+1"}, "tool_call_id": "call_abc"},
    )
    msg_tool_res = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="calculator",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_RESULT,
        content="2",
        metadata={"tool_name": "calculator", "tool_result": "2"},
        parent_id=msg_tool_call.id,
    )

    history = [msg_system_1, msg_system_2, msg_user, msg_thought, msg_tool_call, msg_tool_res]
    turns, system_prompt = history_to_turns(history, prompt="Continuing prompt")

    # Check system prompt resolution
    assert system_prompt == "System Prompt 1"

    # Check turn roles and merging
    # 1. user turn containing system notification 2 + Hello!
    # 2. assistant turn containing thought + tool call
    # 3. tool turn containing tool result
    # 4. user turn containing continuing prompt
    assert len(turns) == 4

    # Turn 1: User turn with System notification & User message
    assert turns[0].role == "user"
    assert len(turns[0].blocks) == 2
    assert turns[0].blocks[0].type == "text"
    assert turns[0].blocks[0].text == "[System Notification]\nSystem Prompt 2"
    assert turns[0].blocks[1].type == "text"
    assert turns[0].blocks[1].text == "Hello!"

    # Turn 2: Assistant turn with thought and tool call
    assert turns[1].role == "assistant"
    assert len(turns[1].blocks) == 2
    assert turns[1].blocks[0].type == "thought"
    assert turns[1].blocks[0].text == "Hmm..."
    assert turns[1].blocks[1].type == "tool_call"
    assert turns[1].blocks[1].name == "calculator"
    assert turns[1].blocks[1].arguments == {"expr": "1+1"}
    assert turns[1].blocks[1].tool_call_id == "call_abc"

    # Turn 3: Tool turn with tool result
    assert turns[2].role == "tool"
    assert len(turns[2].blocks) == 1
    assert turns[2].blocks[0].type == "tool_result"
    assert turns[2].blocks[0].name == "calculator"
    assert turns[2].blocks[0].tool_call_id == "call_abc"
    assert turns[2].blocks[0].result == "2"
    assert not turns[2].blocks[0].is_error

    # Turn 4: Continuing user prompt
    assert turns[3].role == "user"
    assert len(turns[3].blocks) == 1
    assert turns[3].blocks[0].text == "Continuing prompt"


def test_history_to_turns_with_sliced_system_notification() -> None:
    """Verify SYSTEM messages at index 0 of sliced history are not dropped when system_prompt is already provided."""
    msg_wakeup = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="System",
        role=MessageRole.SYSTEM,
        type=MessageType.TEXT,
        content="[System Alert] Background Job finished",
    )
    history = [msg_wakeup]
    turns, resolved_prompt = history_to_turns(history, prompt=None, system_prompt="# Active Persona")

    assert resolved_prompt == "# Active Persona"
    assert len(turns) == 1
    assert turns[0].role == "user"
    assert turns[0].blocks[0].text == "[System Notification]\n[System Alert] Background Job finished"


def test_image_resizing_and_compression() -> None:
    """Verify that _resize_and_compress_image resizes large images and outputs WebP."""
    import io

    from PIL import Image

    from kesoku.agent.llm import _resize_and_compress_image

    # Create a large dummy image in memory (2000x1000)
    img = Image.new("RGB", (2000, 1000), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    large_png_bytes = img_byte_arr.getvalue()

    # Resize and compress to WebP
    processed_bytes, mime_type = _resize_and_compress_image(large_png_bytes, max_size=1024)

    assert mime_type == "image/webp"
    # Verify the new image dimensions by loading it
    result_img = Image.open(io.BytesIO(processed_bytes))
    width, height = result_img.size
    # Should be scaled down such that max dimension is 1024
    assert width == 1024
    assert height == 512


def test_gemini_llm_count_tokens_success() -> None:
    """Verify GeminiLLM.count_tokens successfully calls Gemini API with CountTokensConfig."""
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.total_tokens = 42
    mock_client.models.count_tokens.return_value = mock_res

    msg = Message(
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Count me!",
    )

    with patch("google.genai.Client", return_value=mock_client):
        gemini = GeminiLLM()
        tokens = gemini.count_tokens(
            prompt="Additional prompt",
            system_prompt="Strict instructions",
            history=[msg],
        )

        assert tokens == 42
        mock_client.models.count_tokens.assert_called_once()
        called_kwargs = mock_client.models.count_tokens.call_args[1]
        assert called_kwargs["model"] == gemini.model_name
        assert len(called_kwargs["contents"]) == 1
        assert len(called_kwargs["contents"][0].parts) == 2

        config = called_kwargs["config"]
        assert config is not None
        assert config.system_instruction == "Strict instructions"


def test_gemini_llm_count_tokens_empty_bypass() -> None:
    """Verify GeminiLLM.count_tokens gracefully bypasses Gemini API and uses fallback for empty contents."""
    mock_client = MagicMock()

    with patch("google.genai.Client", return_value=mock_client):
        gemini = GeminiLLM()
        # Empty inputs
        tokens = gemini.count_tokens(prompt="", system_prompt="System prompt only", history=[])

        # Since contents are empty, it must bypass models.count_tokens
        mock_client.models.count_tokens.assert_not_called()
        # Fallback estimation should be used (e.g., 18 chars -> 4 tokens)
        assert tokens == 4


def test_gemini_llm_count_tokens_exception_fallback() -> None:
    """Verify GeminiLLM.count_tokens logs warning and falls back to estimation on API exception."""
    mock_client = MagicMock()
    mock_client.models.count_tokens.side_effect = Exception("API limit exceeded")

    msg = Message(
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Calculate my length.",
    )

    with (
        patch("google.genai.Client", return_value=mock_client),
        patch("kesoku.agent.llm.logger.warning") as mock_warn,
    ):
        gemini = GeminiLLM()
        tokens = gemini.count_tokens(prompt="More text", history=[msg])

        # Fallback estimation: 21 chars + 9 chars = 30 chars / 4 = 7 tokens
        assert tokens == 7
        mock_client.models.count_tokens.assert_called_once()
        mock_warn.assert_called_once()
        assert "Failed to count tokens via Gemini API" in mock_warn.call_args[0][0]


@pytest.mark.asyncio
async def test_gemini_llm_with_cached_content() -> None:
    """Verify GeminiLLM does not set system_instruction or tools in raw config when using cached_content."""
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.parts = [MagicMock(text="Response text", thought=False, function_call=None)]
    mock_res.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5, total_token_count=15)
    mock_client.models.generate_content.return_value = mock_res

    # A mock tool function
    def my_dummy_tool() -> str:
        return "dummy"

    # Case 1: Without cached_content
    with patch("google.genai.Client", return_value=mock_client):
        gemini = GeminiLLM()
        await gemini.generate(
            prompt="Hello",
            system_prompt="My system prompt",
            tools=[my_dummy_tool],
            cached_content=None,
        )
        called_kwargs = mock_client.models.generate_content.call_args[1]
        config = called_kwargs["config"]
        assert config.system_instruction == "My system prompt"
        assert len(config.tools) == 1
        assert config.cached_content is None

    mock_client.reset_mock()

    # Case 2: With cached_content
    with patch("google.genai.Client", return_value=mock_client):
        gemini = GeminiLLM()
        await gemini.generate(
            prompt="Hello",
            system_prompt="My system prompt",
            tools=[my_dummy_tool],
            cached_content="projects/123/locations/global/cachedContents/456",
        )
        called_kwargs = mock_client.models.generate_content.call_args[1]
        config = called_kwargs["config"]
        # When using cached content, they must be omitted in the raw config!
        assert config.system_instruction is None
        assert config.tools is None
        assert config.cached_content == "projects/123/locations/global/cachedContents/456"


@pytest.mark.asyncio
async def test_gemini_llm_thought_signature_propagation() -> None:
    """Verify GeminiLLM propagates thought_signature to all parallel function call parts in the same turn."""
    msg_user = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello!",
    )
    msg_tool_call_1 = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_CALL,
        content="Calling tool_1",
        metadata={
            "tool_name": "tool_1",
            "tool_arguments": {},
            "thought_signature": "018f3d6b5fa186d9b97ce8d38cb72b29f029963400",
        },
    )
    msg_tool_call_2 = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_CALL,
        content="Calling tool_2",
        metadata={
            "tool_name": "tool_2",
            "tool_arguments": {},
            "thought_signature": None,  # Missing signature (parallel tool call)
        },
    )
    msg_tool_res_1 = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="tool_1",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_RESULT,
        content="res 1",
        metadata={"tool_name": "tool_1", "tool_result": "1"},
        parent_id=msg_tool_call_1.id,
    )
    msg_tool_res_2 = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="tool_2",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_RESULT,
        content="res 2",
        metadata={"tool_name": "tool_2", "tool_result": "2"},
        parent_id=msg_tool_call_2.id,
    )

    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.parts = [MagicMock(text="Completed both!", thought=False, function_call=None)]
    mock_res.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5, total_token_count=15)
    mock_client.models.generate_content.return_value = mock_res

    with patch("google.genai.Client", return_value=mock_client):
        gemini = GeminiLLM()
        await gemini.generate(
            history=[msg_user, msg_tool_call_1, msg_tool_call_2, msg_tool_res_1, msg_tool_res_2],
        )

        called_contents = mock_client.models.generate_content.call_args[1]["contents"]
        # Turn 0: user (Hello!)
        # Turn 1: model (tool_1 call, tool_2 call) -> This is the assistant turn
        # Turn 2: user (tool responses)

        # Verify Turn 1 content (assistant)
        assistant_content = called_contents[1]
        assert assistant_content.role == "model"
        assert len(assistant_content.parts) == 2

        # Part 0: tool_1 function call (should have the original signature)
        assert assistant_content.parts[0].function_call.name == "tool_1"
        assert assistant_content.parts[0].thought_signature == bytes.fromhex(
            "018f3d6b5fa186d9b97ce8d38cb72b29f029963400"
        )

        # Part 1: tool_2 function call (should have the signature propagated!)
        assert assistant_content.parts[1].function_call.name == "tool_2"
        assert assistant_content.parts[1].thought_signature == bytes.fromhex(
            "018f3d6b5fa186d9b97ce8d38cb72b29f029963400"
        )


@pytest.mark.asyncio
async def test_gemini_llm_token_caching_calculations() -> None:
    """Verify GeminiLLM correctly aggregates prompt tokens when using cached_content."""
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.parts = [MagicMock(text="Response text", thought=False, function_call=None)]

    # Configure usage metadata:
    # prompt_token_count=10 (raw/uncached in API response), cached_content_token_count=100
    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.cached_content_token_count = 100
    usage.candidates_token_count = 5
    usage.total_token_count = 15  # Ignored/recomputed in our parser
    mock_res.usage_metadata = usage
    mock_client.models.generate_content.return_value = mock_res

    # Case 1: With cached_content -> prompt_tokens should be raw prompt_token_count
    with patch("google.genai.Client", return_value=mock_client):
        gemini = GeminiLLM()
        res = await gemini.generate(
            prompt="Hello",
            cached_content="projects/123/locations/global/cachedContents/456",
        )
        assert res.prompt_tokens == 10
        assert res.cached_tokens == 100
        assert res.total_tokens == 15

    # Case 2: Without cached_content -> prompt_tokens should be raw prompt_token_count
    with patch("google.genai.Client", return_value=mock_client):
        gemini = GeminiLLM()
        res = await gemini.generate(
            prompt="Hello",
            cached_content=None,
        )
        assert res.prompt_tokens == 10
        assert res.cached_tokens == 100
        assert res.total_tokens == 15


def test_history_to_turns_tool_results_sorting() -> None:
    """Verify history_to_turns correctly sorts tool results to match the preceding tool calls order."""
    msg_call_1 = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_CALL,
        content="Call 1",
        metadata={"tool_name": "tool_1", "tool_call_id": "call_1"},
    )
    msg_call_2 = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_CALL,
        content="Call 2",
        metadata={"tool_name": "tool_2", "tool_call_id": "call_2"},
    )

    msg_res_2 = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="tool_2",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_RESULT,
        content="res 2",
        metadata={"tool_name": "tool_2", "tool_result": "res 2", "tool_call_id": "call_2"},
        parent_id=msg_call_2.id,
    )
    msg_res_1 = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="tool_1",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_RESULT,
        content="res 1",
        metadata={"tool_name": "tool_1", "tool_result": "res 1", "tool_call_id": "call_1"},
        parent_id=msg_call_1.id,
    )

    # Input sequence has call_1, call_2, then res_2, res_1
    history = [msg_call_1, msg_call_2, msg_res_2, msg_res_1]
    turns, _ = history_to_turns(history)

    assert len(turns) == 2

    # Turn 0 (Assistant/Model) contains the tool calls
    assert turns[0].role == MessageRole.ASSISTANT
    assert len(turns[0].blocks) == 2
    assert turns[0].blocks[0].tool_call_id == "call_1"
    assert turns[0].blocks[1].tool_call_id == "call_2"

    # Turn 1 (Tool) contains tool results, sorted to match the call order (call_1, call_2)
    assert turns[1].role == MessageRole.TOOL
    assert len(turns[1].blocks) == 2
    assert turns[1].blocks[0].tool_call_id == "call_1"
    assert turns[1].blocks[1].tool_call_id == "call_2"


def test_gemini_llm_count_tokens_thought_signature() -> None:
    """Verify GeminiLLM.count_tokens accounts for thought_signature length in extra_tokens."""
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.total_tokens = 50
    mock_client.models.count_tokens.return_value = mock_res

    # A tool call message containing thought_signature hex string (representing 372 hex characters -> 186 bytes)
    # 186 bytes / 3.72 = 50 extra tokens
    ts_bytes = b"A" * 186
    ts_hex = ts_bytes.hex()

    msg = Message(
        session_id="sess_1",
        chatbot_id="discord",
        channel_id="chan_1",
        sender="Kesoku",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_CALL,
        content="calling a tool",
        metadata={
            "tool_name": "calc",
            "tool_arguments": {},
            "thought_signature": ts_hex,
        },
    )

    with patch("google.genai.Client", return_value=mock_client):
        gemini = GeminiLLM()
        tokens = gemini.count_tokens(history=[msg])

        # Extra tokens = 186 / 3.72 = 50
        # Total tokens = 50 (from API) + 50 (extra) = 100
        assert tokens == 100
        mock_client.models.count_tokens.assert_called_once()


@pytest.mark.asyncio
async def test_claude_llm_prompt_caching() -> None:
    """Verify ClaudeLLM sets cache_control headers when prompt exceeds threshold."""
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.content = [MagicMock(type="text", text="Hello.")]
    mock_res.usage.input_tokens = 20
    mock_res.usage.output_tokens = 5
    mock_res.usage.cache_read_input_tokens = 100
    mock_client.messages.create.return_value = mock_res

    # Use a config where context_caching is enabled and threshold is small
    config = ClaudeConfig(
        context_caching=True,
        context_caching_threshold=10,  # low threshold to trigger caching
    )

    with patch("kesoku.agent.llm.AnthropicVertex", return_value=mock_client):
        claude = ClaudeLLM(config=config)

        # A long prompt to trigger caching (approx 60 chars -> ~15 tokens, threshold is 10 tokens)
        long_prompt = "A" * 60
        system_prompt = "S" * 60

        def dummy_tool(x: int) -> int:
            """Dummy tool description.

            Args:
                x: parameter
            """
            return x

        response = await claude.generate(
            prompt=long_prompt,
            system_prompt=system_prompt,
            tools=[dummy_tool],
        )

        assert response.cached_tokens == 100
        assert response.prompt_tokens == 120  # input_tokens (20) + cached_tokens (100)

        mock_client.messages.create.assert_called_once()
        called_kwargs = mock_client.messages.create.call_args[1]

        # Verify system prompt has cache_control
        assert isinstance(called_kwargs["system"], list)
        assert called_kwargs["system"][0]["text"] == system_prompt
        assert called_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

        # Verify tools has cache_control on the last tool
        tools = called_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["cache_control"] == {"type": "ephemeral"}

        # Verify messages list has cache_control on the last message block
        messages = called_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_claude_llm_prompt_caching_disabled() -> None:
    """Verify ClaudeLLM does not set cache_control when caching is disabled or below threshold."""
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.content = [MagicMock(type="text", text="Hello.")]
    mock_res.usage.input_tokens = 20
    mock_res.usage.output_tokens = 5
    mock_res.usage.cache_read_input_tokens = None
    mock_client.messages.create.return_value = mock_res

    # Use a config where context_caching is disabled
    config_disabled = ClaudeConfig(
        context_caching=False,
        context_caching_threshold=10,
    )

    with patch("kesoku.agent.llm.AnthropicVertex", return_value=mock_client):
        # Test 1: Caching disabled
        claude = ClaudeLLM(config=config_disabled)
        await claude.generate(
            prompt="A" * 60,
            system_prompt="S" * 60,
        )
        called_kwargs = mock_client.messages.create.call_args[1]
        assert called_kwargs["system"] == "S" * 60
        assert "cache_control" not in called_kwargs["messages"][0]["content"][-1]

    mock_client.reset_mock()

    # Use a config where caching threshold is very high (e.g. 1000 tokens)
    config_high_threshold = ClaudeConfig(
        context_caching=True,
        context_caching_threshold=1000,
    )

    with patch("kesoku.agent.llm.AnthropicVertex", return_value=mock_client):
        # Test 2: Prompt size is below threshold
        claude = ClaudeLLM(config=config_high_threshold)
        await claude.generate(
            prompt="A" * 60,
            system_prompt="S" * 60,
        )
        called_kwargs = mock_client.messages.create.call_args[1]
        assert called_kwargs["system"] == "S" * 60
        assert "cache_control" not in called_kwargs["messages"][0]["content"][-1]


@pytest.mark.asyncio
async def test_claude_llm_native_thinking() -> None:
    """Verify ClaudeLLM supports native thinking parameters, parsing, and propagation."""
    # 1. Verify request construction with thinking_level
    config = ClaudeConfig(thinking_level="high")

    mock_client = MagicMock()
    mock_res = MagicMock()

    # Mock thinking block and text block
    mock_thinking_block = MagicMock()
    mock_thinking_block.type = "thinking"
    mock_thinking_block.thinking = "Internal thought process"
    mock_thinking_block.signature = "sig-12345"

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Hello there!"

    mock_res.content = [mock_thinking_block, mock_text_block]
    mock_res.usage.input_tokens = 20
    mock_res.usage.output_tokens = 30
    mock_res.usage.cache_read_input_tokens = None
    mock_client.messages.create.return_value = mock_res

    with patch("kesoku.agent.llm.AnthropicVertex", return_value=mock_client):
        claude = ClaudeLLM(config=config)
        response = await claude.generate(prompt="What is on your mind?")

        # Verify response parsing
        assert response.content == "Hello there!"
        assert response.thought == "Internal thought process"
        assert response.thought_signature == "sig-12345"

        # Verify request parameters
        called_kwargs = mock_client.messages.create.call_args[1]
        assert called_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert called_kwargs["output_config"] == {"effort": "high"}

    mock_client.reset_mock()

    # 2. Verify history propagation with thought signature
    msg_user = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="What is on your mind?",
    )
    msg_thought = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        type=MessageType.THOUGHT,
        content="Internal thought process",
        metadata={"thought_signature": "sig-12345"},
    )
    msg_assistant = Message(
        session_id="session-123",
        chatbot_id="discord",
        channel_id="channel-123",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="Hello there!",
    )

    mock_res_2 = MagicMock()
    mock_res_2.content = [mock_text_block]
    mock_res_2.usage.input_tokens = 10
    mock_res_2.usage.output_tokens = 10
    mock_res_2.usage.cache_read_input_tokens = None
    mock_client.messages.create.return_value = mock_res_2

    with patch("kesoku.agent.llm.AnthropicVertex", return_value=mock_client):
        claude = ClaudeLLM(config=config)
        await claude.generate(
            prompt="Tell me more.",
            history=[msg_user, msg_thought, msg_assistant]
        )

        called_kwargs = mock_client.messages.create.call_args[1]
        messages = called_kwargs["messages"]

        # Expect alternating messages: User (prompt1) -> Assistant (thinking + text) -> User (prompt2)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"

        assistant_msg = messages[1]
        assert assistant_msg["role"] == "assistant"

        # Should contain two blocks: thinking and text
        content = assistant_msg["content"]
        assert len(content) == 2

        assert content[0] == {
            "type": "thinking",
            "thinking": "Internal thought process",
            "signature": "sig-12345"
        }
        assert content[1] == {
            "type": "text",
            "text": "Hello there!"
        }


def test_get_llm_thinking_level_override_and_validation() -> None:
    """Verify `get_llm(thinking_level=...)` applies valid overrides and warns+ignores invalid ones."""
    cfg = KesokuConfig(
        gemini=GeminiConfig(thinking_level="high"),
        claude=ClaudeConfig(thinking_level="medium"),
    )

    with (
        patch("google.genai.Client", return_value=MagicMock()),
        patch("kesoku.agent.llm.logger.warning") as mock_warn,
    ):
        g_low = get_llm(provider="gemini", config=cfg, thinking_level="low")
        assert isinstance(g_low, GeminiLLM)
        assert g_low.config.thinking_level == "low"
        # Original config untouched
        assert cfg.gemini.thinking_level == "high"
        mock_warn.assert_not_called()

        g_xhigh = get_llm(provider="gemini", config=cfg, thinking_level="xhigh")
        assert isinstance(g_xhigh, GeminiLLM)
        assert g_xhigh.config.thinking_level == "xhigh"
        assert cfg.gemini.thinking_level == "high"
        mock_warn.assert_not_called()

        g_invalid = get_llm(provider="gemini", config=cfg, thinking_level="max")
        assert isinstance(g_invalid, GeminiLLM)
        assert g_invalid.config.thinking_level == "high"
        mock_warn.assert_called_once()
        assert "Ignoring unsupported thinking_level 'max' for Gemini" in mock_warn.call_args[0][0]

    with (
        patch("kesoku.agent.llm.AnthropicVertex", return_value=MagicMock()),
        patch("kesoku.agent.llm.logger.warning") as mock_warn,
    ):
        c_max = get_llm(provider="claude", config=cfg, thinking_level="max")
        assert isinstance(c_max, ClaudeLLM)
        assert c_max.config.thinking_level == "max"
        assert cfg.claude.thinking_level == "medium"
        mock_warn.assert_not_called()

        c_invalid = get_llm(provider="claude", config=cfg, thinking_level="minimal")
        assert isinstance(c_invalid, ClaudeLLM)
        assert c_invalid.config.thinking_level == "medium"
        mock_warn.assert_called_once()
        assert "Ignoring unsupported thinking_level 'minimal' for Claude" in mock_warn.call_args[0][0]


def test_kesoku_context_get_llm_caching_and_invalidation(tmp_path: Any) -> None:
    """Verify `KesokuContext.get_llm` caches by key, invalidates on config change, and respects `_llm` injection."""
    db_path = str(tmp_path / "ctx_cache.db")
    DatabaseManager(db_path).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path))

    with (
        patch("google.genai.Client", return_value=MagicMock()),
        patch("kesoku.agent.llm.AnthropicVertex", return_value=MagicMock()),
    ):
        ctx = KesokuContext(config=cfg)
        llm_a1 = ctx.get_llm(provider="gemini", thinking_level="low")
        llm_a2 = ctx.get_llm(provider="gemini", thinking_level="low")
        llm_high = ctx.get_llm(provider="gemini", thinking_level="high")
        llm_lite = ctx.get_llm(provider="gemini", model_name="gemini-3.5-flash-lite", thinking_level="low")
        llm_ctx_comp = ctx.get_llm(provider="gemini", use_context_compression=True, thinking_level="low")

        assert llm_a1 is llm_a2
        assert llm_a1 is not llm_high
        assert llm_a1 is not llm_lite
        assert llm_a1 is not llm_ctx_comp

        # Config reload invalidates cached instances
        cfg2 = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path))
        ctx._config = cfg2
        llm_after_reload = ctx.get_llm(provider="gemini", thinking_level="low")
        assert llm_after_reload is not llm_a1

        # Injected `_llm` short-circuits default lookups, while explicit overrides still resolve dynamically
        injected = MockLLM(mock_response="injected")
        ctx_inj = KesokuContext(config=cfg, llm=injected)
        assert ctx_inj.get_llm() is injected
        assert ctx_inj.get_llm(use_context_compression=True) is injected
        override_llm = ctx_inj.get_llm(thinking_level="low")
        assert override_llm is not injected
        assert isinstance(override_llm, GeminiLLM)


@pytest.mark.asyncio
async def test_gemini_llm_streaming_parity_and_edge_cases() -> None:
    """Verify GeminiLLM streaming emits text-only deltas, guards None chunks, and matches non-streaming parse."""
    sig_bytes = b"\x01\x02\x03\x04"
    part_thought = types.Part.from_text(text="Thinking step...")
    part_thought.thought = True
    part_text_1 = types.Part.from_text(text="Hello")
    part_text_2 = types.Part.from_text(text=" world!")
    part_fc = types.Part.from_function_call(name="search", args={"q": "kesoku"})
    part_fc.thought_signature = sig_bytes
    part_empty_sig = types.Part.from_text(text="")
    part_empty_sig.thought_signature = sig_bytes

    partial_usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=12,
        candidates_token_count=2,
        total_token_count=14,
    )
    final_usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=12,
        candidates_token_count=9,
        total_token_count=21,
        cached_content_token_count=4,
    )

    chunks = [
        # Chunk 0: candidates=None (e.g. initial metadata/feedback chunk)
        types.GenerateContentResponse.model_construct(candidates=None, usage_metadata=None, prompt_feedback=None),
        # Chunk 1: thought part + partial usage (must NOT trigger stream callback)
        types.GenerateContentResponse.model_construct(
            candidates=[
                types.Candidate.model_construct(
                    content=types.Content.model_construct(role="model", parts=[part_thought]),
                    finish_reason=None,
                )
            ],
            usage_metadata=partial_usage,
        ),
        # Chunk 2 & 3: text deltas
        types.GenerateContentResponse.model_construct(
            candidates=[
                types.Candidate.model_construct(
                    content=types.Content.model_construct(role="model", parts=[part_text_1]),
                    finish_reason=None,
                )
            ],
            usage_metadata=None,
        ),
        types.GenerateContentResponse.model_construct(
            candidates=[
                types.Candidate.model_construct(
                    content=types.Content.model_construct(role="model", parts=[part_text_2, part_fc, part_empty_sig]),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            usage_metadata=None,
        ),
        # Chunk 4: content.parts=None with final usage metadata
        types.GenerateContentResponse.model_construct(
            candidates=[
                types.Candidate.model_construct(
                    content=types.Content.model_construct(role="model", parts=None),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            usage_metadata=final_usage,
        ),
    ]

    non_stream_raw = types.GenerateContentResponse.model_construct(
        candidates=[
            types.Candidate.model_construct(
                content=types.Content.model_construct(
                    role="model",
                    parts=[part_thought, part_text_1, part_text_2, part_fc, part_empty_sig],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=final_usage,
    )

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.side_effect = lambda **_: iter(chunks)
    mock_client.models.generate_content.return_value = non_stream_raw

    seen_deltas: list[tuple[str, str]] = []

    async def on_chunk(delta: str, accumulated: str) -> None:
        seen_deltas.append((delta, accumulated))
        if delta == "Hello":
            raise RuntimeError("callback failure must not abort stream")

    with patch("google.genai.Client", return_value=mock_client):
        gemini = GeminiLLM()
        stream_res = await gemini.generate(prompt="Hi", stream=True, on_stream_chunk=on_chunk)
        non_stream_res = await gemini.generate(prompt="Hi", stream=False)

    mock_client.models.generate_content_stream.assert_called_once()
    mock_client.models.generate_content.assert_called_once()

    # Thoughts and empty-text signature parts were excluded from callback deltas
    assert seen_deltas == [
        ("Hello", "Hello"),
        (" world!", "Hello world!"),
    ]
    assert stream_res.content == non_stream_res.content == "Hello world!"
    assert stream_res.thought == non_stream_res.thought == "Thinking step..."
    assert stream_res.tool_calls == non_stream_res.tool_calls
    assert len(stream_res.tool_calls) == 1
    assert stream_res.tool_calls[0].thought_signature == sig_bytes.hex()
    assert stream_res.prompt_tokens == non_stream_res.prompt_tokens == 12
    assert stream_res.candidates_tokens == non_stream_res.candidates_tokens == 9
    assert stream_res.total_tokens == non_stream_res.total_tokens == 21
    assert stream_res.cached_tokens == non_stream_res.cached_tokens == 4
    assert stream_res.finish_reason == non_stream_res.finish_reason == "STOP"


@pytest.mark.asyncio
async def test_claude_llm_streaming_and_callback_resilience() -> None:
    """Verify ClaudeLLM streaming yields text deltas via `messages.stream` and parses the final Message."""
    final_msg = MagicMock()
    thinking_block = MagicMock(type="thinking", thinking="Claude reasoning", signature="sig-c")
    text_block = MagicMock(type="text", text="Bonjour le monde!")
    tool_block = MagicMock(type="tool_use", id="toolu_1", input={"x": 1})
    tool_block.name = "calc"
    final_msg.content = [thinking_block, text_block, tool_block]
    final_msg.usage.input_tokens = 15
    final_msg.usage.output_tokens = 8
    final_msg.usage.cache_read_input_tokens = 0
    final_msg.usage.cache_creation_input_tokens = 0

    events = [
        SimpleNamespace(type="thinking", thinking="Claude reasoning"),
        SimpleNamespace(type="text", text="Bonjour", snapshot="Bonjour"),
        SimpleNamespace(type="text", text=" le monde!", snapshot="Bonjour le monde!"),
    ]

    class _FakeStreamCtx:
        def __enter__(self) -> "_FakeStreamCtx":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        def __iter__(self) -> Any:
            return iter(events)

        def get_final_message(self) -> Any:
            return final_msg

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _FakeStreamCtx()
    mock_client.messages.create.return_value = final_msg

    seen: list[tuple[str, str]] = []

    async def flaky_cb(delta: str, accumulated: str) -> None:
        seen.append((delta, accumulated))
        if delta == "Bonjour":
            raise ValueError("ignored callback error")

    with patch("kesoku.agent.llm.AnthropicVertex", return_value=mock_client):
        claude = ClaudeLLM(config=ClaudeConfig())
        stream_res = await claude.generate(prompt="Salut", stream=True, on_stream_chunk=flaky_cb)
        non_stream_res = await claude.generate(prompt="Salut", stream=False)

    mock_client.messages.stream.assert_called_once()
    mock_client.messages.create.assert_called_once()
    assert seen == [
        ("Bonjour", "Bonjour"),
        (" le monde!", "Bonjour le monde!"),
    ]
    assert stream_res.content == non_stream_res.content == "Bonjour le monde!"
    assert stream_res.thought == non_stream_res.thought == "Claude reasoning"
    assert stream_res.thought_signature == non_stream_res.thought_signature == "sig-c"
    assert stream_res.tool_calls == non_stream_res.tool_calls
    assert stream_res.total_tokens == non_stream_res.total_tokens == 23


@pytest.mark.asyncio
async def test_gemini_llm_stream_requests_response_utf8_with_0x85_bytes() -> None:
    """Verify `requests.Response` SSE with `ISO-8859-1` and `0x85` UTF-8 bytes streams intact Chinese text."""
    chunk_text_1 = "去印度旅游有什么"
    chunk_text_2 = "注意事项，充满魅力"
    full_expected = chunk_text_1 + chunk_text_2

    sse_event_1 = (
        "data: "
        + json.dumps(
            {"candidates": [{"content": {"role": "model", "parts": [{"text": chunk_text_1}]}}]},
            ensure_ascii=False,
        )
        + "\r\n\r\n"
    ).encode("utf-8")
    sse_event_2 = (
        "data: "
        + json.dumps(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": chunk_text_2}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 9,
                    "candidatesTokenCount": 14,
                    "totalTokenCount": 23,
                },
            },
            ensure_ascii=False,
        )
        + "\r\n\r\n"
    ).encode("utf-8")
    raw_sse = sse_event_1 + sse_event_2

    # Sanity-check that `旅` (\xe6\x97\x85) and `充` (\xe5\x85\x85) embed literal 0x85 bytes.
    assert b"\xe6\x97\x85" in sse_event_1
    assert b"\xe5\x85\x85" in sse_event_2
    assert raw_sse.count(b"\x85") >= 2

    # Split a network chunk right before the 0x85 continuation byte of `旅` (\xe6\x97 | \x85)
    # to also verify multi-byte UTF-8 reassembly across transport chunks.
    split_pos = raw_sse.index(b"\xe6\x97\x85") + 2
    network_chunks = [raw_sse[:split_pos], raw_sse[split_pos:]]

    class _ChunkedRawStream:
        def __init__(self, payloads: list[bytes]) -> None:
            self._payloads = payloads
            self.released = False

        def stream(self, amt: int | None = None, decode_content: bool = True) -> Any:
            _ = (amt, decode_content)
            yield from self._payloads

        def release_conn(self) -> None:
            self.released = True

        def close(self) -> None:
            self.released = True

    raw_stream = _ChunkedRawStream(network_chunks)
    req_resp = requests.Response()
    req_resp.status_code = 200
    req_resp.headers["Content-Type"] = "text/event-stream"
    req_resp.encoding = requests.utils.get_encoding_from_headers(req_resp.headers)
    assert req_resp.encoding == "ISO-8859-1"
    req_resp.raw = raw_stream

    http_resp = _api_client.HttpResponse(req_resp.headers, req_resp)
    gemini = GeminiLLM(
        config=GeminiConfig(
            auth_mode="api_key",
            api_key="test-key",
            model_name="gemini-3.5-flash-lite",
            thinking_level="low",
        )
    )

    seen_chunks: list[tuple[str, str]] = []

    async def on_chunk(delta: str, accumulated: str) -> None:
        seen_chunks.append((delta, accumulated))

    with patch.object(gemini.client._api_client, "_request", return_value=http_resp):
        res = await gemini.generate(
            prompt="去印度旅游有什么注意事项",
            stream=True,
            on_stream_chunk=on_chunk,
        )

    assert seen_chunks == [
        (chunk_text_1, chunk_text_1),
        (chunk_text_2, full_expected),
    ]
    assert res.content == full_expected
    assert res.finish_reason == "STOP"
    assert res.total_tokens == 23
    assert raw_stream.released is True


def test_patch_requests_response_utf8_lines_idempotent_and_bytes_passthrough() -> None:
    """Verify `_patch_requests_response_utf8_lines` is idempotent and preserves `decode_unicode=False`."""
    payload = "去印度旅游，充满魅力\n第二行\n".encode()
    resp = requests.Response()
    resp.status_code = 200
    resp.headers["Content-Type"] = "text/event-stream"
    resp.encoding = "ISO-8859-1"
    resp.raw = io.BytesIO(payload)

    patched_once = _patch_requests_response_utf8_lines(resp)
    iter_fn = patched_once.iter_lines
    patched_twice = _patch_requests_response_utf8_lines(resp)
    assert patched_twice is resp
    assert patched_twice.iter_lines is iter_fn
    assert resp.encoding == "utf-8"

    lines = list(resp.iter_lines(decode_unicode=True, delimiter="\n"))
    assert lines == ["去印度旅游，充满魅力", "第二行", ""]

    resp_bytes = requests.Response()
    resp_bytes.status_code = 200
    resp_bytes.encoding = "ISO-8859-1"
    resp_bytes.raw = io.BytesIO(payload)
    _patch_requests_response_utf8_lines(resp_bytes)
    raw_lines = list(resp_bytes.iter_lines(decode_unicode=False))
    assert raw_lines == ["去印度旅游，充满魅力".encode(), "第二行".encode()]


def test_gemini_thinking_level_xhigh_config_and_wire_mapping() -> None:
    """Verify `thinking_level='xhigh'` validates on `GeminiConfig` and maps cleanly to uppercase `'XHIGH'`."""
    cfg = GeminiConfig(model_name="gemini-4-hydrogen", thinking_level="xhigh", auth_mode="api_key", api_key="test-key")
    assert cfg.thinking_level == "xhigh"

    with patch("google.genai.Client", return_value=MagicMock()):
        llm = GeminiLLM(config=cfg)
        turns, sys_prompt = history_to_turns(None, "Solve this", "System prompt")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            native = llm._build_native_input(turns=turns, system_prompt=sys_prompt, tools=None)

    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    gen_cfg = native["config"]
    assert gen_cfg.thinking_config is not None
    assert gen_cfg.thinking_config.include_thoughts is True
    lvl_val = getattr(gen_cfg.thinking_config.thinking_level, "value", gen_cfg.thinking_config.thinking_level)
    assert lvl_val == "XHIGH"
    dumped = gen_cfg.thinking_config.model_dump(mode="json", exclude_none=True)
    assert dumped["thinking_level"] == "XHIGH"


@pytest.mark.asyncio
async def test_gemini_explicit_cache_unsupported_model_detection_and_isolation() -> None:
    """Verify 400 explicit-cache capability errors mark only that model unsupported, log INFO once, and skip RPCs."""
    with _EXPLICIT_CACHE_UNSUPPORTED_LOCK:
        saved = set(_EXPLICIT_CACHE_UNSUPPORTED_MODELS)
        _EXPLICIT_CACHE_UNSUPPORTED_MODELS.clear()

    try:
        assert MockLLM().supports_explicit_cache() is False
        with patch("kesoku.agent.llm.AnthropicVertex", return_value=MagicMock()):
            assert ClaudeLLM(ClaudeConfig()).supports_explicit_cache() is False

        mock_client = MagicMock()

        def _create_side_effect(*, model: str, config: Any) -> Any:
            _ = config
            if model == "gemini-4-hydrogen":
                raise RuntimeError(
                    "400 INVALID_ARGUMENT. {'error': {'code': 400, "
                    "'message': \"Model 'gemini-4-hydrogen' does not support explicit context caching.\", "
                    "'status': 'INVALID_ARGUMENT'}}"
                )
            return SimpleNamespace(name=f"projects/p/locations/global/cachedContents/cache_{model}")

        mock_client.caches.create.side_effect = _create_side_effect

        with patch("google.genai.Client", return_value=mock_client):
            h2_llm = GeminiLLM(
                GeminiConfig(
                    model_name="gemini-4-hydrogen",
                    context_caching=True,
                    context_caching_mode="explicit",
                )
            )
            flash_llm = GeminiLLM(
                GeminiConfig(
                    model_name="gemini-3.8-flash",
                    context_caching=True,
                    context_caching_mode="explicit",
                )
            )
            disabled_llm = GeminiLLM(GeminiConfig(model_name="gemini-3.8-flash", context_caching=False))
            implicit_llm = GeminiLLM(
                GeminiConfig(model_name="gemini-3.8-flash", context_caching=True, context_caching_mode="implicit")
            )

            assert disabled_llm.supports_explicit_cache() is False
            assert implicit_llm.supports_explicit_cache() is False
            assert h2_llm.supports_explicit_cache() is True
            assert flash_llm.supports_explicit_cache() is True

            msg = Message(
                session_id="s1",
                chatbot_id="c1",
                channel_id="ch1",
                sender="u1",
                role=MessageRole.USER,
                content="Hello hydrogen",
            )

            with (
                patch("kesoku.agent.llm.logger.info") as mock_info,
                patch("kesoku.agent.llm.logger.warning") as mock_warn,
            ):
                res1 = await h2_llm.create_cache([msg], system_prompt="sys")

            assert res1 is None
            assert is_explicit_cache_unsupported_model("gemini-4-hydrogen") is True
            assert "gemini-4-hydrogen" in _EXPLICIT_CACHE_UNSUPPORTED_MODELS
            assert h2_llm.supports_explicit_cache() is False
            mock_warn.assert_not_called()
            info_msgs = [call.args[0] for call in mock_info.call_args_list]
            assert (
                "Model 'gemini-4-hydrogen' does not support explicit context caching; falling back to implicit caching."
                in info_msgs
            )
            assert mock_client.caches.create.call_count == 1

            # Second call on `gemini-4-hydrogen` returns None immediately with zero RPCs and zero duplicate logs.
            with (
                patch("kesoku.agent.llm.logger.info") as mock_info_2,
                patch("kesoku.agent.llm.logger.warning") as mock_warn_2,
            ):
                res2 = await h2_llm.create_cache([msg], system_prompt="sys")

            assert res2 is None
            assert mock_client.caches.create.call_count == 1
            mock_info_2.assert_not_called()
            mock_warn_2.assert_not_called()

            # `gemini-3.8-flash` remains explicit-cache capable and creates its cache cleanly.
            assert flash_llm.supports_explicit_cache() is True
            flash_cache = await flash_llm.create_cache([msg], system_prompt="sys")
            assert flash_cache == "projects/p/locations/global/cachedContents/cache_gemini-3.8-flash"
            assert mock_client.caches.create.call_count == 2
    finally:
        with _EXPLICIT_CACHE_UNSUPPORTED_LOCK:
            _EXPLICIT_CACHE_UNSUPPORTED_MODELS.clear()
            _EXPLICIT_CACHE_UNSUPPORTED_MODELS.update(saved)


class _DummyChatbot(Chatbot):
    """Minimal concrete Chatbot for testing base Chatbot behavior."""

    def __init__(self, chatbot_id: str, gateway: Gateway) -> None:
        super().__init__(chatbot_id=chatbot_id, gateway=gateway)
        self.sent_chunks: list[list[str]] = []
        self.sent_questions: list[dict[str, Any]] = []
        self.sent_files: list[str] = []

    async def handle_message(self, message: Message) -> None:
        await self.render_outgoing_message(message)

    async def send_text_chunks(self, channel_id: str, chunks: list[str], message: Message) -> None:
        self.sent_chunks.append(list(chunks))

    async def send_file_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        self.sent_files.append(file_path)

    async def send_question_segment(
        self,
        channel_id: str,
        question: str,
        choices: list[str],
        message: Message,
        multi: bool = False,
        max_choices: int = 1,
    ) -> None:
        self.sent_questions.append(
            {"question": question, "choices": list(choices), "multi": multi, "max": max_choices}
        )


def _make_png_bytes(width: int = 1600, height: int = 1200, color: tuple[int, int, int] = (40, 120, 200)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _make_gateway(tmp_path: Any, cfg: KesokuConfig | None = None) -> tuple[Gateway, AsyncDatabaseManager]:
    sessions_dir = str(tmp_path / "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    db_path = str(tmp_path / "kesoku.db")
    if cfg is None:
        cfg = KesokuConfig(
            agent_working_dir=str(tmp_path),
            workspace=WorkspaceConfig(sessions_dir=sessions_dir, db_path=db_path),
        )
    ctx = KesokuContext(config=cfg)
    ctx.sync_db.init_tables()
    gw = Gateway(context=ctx)
    return gw, ctx.db


def _make_discord_bot(gateway: Gateway, chatbot_id: str = "discord_v015") -> DiscordChatbot:
    bot = DiscordChatbot(
        chatbot_id=chatbot_id,
        gateway=gateway,
        bot_token="fake-token",
        discord_config=DiscordConfig(chatbot_id=chatbot_id, bot_token="fake-token"),
    )
    bot._deferred_poll_initial_sec = 0.02
    bot._deferred_poll_max_sec = 0.04
    bot._deferred_dead_grace_sec = 0.08
    return bot


# origin: tests/test_round3_fixes.py K1
@pytest.mark.asyncio
async def test_k1_gemini_parse_prompt_blocked_non_stream_and_stream() -> None:
    cfg = GeminiConfig(auth_mode="api_key", api_key="dummy")
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        raw_non_stream = types.GenerateContentResponse.model_construct(
            candidates=[],
            prompt_feedback=types.GenerateContentResponsePromptFeedback.model_construct(
                block_reason=types.BlockedReason.PROHIBITED_CONTENT
            ),
        )
        mock_client.models.generate_content.return_value = raw_non_stream
        llm = GeminiLLM(config=cfg)
        res = await llm.generate(prompt="hello")
        assert res.finish_reason == "PROMPT_BLOCKED:PROHIBITED_CONTENT"
        assert res.content == ""

        # Streaming chunk with candidates=[] and prompt_feedback.block_reason
        stream_chunk = SimpleNamespace(
            candidates=[],
            usage_metadata=None,
            prompt_feedback=SimpleNamespace(block_reason=types.BlockedReason.PROHIBITED_CONTENT),
        )
        mock_client.models.generate_content_stream.return_value = iter([stream_chunk])
        res_stream = await llm.generate(prompt="hello", stream=True)
        assert res_stream.finish_reason == "PROMPT_BLOCKED:PROHIBITED_CONTENT"

        # Unblocked empty MagicMock stays None (does not false-trigger on MagicMock attributes)
        mock_client.models.generate_content.return_value = MagicMock(parts=[], candidates=[])
        res_plain = await llm.generate(prompt="hello")
        assert res_plain.finish_reason is None


# origin: tests/test_round3_fixes.py K1
@pytest.mark.asyncio
async def test_k1_turn_executor_prompt_blocked_single_call_and_history_filtered(tmp_path: Path) -> None:
    db_path = str(tmp_path / "k1.db")
    DatabaseManager(db_path).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(tmp_path / "sessions")))
    cfg.gemini.stream = True
    gw = Gateway(context=KesokuContext(config=cfg))
    gw.register_stream_handler("discord_test", AsyncMock())
    await gw.create_session("s_k1", title="K1 Session", chatbot_id="discord_test", channel_id="c1")

    # Turn 1: clean prior exchange
    u1 = Message(
        session_id="s_k1",
        chatbot_id="discord_test",
        channel_id="c1",
        sender="u",
        role=MessageRole.USER,
        content="Normal question",
        status=MessageStatus.PROCESSED,
        timestamp=100.0,
    )
    a1 = Message(
        session_id="s_k1",
        chatbot_id="discord_test",
        channel_id="c1",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Normal answer",
        status=MessageStatus.DELIVERED,
        parent_id=u1.id,
        timestamp=101.0,
    )
    await gw.post(u1)
    await gw.post(a1)

    # Turn 2: two merged/queued user rows in the blocked turn
    u2_first = Message(
        session_id="s_k1",
        chatbot_id="discord_test",
        channel_id="c1",
        sender="u",
        role=MessageRole.USER,
        content="Queued part 1",
        status=MessageStatus.INTERRUPTED,
        timestamp=200.0,
    )
    u2_active = Message(
        session_id="s_k1",
        chatbot_id="discord_test",
        channel_id="c1",
        sender="u",
        role=MessageRole.USER,
        content="Queued part 2",
        status=MessageStatus.PENDING_AGENT,
        timestamp=201.0,
    )
    await gw.post(u2_first)
    await gw.post(u2_active)

    calls = 0

    class BlockedLLM(BaseLLM):
        config = cfg.gemini

        async def generate(self, **kwargs: Any) -> LLMResponse:
            nonlocal calls
            calls += 1
            on_chunk = kwargs.get("on_stream_chunk")
            if on_chunk:
                await on_chunk("partial", "partial")
            return LLMResponse(content="", finish_reason="PROMPT_BLOCKED:PROHIBITED_CONTENT")

    llm = BlockedLLM()
    ctx = KesokuContext(config=cfg, llm=llm)
    gw = Gateway(context=ctx)
    stream_events: list[tuple[Any, ...]] = []
    gw.register_stream_handler("discord_test", AsyncMock(side_effect=lambda *args: stream_events.append(args)))

    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    tool_runner.tool_context = ToolContext(gateway=gw, session_id="s_k1")
    executor = TurnExecutor("s_k1", gw, tool_runner, MagicMock(spec=TurnLogger), context=ctx)

    worker = MagicMock()
    type(worker).running = PropertyMock(return_value=True)
    worker.drain_queue_and_pivot = AsyncMock(side_effect=lambda m: m)
    worker.queue_empty.return_value = True

    await executor.process_turn(u2_active, worker, str(tmp_path / "staging"))

    # (b) Exactly 1 LLM call, 0 System nudges, 1 Notification warning with /compact & /clear, stream discarded
    assert calls == 1
    raw_msgs = await gw.db.get_session_messages("s_k1")
    assert [m for m in raw_msgs if m.role == MessageRole.SYSTEM] == []
    notices = [m for m in raw_msgs if m.role == MessageRole.ASSISTANT and m.sender == "Notification"]
    assert len(notices) == 1
    assert "/compact" in notices[0].content
    assert "/clear" in notices[0].content
    assert notices[0].metadata.get("delivery_notice") is True
    assert notices[0].metadata.get("prompt_blocked") == "PROHIBITED_CONTENT"
    assert any(len(ev) >= 4 and ev[3] is None for ev in stream_events)

    # Both merged user rows stamped with prompt_blocked
    by_id = {m.id: m for m in raw_msgs}
    assert by_id[u2_first.id].metadata.get("prompt_blocked") == "PROHIBITED_CONTENT"
    assert by_id[u2_active.id].metadata.get("prompt_blocked") == "PROHIBITED_CONTENT"

    # (c) build_history excludes both the Notification warning and the blocked user rows
    clean_hist = await build_history(gw, "s_k1", heal_orphans=False)
    assert [m.id for m in clean_hist] == [u1.id, a1.id]

    # Multi-step subcase: Step 1 executes a tool, Step 2 returns PROMPT_BLOCKED -> intermediate rows also stamped

    await gw.create_session("s_k1_tools", title="K1 Tools", chatbot_id="discord_test", channel_id="c2")
    step_calls = 0

    class ToolThenBlockedLLM(BaseLLM):
        config = cfg.gemini

        async def generate(self, **kwargs: Any) -> LLMResponse:
            nonlocal step_calls
            step_calls += 1
            if step_calls == 1:
                return LLMResponse(
                    content="",
                    thought="Checking file",
                    tool_calls=[ToolCallRequest(name="read_file", arguments={"file_name": "x.txt"}, tool_call_id="t1")],
                )
            return LLMResponse(content="", finish_reason="PROMPT_BLOCKED:SAFETY")

    ctx2 = KesokuContext(config=cfg, llm=ToolThenBlockedLLM())
    gw2 = Gateway(context=ctx2)
    tr2 = MagicMock()
    tr2.tool_registry.get_tools_list.return_value = []
    tr2.tool_context = ToolContext(gateway=gw2, session_id="s_k1_tools")

    async def fake_exec_tool(call: Any, tc_msg: Message, **kwargs: Any) -> Message:
        return Message(
            session_id="s_k1_tools",
            chatbot_id="discord_test",
            channel_id="c2",
            sender=call.name,
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="tool out",
            status=MessageStatus.RESPONDED,
            parent_id=tc_msg.id,
            metadata={"tool_name": call.name, "tool_call_id": call.tool_call_id},
        )

    tr2.execute_tool = fake_exec_tool
    exec2 = TurnExecutor("s_k1_tools", gw2, tr2, MagicMock(spec=TurnLogger), context=ctx2)
    u_multi = Message(
        session_id="s_k1_tools",
        chatbot_id="discord_test",
        channel_id="c2",
        sender="u",
        role=MessageRole.USER,
        content="Trigger tool then block",
        status=MessageStatus.PENDING_AGENT,
        timestamp=300.0,
    )
    await gw2.post(u_multi)
    await exec2.process_turn(u_multi, worker, str(tmp_path / "staging"))
    assert step_calls == 2
    raw2 = await gw2.db.get_session_messages("s_k1_tools")
    non_notice2 = [m for m in raw2 if m.sender != "Notification"]
    assert len(non_notice2) == 4  # USER + THOUGHT + TOOL_CALL + TOOL_RESULT
    assert all(m.metadata.get("prompt_blocked") == "SAFETY" for m in non_notice2)
    assert await build_history(gw2, "s_k1_tools", heal_orphans=False) == []


# origin: tests/test_round3_fixes.py K1
@pytest.mark.asyncio
async def test_k1_manual_compact_guardrail_g2_and_compact_all(tmp_path: Path) -> None:
    db_path = str(tmp_path / "k1_compact.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(tmp_path / "sessions")))
    cfg.agent.protect_front_turns = 3
    cfg.agent.protect_tail_turns = 6
    cfg.agent.watermark_chunk_min_turns = 1

    class SummaryLLM(BaseLLM):
        async def generate(self, **kwargs: Any) -> LLMResponse:
            return LLMResponse(content="<summary>Condensed safe summary.</summary>", total_tokens=20)

    ctx = KesokuContext(config=cfg, db=db, llm=SummaryLLM())
    gw = Gateway(context=ctx)
    bot = _DummyChatbot("discord_test", gw)
    sess = await gw.create_session("s_g2", title="G2", chatbot_id="discord_test", channel_id="chan_g2")

    # 2 clean turns + 1 prompt_blocked user turn at the end
    for idx in range(2):
        um = Message(
            session_id=sess.id,
            chatbot_id="discord_test",
            channel_id="chan_g2",
            sender="u",
            role=MessageRole.USER,
            content=f"Turn {idx} user text",
            status=MessageStatus.PROCESSED,
            timestamp=10.0 * idx + 1,
        )
        am = Message(
            session_id=sess.id,
            chatbot_id="discord_test",
            channel_id="chan_g2",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            content=f"Turn {idx} reply text",
            status=MessageStatus.DELIVERED,
            parent_id=um.id,
            timestamp=10.0 * idx + 2,
        )
        await gw.post(um)
        await gw.post(am)

    blocked_u = Message(
        session_id=sess.id,
        chatbot_id="discord_test",
        channel_id="chan_g2",
        sender="u",
        role=MessageRole.USER,
        content="Blocked user turn",
        status=MessageStatus.PROCESSED,
        timestamp=50.0,
        metadata={"prompt_blocked": "PROHIBITED_CONTENT"},
    )
    await gw.post(blocked_u)

    with patch("kesoku.context.get_llm", return_value=SummaryLLM()):
        reply = await bot.manual_compact_session_by_channel("chan_g2", "")

    assert reply.startswith("🔄 Compacted 2 turn(s) into ")
    assert "including recent turns" in reply
    # Guardrail G2: shared cfg.agent was NEVER mutated in place
    assert cfg.agent.protect_front_turns == 3
    assert cfg.agent.protect_tail_turns == 6

    # Also verify `/compact all` on an unblocked 2-turn session
    sess2 = await gw.create_session("s_all", title="All", chatbot_id="discord_test", channel_id="chan_all")
    for idx in range(2):
        um = Message(
            session_id=sess2.id,
            chatbot_id="discord_test",
            channel_id="chan_all",
            sender="u",
            role=MessageRole.USER,
            content=f"Clean {idx}",
            status=MessageStatus.PROCESSED,
            timestamp=10.0 * idx + 1,
        )
        am = Message(
            session_id=sess2.id,
            chatbot_id="discord_test",
            channel_id="chan_all",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            content=f"Reply {idx}",
            status=MessageStatus.DELIVERED,
            parent_id=um.id,
            timestamp=10.0 * idx + 2,
        )
        await gw.post(um)
        await gw.post(am)

    with patch("kesoku.context.get_llm", return_value=SummaryLLM()):
        reply_all = await bot.manual_compact_session_by_channel("chan_all", "all")
    assert reply_all.startswith("🔄 Compacted 2 turn(s) into ")
    assert "including recent turns" in reply_all
    assert cfg.agent.protect_front_turns == 3
    assert cfg.agent.protect_tail_turns == 6


# origin: tests/test_round3_fixes.py K1
@pytest.mark.asyncio
async def test_k1_second_consecutive_block_schedules_bg_compaction_and_swallows_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = str(tmp_path / "k1_consec.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(tmp_path / "sessions")))

    class AlwaysBlockedLLM(BaseLLM):
        async def generate(self, **kwargs: Any) -> LLMResponse:
            return LLMResponse(content="", finish_reason="PROMPT_BLOCKED:PROHIBITED_CONTENT")

    ctx = KesokuContext(config=cfg, db=db, llm=AlwaysBlockedLLM())
    gw = Gateway(context=ctx)
    await gw.create_session("s_consec", title="Consec", chatbot_id="discord_test", channel_id="c_consec")

    # Seed 1 normal turn so history is non-empty
    u0 = Message(
        session_id="s_consec",
        chatbot_id="discord_test",
        channel_id="c_consec",
        sender="u",
        role=MessageRole.USER,
        content="Hello",
        status=MessageStatus.PROCESSED,
        timestamp=1.0,
    )
    a0 = Message(
        session_id="s_consec",
        chatbot_id="discord_test",
        channel_id="c_consec",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Hi!",
        status=MessageStatus.DELIVERED,
        parent_id=u0.id,
        timestamp=2.0,
    )
    await gw.post(u0)
    await gw.post(a0)

    compact_calls: list[dict[str, Any]] = []

    async def fake_force_compact(
        self_comp: HistoryCompressor,
        session_id: str,
        history: list[Message],
        llm: BaseLLM,
        config: KesokuConfig,
        *,
        directive: str | None = None,
        staging_dir: str | None = None,
        protect_front_turns: int | None = None,
        protect_tail_turns: int | None = None,
    ) -> int:
        compact_calls.append(
            {
                "session_id": session_id,
                "protect_front_turns": protect_front_turns,
                "protect_tail_turns": protect_tail_turns,
            }
        )
        if len(compact_calls) >= 2:
            raise RuntimeError("compressor also blocked")
        return 1

    monkeypatch.setattr(HistoryCompressor, "force_compact_session", fake_force_compact)

    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    tool_runner.tool_context = ToolContext(gateway=gw, session_id="s_consec")
    executor = TurnExecutor("s_consec", gw, tool_runner, MagicMock(spec=TurnLogger), context=ctx)

    worker = MagicMock()
    type(worker).running = PropertyMock(return_value=True)
    worker.drain_queue_and_pivot = AsyncMock(side_effect=lambda m: m)
    worker.queue_empty.return_value = True

    # 1st blocked turn -> 0 background compactions
    b1 = Message(
        session_id="s_consec",
        chatbot_id="discord_test",
        channel_id="c_consec",
        sender="u",
        role=MessageRole.USER,
        content="Block 1",
        status=MessageStatus.PENDING_AGENT,
        timestamp=10.0,
    )
    await gw.post(b1)
    await executor.process_turn(b1, worker, str(tmp_path / "staging"))
    if executor._bg_compaction_tasks:
        await asyncio.gather(*list(executor._bg_compaction_tasks))
    assert len(compact_calls) == 0

    # 2nd consecutive blocked turn -> schedules background compaction once
    b2 = Message(
        session_id="s_consec",
        chatbot_id="discord_test",
        channel_id="c_consec",
        sender="u",
        role=MessageRole.USER,
        content="Block 2",
        status=MessageStatus.PENDING_AGENT,
        timestamp=20.0,
    )
    await gw.post(b2)
    await executor.process_turn(b2, worker, str(tmp_path / "staging"))
    if executor._bg_compaction_tasks:
        await asyncio.gather(*list(executor._bg_compaction_tasks))
    assert len(compact_calls) == 1
    assert compact_calls[0]["protect_tail_turns"] == 0
    assert compact_calls[0]["protect_front_turns"] == 0

    # 3rd blocked turn where compressor raises -> swallowed cleanly without raising
    b3 = Message(
        session_id="s_consec",
        chatbot_id="discord_test",
        channel_id="c_consec",
        sender="u",
        role=MessageRole.USER,
        content="Block 3",
        status=MessageStatus.PENDING_AGENT,
        timestamp=30.0,
    )
    await gw.post(b3)
    await executor.process_turn(b3, worker, str(tmp_path / "staging"))
    if executor._bg_compaction_tasks:
        await asyncio.gather(*list(executor._bg_compaction_tasks))
    assert len(compact_calls) == 2


# origin: tests/test_round3_fixes.py K9
@pytest.mark.asyncio
async def test_k9_gemini_request_timeout_http_options_and_non_stream_and_stream_timeouts() -> None:
    assert GeminiConfig().request_timeout_seconds == 300.0

    cfg = GeminiConfig(auth_mode="vertex", project_id="proj", location="us-central1", request_timeout_seconds=0.08)
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        llm = GeminiLLM(config=cfg)

        _, client_kwargs = mock_client_cls.call_args
        assert isinstance(client_kwargs.get("http_options"), types.HttpOptions)
        assert client_kwargs["http_options"].timeout == 80

        # Non-streaming wait_for timeout
        def slow_call(**kwargs: Any) -> Any:
            time.sleep(0.5)
            return MagicMock(parts=[], candidates=[])

        mock_client.models.generate_content.side_effect = slow_call
        with pytest.raises(TimeoutError):
            await llm._call_llm({"contents": [], "config": types.GenerateContentConfig()})

    # Streaming idle-chunk timeout in _drain_blocking_stream
    def stalled_stream() -> Any:
        yield "first_chunk"
        time.sleep(0.5)
        yield "never_reached"

    seen: list[Any] = []

    async def on_item(item: Any) -> None:
        seen.append(item)

    with pytest.raises(TimeoutError):
        await _drain_blocking_stream(stalled_stream, on_item, idle_timeout=0.08)
    assert seen == ["first_chunk"]


# origin: tests/test_ttft_async_tts_v015.py
@pytest.mark.asyncio
async def test_gemini_config_ttl_and_cache_rpcs_off_event_loop() -> None:
    """Verify `context_cache_ttl_seconds == 1200` and create/delete/update_cache_ttl run via `to_thread`."""
    cfg = GeminiConfig()
    assert cfg.context_cache_ttl_seconds == 1200
    assert resolve_gemini_cache_ttl(cfg) == 1200

    loop_tid = threading.get_ident()
    seen_threads: dict[str, int] = {}

    mock_client = MagicMock()

    def _fake_create(**kwargs: Any) -> Any:
        seen_threads["create"] = threading.get_ident()
        assert kwargs["config"].ttl == "1200s"
        c = MagicMock()
        c.name = "projects/p/locations/global/cachedContents/c1"
        return c

    def _fake_update(**kwargs: Any) -> Any:
        seen_threads["update"] = threading.get_ident()
        assert kwargs["name"] == "projects/p/locations/global/cachedContents/c1"
        assert kwargs["config"].ttl == "1200s"
        return MagicMock()

    def _fake_delete(**kwargs: Any) -> None:
        seen_threads["delete"] = threading.get_ident()
        assert kwargs["name"] == "projects/p/locations/global/cachedContents/c1"

    mock_client.caches.create.side_effect = _fake_create
    mock_client.caches.update.side_effect = _fake_update
    mock_client.caches.delete.side_effect = _fake_delete

    with patch("google.genai.Client", return_value=mock_client):
        llm = GeminiLLM(cfg)
        m = Message(
            session_id="s",
            chatbot_id="c",
            channel_id="ch",
            sender="u",
            role=MessageRole.USER,
            content="hello",
        )
        cname = await llm.create_cache([m], system_prompt="sys")
        assert cname == "projects/p/locations/global/cachedContents/c1"
        assert await llm.update_cache_ttl(cname) is True
        await llm.delete_cache(cname)

        assert seen_threads["create"] != loop_tid
        assert seen_threads["update"] != loop_tid
        assert seen_threads["delete"] != loop_tid

        # 404 on update_cache_ttl returns False cleanly
        mock_client.caches.update.side_effect = RuntimeError("404 NOT_FOUND: CachedContent expired")
        assert await llm.update_cache_ttl(cname, 1200) is False

        # 400 INVALID_ARGUMENT < min token count returns None quietly without warning spam
        mock_client.caches.create.side_effect = RuntimeError(
            "400 INVALID_ARGUMENT: CachedContent total token count 3900 is less than minimum 4096"
        )
        assert await llm.create_cache([m], system_prompt="sys") is None


# origin: tests/test_ttft_async_tts_v015.py
@pytest.mark.asyncio
async def test_gemini_streaming_and_sync_parity_with_cached_content() -> None:
    """Verify both `_call_llm` and `_call_llm_stream` pass `cached_content` and omit `system_instruction`/`tools`."""
    mock_client = MagicMock()

    sync_res = MagicMock()
    sync_res.parts = [MagicMock(text="Sync answer", thought=False, function_call=None)]
    sync_res.candidates = [MagicMock(finish_reason="STOP", finish_message=None, safety_ratings=None)]
    sync_res.usage_metadata = MagicMock(
        prompt_token_count=5000,
        candidates_token_count=20,
        total_token_count=5020,
        cached_content_token_count=4800,
    )
    sync_res.prompt_feedback = None
    mock_client.models.generate_content.return_value = sync_res

    part_thought = MagicMock(text="thinking...", thought=True, function_call=None, thought_signature=None)
    part_text = MagicMock(text="Stream answer", thought=False, function_call=None, thought_signature=None)
    chunk1 = MagicMock(
        candidates=[MagicMock(content=MagicMock(parts=[part_thought]), finish_reason=None, safety_ratings=None)],
        usage_metadata=None,
        prompt_feedback=None,
    )
    chunk2 = MagicMock(
        candidates=[MagicMock(content=MagicMock(parts=[part_text]), finish_reason="STOP", safety_ratings=None)],
        usage_metadata=sync_res.usage_metadata,
        prompt_feedback=None,
    )
    mock_client.models.generate_content_stream.return_value = iter([chunk1, chunk2])

    def dummy_tool(x: str) -> str:
        """Dummy tool."""
        return x

    tail_msg = Message(
        session_id="s",
        chatbot_id="c",
        channel_id="ch",
        sender="u",
        role=MessageRole.USER,
        content="Tail uncached prompt",
    )

    with patch("google.genai.Client", return_value=mock_client):
        llm = GeminiLLM(GeminiConfig(thinking_level="low"))
        cache_id = "projects/p/locations/global/cachedContents/999"

        r_sync = await llm.generate(
            system_prompt="Cached sys",
            history=[tail_msg],
            tools=[dummy_tool],
            cached_content=cache_id,
            stream=False,
        )
        streamed_deltas: list[str] = []

        async def _on_chunk(d: str, _acc: str) -> None:
            streamed_deltas.append(d)

        r_stream = await llm.generate(
            system_prompt="Cached sys",
            history=[tail_msg],
            tools=[dummy_tool],
            cached_content=cache_id,
            stream=True,
            on_stream_chunk=_on_chunk,
        )

    assert r_sync.content == "Sync answer"
    assert r_stream.content == "Stream answer"
    assert streamed_deltas == ["Stream answer"]
    assert r_sync.cached_tokens == r_stream.cached_tokens == 4800

    for call_obj in (
        mock_client.models.generate_content.call_args,
        mock_client.models.generate_content_stream.call_args,
    ):
        gen_cfg = call_obj.kwargs["config"]
        assert gen_cfg.cached_content == cache_id
        assert gen_cfg.system_instruction is None
        assert gen_cfg.tools is None
        assert gen_cfg.automatic_function_calling.disable is True
        sent_contents = call_obj.kwargs["contents"]
        assert len(sent_contents) == 1
        assert sent_contents[0].parts[0].text == "Tail uncached prompt"


# origin: tests/test_ttft_async_tts_v015.py
@pytest.mark.asyncio
async def test_post_turn_explicit_cache_lifecycle_4600_floor_rolling_refresh_and_ttl(tmp_path: Any) -> None:
    """Verify post-turn 4600 floor, rolling refresh (`create -> swap -> delete`), and sliding TTL renewal."""
    cfg = KesokuConfig(
        agent_working_dir=str(tmp_path),
        workspace=WorkspaceConfig(sessions_dir=str(tmp_path / "sessions"), db_path=str(tmp_path / "k.db")),
        gemini=GeminiConfig(
            model_name="gemini-3.8-flash",
            context_caching=True,
            context_caching_mode="explicit",
            context_caching_threshold=4096,  # Default -> enforces 4600 safety floor!
            context_cache_ttl_seconds=1200,
        ),
    )
    gw, _ = await _make_gateway(tmp_path, cfg=cfg)
    await gw.create_session("s_life", title="Lifecycle", system_prompt="System prompt")
    staging_dir = str(tmp_path / "sessions" / "s_life")
    os.makedirs(staging_dir, exist_ok=True)

    events: list[str] = []

    class GeminiLLM(BaseLLM):
        model_name = "gemini-3.8-flash"

        def __init__(self) -> None:
            self.config = cfg.gemini
            self.seq = 0
            self.generates: list[dict[str, Any]] = []
            self.ttl_renew_ok = True
            self.next_prompt_tokens = 5000
            self.next_cached_tokens = 4800
            self.worker_ref: Any = None

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:
            raise AssertionError("Blocking count_tokens() RPC must NEVER be called!")

        async def create_cache(
            self,
            contents: list[Message],
            system_prompt: str | None = None,
            tools: list[Any] | None = None,
            display_name: str | None = None,
            ttl_seconds: int | None = None,
        ) -> str | None:
            self.seq += 1
            cname = f"cache_v{self.seq}"
            events.append(f"create:{cname}:len={len(contents)}:ttl={ttl_seconds}")
            return cname

        async def delete_cache(self, cache_name: str) -> None:
            # Verify atomic swap ALREADY happened on worker BEFORE old cache is deleted!
            active_now = getattr(self.worker_ref, "active_cache_name", None)
            events.append(f"delete:{cache_name}:active_during_delete={active_now}")

        async def update_cache_ttl(self, cache_name: str, ttl_seconds: int = 1200) -> bool:
            events.append(f"ttl:{cache_name}:{ttl_seconds}")
            return self.ttl_renew_ok

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            cached_content: str | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            _ = (prompt, system_prompt, tools, kwargs)
            self.generates.append({"cached_content": cached_content, "hist_len": len(history or [])})
            return LLMResponse(
                content="Reply text.",
                prompt_tokens=self.next_prompt_tokens,
                cached_tokens=self.next_cached_tokens if cached_content else 0,
                candidates_tokens=20,
                total_tokens=self.next_prompt_tokens + 20,
            )

    llm = GeminiLLM()
    ctx = KesokuContext(config=cfg, db=gw.context.sync_db, llm=llm)
    gw.context = ctx
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    tool_runner.tool_context = MagicMock(transitioned_to_session=None)

    worker = SessionWorker("s_life", gw, context=ctx)
    llm.worker_ref = worker

    async def _run_turn(uid: str, text: str) -> None:
        umsg = Message(
            id=uid,
            session_id="s_life",
            chatbot_id="cli",
            channel_id="ch",
            sender="u",
            role=MessageRole.USER,
            content=text,
            status=MessageStatus.PENDING_AGENT,
        )
        await gw.post(umsg)
        ex = TurnExecutor("s_life", gw, tool_runner, context=ctx)
        worker._running = True
        with patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
            await ex.process_turn(current_msg=umsg, worker=worker, session_staging_dir=staging_dir)

    # 1. Short Turn 1 (~4200 est tokens: above 4096 but BELOW 4600 safety floor!) -> NO cache created!
    await _run_turn("u1", "a" * 16400)  # ~4100 est tokens
    assert events == []
    assert worker.active_cache_name is None

    # 2. Turn 2 pushes prefix >= 4600 est tokens -> R1 uncached, post-turn creates `cache_v1`!
    await _run_turn("u2", "b" * 4000)
    assert llm.generates[-1]["cached_content"] is None
    assert events == ["create:cache_v1:len=4:ttl=1200"]
    assert worker.active_cache_name == "cache_v1"

    # 3. Turn 3 hits `cache_v1` and has remaining TTL < 600s -> renews TTL via `update_cache_ttl`!
    entry_v1 = worker.explicit_caches["GeminiLLM:gemini-3.8-flash"]
    entry_v1.expires_at = time.time() + 300.0  # < 600s remaining
    events.clear()
    await _run_turn("u3", "short follow-up")
    assert llm.generates[-1]["cached_content"] == "cache_v1"
    assert llm.generates[-1]["hist_len"] == 1  # Only incremental tail `[u3]` sent!
    assert events == ["ttl:cache_v1:1200"]
    assert entry_v1.expires_at > time.time() + 1000.0

    # 4. Turn 4: `update_cache_ttl` hits 404 (`False`) -> falls back to creating fresh `cache_v2`!
    entry_v1.expires_at = time.time() + 200.0
    llm.ttl_renew_ok = False
    events.clear()
    await _run_turn("u4", "another short turn")
    assert events == ["ttl:cache_v1:1200", "create:cache_v2:len=8:ttl=1200"]
    assert worker.active_cache_name == "cache_v2"

    # 5. Turn 5: `uncached_suffix >= 8192` -> Rolling refresh: CREATE new `cache_v3` -> SWAP -> DELETE `cache_v2`!
    llm.next_prompt_tokens = 16000
    llm.next_cached_tokens = 5000  # 16000 - 5000 = 11000 >= max(8192, 1250)
    events.clear()
    await _run_turn("u5", "trigger rolling refresh")
    assert events == [
        "create:cache_v3:len=10:ttl=1200",
        "delete:cache_v2:active_during_delete=cache_v3",
    ]
    assert worker.active_cache_name == "cache_v3"
    discard_session_cache_state("s_life")


# origin: tests/test_ttft_async_tts_v015.py
@pytest.mark.asyncio
async def test_channel_model_override_isolation_404_stream_fallback_and_session_delete(tmp_path: Any) -> None:
    """Verify `#lite` model isolation, streaming 404 same-round fallback, and `/clear` + `delete_session` cleanup."""
    cfg = KesokuConfig(
        agent_working_dir=str(tmp_path),
        workspace=WorkspaceConfig(sessions_dir=str(tmp_path / "sessions"), db_path=str(tmp_path / "iso.db")),
        gemini=GeminiConfig(
            model_name="gemini-3.8-flash",
            context_caching=True,
            context_caching_mode="explicit",
            context_caching_threshold=50,
            stream=True,
        ),
        discord=DiscordConfig(
            enabled=True,
            chatbot_id="discord_iso",
            bot_token="tok",
            channels=[
                DiscordChannelOverride(
                    channels=["lite"],
                    llm="gemini",
                    model_name="gemini-3.5-flash-lite",
                    thinking_level="low",
                    stream=True,
                )
            ],
        ),
    )
    gw, _ = await _make_gateway(tmp_path, cfg=cfg)
    stream_events: list[str | None] = []

    async def _stream_handler(_cid: str, _sid: str, _tid: str, text: str | None) -> None:
        stream_events.append(text)

    gw.register_stream_handler("discord_iso", _stream_handler)
    await gw.create_session("s_iso", title="Iso", chatbot_id="discord_iso", channel_id="ch_main")
    staging_dir = str(tmp_path / "sessions" / "s_iso")
    os.makedirs(staging_dir, exist_ok=True)

    deleted_all: list[tuple[str, str]] = []
    seen_calls: list[tuple[str, str | None, bool]] = []
    fail_404_once = False

    class GeminiLLM(BaseLLM):
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.config = GeminiConfig(model_name=model_name, stream=True, context_caching_threshold=50)
            self.created = 0

        async def create_cache(self, contents: list[Message], **_kw: Any) -> str | None:
            _ = contents
            self.created += 1
            return f"cache_{self.model_name}_{self.created}"

        async def delete_cache(self, cache_name: str) -> None:
            deleted_all.append((self.model_name, cache_name))

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            cached_content: str | None = None,
            stream: bool = False,
            on_stream_chunk: Any = None,
        ) -> LLMResponse:
            _ = (prompt, system_prompt, history, tools)
            nonlocal fail_404_once
            seen_calls.append((self.model_name, cached_content, stream))
            if fail_404_once and cached_content is not None:
                fail_404_once = False
                if on_stream_chunk is not None:
                    await on_stream_chunk("partial", "partial")
                raise RuntimeError("404 NOT_FOUND: Requested entity CachedContent was not found")
            if on_stream_chunk is not None:
                await on_stream_chunk("Hello ", "Hello ")
                await on_stream_chunk("world!", "Hello world!")
            return LLMResponse(content="Hello world!", prompt_tokens=200, cached_tokens=150 if cached_content else 0)

    llms = {
        "gemini-3.8-flash": GeminiLLM("gemini-3.8-flash"),
        "gemini-3.5-flash-lite": GeminiLLM("gemini-3.5-flash-lite"),
    }
    ctx = KesokuContext(config=cfg, db=gw.context.sync_db)
    ctx.get_llm = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda model_name=None, **_kw: llms[model_name or "gemini-3.8-flash"]
    )
    gw.context = ctx

    agent = Agent(gw, context=ctx)
    worker = SessionWorker("s_iso", gw, context=ctx)
    agent.workers["s_iso"] = worker
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = ["t1"]
    tool_runner.tool_context = MagicMock(transitioned_to_session=None)

    async def _turn(mid: str, ch_name: str, text: str) -> None:
        m = Message(
            id=mid,
            session_id="s_iso",
            chatbot_id="discord_iso",
            channel_id=ch_name,
            sender="u",
            role=MessageRole.USER,
            content=text,
            status=MessageStatus.PENDING_AGENT,
            metadata={"channel_name": ch_name},
        )
        await gw.post(m)
        ex = TurnExecutor("s_iso", gw, tool_runner, context=ctx)
        worker._running = True
        with patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
            await ex.process_turn(current_msg=m, worker=worker, session_staging_dir=staging_dir)

    # Turn 1 on main (`gemini-3.8-flash`) -> creates `cache_gemini-3.8-flash_1` post-turn
    await _turn("m1", "main", "Long enough initial prompt " * 20)
    assert "GeminiLLM:gemini-3.8-flash" in worker.explicit_caches

    # Turn 2 on `#lite` (`gemini-3.5-flash-lite`) -> MUST NOT reuse `cache_gemini-3.8-flash_1` OR delete it!
    await _turn("m2", "lite", "Quick question on lite channel " * 10)
    assert seen_calls[-1] == ("gemini-3.5-flash-lite", None, True)
    assert deleted_all == []
    assert "GeminiLLM:gemini-3.8-flash" in worker.explicit_caches
    assert "GeminiLLM:gemini-3.5-flash-lite" in worker.explicit_caches

    # Turn 3 back on main (`gemini-3.8-flash`) with simulated 404 on stream -> discards preview & retries same round!
    fail_404_once = True
    stream_events.clear()
    await _turn("m3", "main", "Back on main channel")
    assert seen_calls[-2] == ("gemini-3.8-flash", "cache_gemini-3.8-flash_1", True)
    assert seen_calls[-1] == ("gemini-3.8-flash", None, True)
    assert None in stream_events  # Partial preview was discarded cleanly before retry!

    # `Gateway.delete_session` cleans up ALL active caches across all models for `s_iso`!
    await gw.delete_session("s_iso")
    deleted_names = {c for _, c in deleted_all}
    assert "cache_gemini-3.8-flash_2" in deleted_names
    assert "cache_gemini-3.5-flash-lite_1" in deleted_names


# origin: tests/test_ttft_async_tts_v015.py
@pytest.mark.asyncio
async def test_attachment_webp_lru_hit_mtime_invalidation_and_discord_parallel_prewarm(tmp_path: Any) -> None:
    """Verify 0-recompress LRU hit, mtime invalidation, maxsize=64 eviction, and Discord parallel download + prewarm."""
    with _IMAGE_ATTACHMENT_LRU_LOCK:
        _IMAGE_ATTACHMENT_LRU.clear()

    img1 = str(tmp_path / "photo_a.png")
    with open(img1, "wb") as fh:
        fh.write(_make_png_bytes(1400, 1000))

    with patch("kesoku.agent.llm._resize_and_compress_image", wraps=__import__(
        "kesoku.agent.llm", fromlist=["_resize_and_compress_image"]
    )._resize_and_compress_image) as spy_resize:
        b1 = _load_attachments([{"path": img1, "mime_type": "image/png"}])
        b2 = _load_attachments([{"path": img1, "mime_type": "image/png"}])
        assert spy_resize.call_count == 1  # Second round is an instant RAM LRU hit!
        assert b1[0].data == b2[0].data

        # Changing mtime invalidates the LRU entry
        future_ns = os.stat(img1).st_mtime_ns + 10_000_000
        os.utime(img1, ns=(future_ns, future_ns))
        _load_attachments([{"path": img1, "mime_type": "image/png"}])
        assert spy_resize.call_count == 2

    # Verify maxsize=64 eviction bound
    small_png = _make_png_bytes(64, 64)
    for idx in range(70):
        p = str(tmp_path / f"evict_{idx}.png")
        with open(p, "wb") as fh:
            fh.write(small_png)
        _load_attachments([{"path": p, "mime_type": "image/png"}])
    with _IMAGE_ATTACHMENT_LRU_LOCK:
        assert len(_IMAGE_ATTACHMENT_LRU) == _IMAGE_ATTACHMENT_LRU_MAXSIZE

    # Verify Discord parallel download (semaphore=6) + `asyncio.to_thread(prewarm_image_attachments, ...)`
    gw, _ = await _make_gateway(tmp_path)
    session = await gw.create_session("s_att", title="Att", chatbot_id="discord_v015", channel_id="44")
    bot = _make_discord_bot(gw)

    active_downloads = 0
    peak_downloads = 0
    save_starts: list[float] = []
    save_ends: list[float] = []
    png_payload = _make_png_bytes(1280, 960, color=(90, 30, 120))

    def _make_mock_att(idx: int, fname: str) -> Any:
        att = MagicMock()
        att.id = 1000 + idx
        att.filename = fname
        att.content_type = "image/png"

        async def _save(target_path: str) -> None:
            nonlocal active_downloads, peak_downloads
            save_starts.append(time.perf_counter())
            active_downloads += 1
            peak_downloads = max(peak_downloads, active_downloads)
            try:
                await asyncio.sleep(0.08)
                with open(target_path, "wb") as fh:
                    fh.write(png_payload)
            finally:
                active_downloads -= 1
                save_ends.append(time.perf_counter())

        att.save = AsyncMock(side_effect=_save)
        return att

    # 8 attachments including duplicate clipboard `"image.png"` names!
    raw_msg = MagicMock()
    raw_msg.attachments = [
        _make_mock_att(i, "image.png" if i < 3 else f"shot_{i}.png") for i in range(8)
    ]

    loop_tid = threading.get_ident()
    prewarm_tids: list[int] = []
    orig_prewarm = prewarm_image_attachments

    def _spy_prewarm(paths: Any) -> int:
        prewarm_tids.append(threading.get_ident())
        return orig_prewarm(paths)

    dto = MagicMock()
    with patch("kesoku.agent.llm.prewarm_image_attachments", side_effect=_spy_prewarm):
        saved_atts = await bot.process_attachments_hook(session, dto, raw_msg)
    download_elapsed = max(save_ends) - min(save_starts)

    assert len(saved_atts) == 8
    assert len({a.path for a in saved_atts}) == 8  # Zero filename collision overwrite!
    assert 2 <= peak_downloads <= 6  # Concurrent and bounded by semaphore(6)!
    assert download_elapsed < 0.45  # 8 * 0.08s = 0.64s serial -> ~0.16s parallel (2 waves of 6+2)!
    assert prewarm_tids and prewarm_tids[0] != loop_tid  # Ran off the asyncio event loop!

    # Now `history_to_turns` hits warm LRU with ZERO `_resize_and_compress_image` calls on the event loop!
    u_msg = Message(
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="44",
        sender="u",
        role=MessageRole.USER,
        content="Check all 8 images",
        metadata={"attachments": [a.model_dump() for a in saved_atts]},
    )
    with patch("kesoku.agent.llm._resize_and_compress_image") as mock_no_resize:
        turns, _ = history_to_turns([u_msg])
        mock_no_resize.assert_not_called()
    assert len(turns[0].blocks) == 9  # 1 TextBlock + 8 ImageBlocks


# origin: tests/test_ttft_async_tts_v015.py
@pytest.mark.asyncio
async def test_post_compaction_cache_rebuild_and_clear_session_cleanup(tmp_path: Any) -> None:
    """Verify post-turn compaction deletes old cache & immediately rebuilds over compacted prefix, and `/clear` cleans up."""  # noqa: E501
    from kesoku.db import SummaryNode

    cfg = KesokuConfig(
        agent_working_dir=str(tmp_path),
        workspace=WorkspaceConfig(sessions_dir=str(tmp_path / "sessions"), db_path=str(tmp_path / "reb.db")),
        gemini=GeminiConfig(
            model_name="gemini-3.8-flash",
            context_caching=True,
            context_caching_mode="explicit",
            context_caching_threshold=30,
        ),
    )
    gw, db = await _make_gateway(tmp_path, cfg=cfg)
    bot = _make_discord_bot(gw, chatbot_id="discord_reb")
    bot.bot.close = AsyncMock()  # type: ignore[method-assign]
    session = await gw.create_session("s_reb", title="Rebuild", chatbot_id="discord_reb", channel_id="88")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    ops: list[str] = []

    class GeminiLLM(BaseLLM):
        model_name = "gemini-3.8-flash"

        def __init__(self) -> None:
            self.config = cfg.gemini
            self.n = 0

        async def create_cache(self, contents: list[Message], **_kw: Any) -> str | None:
            self.n += 1
            name = f"reb_cache_{self.n}"
            has_scaffold = any("[Note: This conversation uses custom turn-based" in (m.content or "") for m in contents)
            ops.append(f"create:{name}:scaffold={has_scaffold}")
            return name

        async def delete_cache(self, cache_name: str) -> None:
            ops.append(f"delete:{cache_name}")

        async def generate(self, **_kw: Any) -> LLMResponse:
            return LLMResponse(content="Long assistant reply " * 10, total_tokens=50)

    llm = GeminiLLM()
    ctx = KesokuContext(config=cfg, db=gw.context.sync_db, llm=llm)
    gw.context = ctx
    agent = Agent(gw, context=ctx)
    worker = SessionWorker("s_reb", gw, context=ctx)
    agent.workers["s_reb"] = worker
    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    tool_runner.tool_context = MagicMock(transitioned_to_session=None)

    # Turn 1 creates `reb_cache_1` (no scaffold yet)
    u1 = Message(
        id="u_r1",
        session_id="s_reb",
        chatbot_id="discord_reb",
        channel_id="88",
        sender="u",
        role=MessageRole.USER,
        content="Initial user turn " * 15,
        status=MessageStatus.PENDING_AGENT,
    )
    await gw.post(u1)
    ex1 = TurnExecutor("s_reb", gw, tool_runner, context=ctx)
    worker._running = True
    with patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
        await ex1.process_turn(current_msg=u1, worker=worker, session_staging_dir=staging_dir)
    assert ops == ["create:reb_cache_1:scaffold=False"]

    # Turn 2 triggers post-turn compaction -> deletes `reb_cache_1` and IMMEDIATELY creates `reb_cache_2` over scaffold!
    async def _compact_t1(session_id: str, history: list[Message], *_a: Any, **_kw: Any) -> bool:
        node = SummaryNode(
            id="sum-node-reb",
            session_id=session_id,
            level=0,
            summary="Dense summary of Turn 1 " * 10,
            start_timestamp=time.time() - 50,
            end_timestamp=time.time() - 10,
            token_count=80,
            source_token_count=200,
            parent_id=None,
        )
        await db.insert_summary_node(node)
        await db.update_messages_summary_node([m.id for m in history[:2]], "sum-node-reb")
        return True

    u2 = Message(
        id="u_r2",
        session_id="s_reb",
        chatbot_id="discord_reb",
        channel_id="88",
        sender="u",
        role=MessageRole.USER,
        content="Second user turn " * 15,
        status=MessageStatus.PENDING_AGENT,
    )
    await gw.post(u2)
    ex2 = TurnExecutor("s_reb", gw, tool_runner, context=ctx)
    worker._running = True
    ops.clear()
    with patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", side_effect=_compact_t1):
        await ex2.process_turn(current_msg=u2, worker=worker, session_staging_dir=staging_dir)

    assert ops == ["delete:reb_cache_1", "create:reb_cache_2:scaffold=True"]
    assert worker.active_cache_name == "reb_cache_2"

    # `/clear` (`clear_session_by_channel`) stops worker and deletes `reb_cache_2`!
    ops.clear()
    await bot.clear_session_by_channel("88")
    assert ops == ["delete:reb_cache_2"]
    discard_session_cache_state("s_reb")


def _gemini_chunk(parts: list[Any], usage: Any = None, finish: Any = None) -> Any:
    """Build a streamed Gemini chunk carrying `parts` (test helper)."""
    return types.GenerateContentResponse.model_construct(
        candidates=[
            types.Candidate.model_construct(
                content=types.Content.model_construct(role="model", parts=parts),
                finish_reason=finish,
            )
        ],
        usage_metadata=usage,
    )


def test_summarize_gemini_parts_run_length_shapes_and_limit() -> None:
    """`_summarize_gemini_parts` merges consecutive thought/text runs and folds overflow into a tail entry."""
    sig = b"\x0a\x0b"
    th1 = types.Part.from_text(text="abc")
    th1.thought = True
    th2 = types.Part.from_text(text="de")
    th2.thought = True
    tx1 = types.Part.from_text(text="Hello")
    tx2 = types.Part.from_text(text=" there")
    tx_sig = types.Part.from_text(text="!!")
    tx_sig.thought_signature = sig
    empty_sig = types.Part.from_text(text="")
    empty_sig.thought_signature = sig
    fc = types.Part.from_function_call(name="lookup", args={})
    fc_sig = types.Part.from_function_call(name="fetch", args={})
    fc_sig.thought_signature = sig

    shapes = _summarize_gemini_parts([th1, th2, tx1, tx2, tx_sig, empty_sig, fc, fc_sig])
    assert shapes == ["thought:5", "text:11", "text+sig:2", "sig", "fc:lookup", "fc+sig:fetch"]
    assert _summarize_gemini_parts(None) == []

    many = [types.Part.from_function_call(name=f"f{i}", args={}) for i in range(20)]
    limited = _summarize_gemini_parts(many, limit=16)
    assert len(limited) == 16
    assert limited[:15] == [f"fc:f{i}" for i in range(15)]
    assert limited[-1] == "+5"


@pytest.mark.asyncio
async def test_gemini_parse_records_part_shapes_and_thoughts_tokens() -> None:
    """Non-streaming parse exposes `part_shapes` and `usage_metadata.thoughts_token_count` without changing content."""
    sig = b"\x01\x02"
    th = types.Part.from_text(text="Reasoning about the request.")
    th.thought = True
    tx = types.Part.from_text(text="Final answer.")
    tail = types.Part.from_text(text="")
    tail.thought_signature = sig
    raw = types.GenerateContentResponse.model_construct(
        candidates=[
            types.Candidate.model_construct(
                content=types.Content.model_construct(role="model", parts=[th, tx, tail]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=20, candidates_token_count=8, total_token_count=65, thoughts_token_count=37
        ),
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = raw
    with patch("google.genai.Client", return_value=mock_client):
        res = await GeminiLLM().generate(prompt="Hi", stream=False)

    assert res.content == "Final answer."
    assert res.thought == "Reasoning about the request."
    assert res.thoughts_tokens == 37
    assert res.part_shapes == [f"thought:{len('Reasoning about the request.')}", f"text:{len('Final answer.')}", "sig"]


@pytest.mark.asyncio
async def test_gemini_stream_unflagged_leading_text_leak_fixture() -> None:
    """Regression fixture: un-flagged planning text is passed through by the provider layer.

    Chunks `[text(plan)] [text(a1)] [text(a2)] [text("")+sig]` with `thoughts_token_count=0` reproduce the
    observed symptom: the plan is streamed live and glued to the answer with no thought part. The structural
    telemetry records the shape so such responses are diagnosable; the defence lives in the reply channel.
    """
    plan = "Plan: first restate the question, then answer briefly."
    a1 = "The answer is"
    a2 = " forty-two."
    tail = types.Part.from_text(text="")
    tail.thought_signature = b"\x05\x06"
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=50, candidates_token_count=30, total_token_count=80, thoughts_token_count=0
    )
    chunks = [
        _gemini_chunk([types.Part.from_text(text=plan)]),
        _gemini_chunk([types.Part.from_text(text=a1)]),
        _gemini_chunk([types.Part.from_text(text=a2)]),
        _gemini_chunk([tail], usage=usage, finish=types.FinishReason.STOP),
    ]
    mock_client = MagicMock()
    mock_client.models.generate_content_stream.side_effect = lambda **_: iter(chunks)

    seen: list[str] = []

    async def on_chunk(_delta: str, accumulated: str) -> None:
        seen.append(accumulated)

    with patch("google.genai.Client", return_value=mock_client):
        res = await GeminiLLM(GeminiConfig(stream=True)).generate(prompt="Hi", stream=True, on_stream_chunk=on_chunk)

    # Documented provider-level pass-through: the plan is visible text, streamed and glued to the answer.
    assert seen[0] == plan
    assert res.content == plan + a1 + a2
    assert res.thought is None
    assert res.thoughts_tokens == 0
    assert res.part_shapes == [f"text:{len(plan + a1 + a2)}", "sig"]


def test_history_to_turns_replays_marker_only_for_contracted_rows() -> None:
    """Only assistant TEXT rows with `reply_channel == "found"` are replayed with the marker line prepended."""
    from kesoku.constants import REPLY_MARKER

    def _m(role: str, content: str, mtype: str = "text", meta: dict[str, Any] | None = None) -> Message:
        return Message(
            session_id="s",
            chatbot_id="cli",
            channel_id="c",
            sender="x",
            role=role,
            type=mtype,
            content=content,
            metadata=meta or {},
        )

    history = [
        _m("user", "first question"),
        _m("assistant", "legacy reply"),
        _m("user", "second question"),
        _m("assistant", "missing-marker reply", meta={"reply_channel": "missing"}),
        _m("user", "third question"),
        _m("assistant", "contracted reply", meta={"reply_channel": "found"}),
        _m("user", "fourth question"),
    ]
    turns, _ = history_to_turns(history, None, None)
    assistant_texts = [
        b.text for t in turns if t.role == MessageRole.ASSISTANT for b in t.blocks if getattr(b, "type", "") == "text"
    ]
    assert assistant_texts == ["legacy reply", "missing-marker reply", f"{REPLY_MARKER}\ncontracted reply"]

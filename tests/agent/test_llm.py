"""Unit tests for Kesoku LLM module and Anthropic Claude support."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from kesoku.agent.llm import (
    ClaudeLLM,
    GeminiLLM,
    MockLLM,
    function_to_anthropic_tool,
    get_llm,
    history_to_turns,
)
from kesoku.config import ClaudeConfig, GeminiConfig, KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message


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

        g_invalid = get_llm(provider="gemini", config=cfg, thinking_level="xhigh")
        assert isinstance(g_invalid, GeminiLLM)
        assert g_invalid.config.thinking_level == "high"
        mock_warn.assert_called_once()
        assert "Ignoring unsupported thinking_level 'xhigh' for Gemini" in mock_warn.call_args[0][0]

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



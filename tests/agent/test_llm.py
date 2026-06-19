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
    history_to_turns,
)
from kesoku.constants import MessageRole, MessageType
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


def test_get_llm_with_lcm_override() -> None:
    """Verify get_llm applies use_lcm overrides for provider and model configuration."""
    from kesoku.config import KesokuConfig
    cfg = KesokuConfig()
    cfg.agent.llm = "gemini"
    cfg.agent.lcm_llm = "claude"
    cfg.gemini.model_name = "gemini-3.5-pro"
    cfg.gemini.lcm_model_name = "gemini-2.0-flash-lite"
    cfg.claude.model_name = "claude-3-5-sonnet"
    cfg.claude.lcm_model_name = "claude-3-5-haiku"

    # Case 1: no use_lcm (resolves to gemini model_name)
    with patch("kesoku.agent.llm.GeminiLLM") as mock_gemini:
        get_llm(config=cfg)
        mock_gemini.assert_called_once()
        called_config = mock_gemini.call_args[1]["config"]
        assert called_config.model_name == "gemini-3.5-pro"

    # Case 2: use_lcm=True, which overrides provider to claude and uses claude's lcm_model_name
    with patch("kesoku.agent.llm.ClaudeLLM") as mock_claude:
        get_llm(config=cfg, use_lcm=True)
        mock_claude.assert_called_once()
        called_config = mock_claude.call_args[1]["config"]
        assert called_config.model_name == "claude-3-5-haiku"

    # Case 3: use_lcm=True, but provider explicitly overridden to gemini (uses gemini's lcm_model_name)
    with patch("kesoku.agent.llm.GeminiLLM") as mock_gemini:
        get_llm(provider="gemini", config=cfg, use_lcm=True)
        mock_gemini.assert_called_once()
        called_config = mock_gemini.call_args[1]["config"]
        assert called_config.model_name == "gemini-2.0-flash-lite"



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




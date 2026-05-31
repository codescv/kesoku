"""Unit tests for Kesoku WeChat chatbot adapter."""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kesoku.config import KesokuConfig, WechatConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message, Session
from kesoku.gateway.chatbot.wechat.adapter import (
    WechatChatbot,
    _guess_chat_type,
    _looks_like_chatty_line_for_weixin,
    _normalize_markdown_blocks,
    _split_text_for_weixin_delivery,
    _wrap_copy_friendly_lines_for_weixin,
)
from kesoku.gateway.gateway import Gateway


def mock_http_method(
    mock_method: MagicMock,
    response_json: str = '{"ret": 0, "errcode": 0}',
    ok: bool = True,
) -> None:
    """Configure a MagicMock to act as an aiohttp async context manager returning a mock response."""
    mock_response = MagicMock()
    mock_response.ok = ok
    mock_response.status = 200 if ok else 400
    mock_response.text = AsyncMock(return_value=response_json)
    mock_response.read = AsyncMock(return_value=response_json.encode("utf-8"))

    mock_method.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_method.return_value.__aexit__ = AsyncMock()


@pytest.fixture
def mock_config(tmp_path) -> KesokuConfig:
    """Provide a mock Kesoku configuration with WeChat enabled."""
    cfg = KesokuConfig()
    cfg.workspace.sessions_dir = str(tmp_path / "sessions")
    cfg.workspace.db_path = str(tmp_path / "kesoku.db")
    cfg.workspace.skills_dir = str(tmp_path / "skills")
    cfg.wechat = WechatConfig(
        enabled=True,
        chatbot_id="wechat_test",
        account_id="test_bot_id",
        token="test_token_123",
        base_url="https://test.ilink.com",
    )
    return cfg


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Provide a mock Gateway instance."""
    gw = MagicMock(spec=Gateway)
    db = AsyncMock()
    gw.db = db
    db.get_session_by_channel = AsyncMock(return_value=None)
    db.update_session_updated_at = AsyncMock()
    db.update_message_status = AsyncMock()
    gw.create_session = AsyncMock(return_value=Session(id="sess123", title="Test WeChat Session"))
    gw.post = AsyncMock()
    gw.agent = MagicMock()
    return gw


def test_guess_chat_type() -> None:
    """Test chat type guesser logic."""
    # DM cases
    msg1 = {"from_user_id": "user123", "msg_type": 1}
    assert _guess_chat_type(msg1, "bot123") == ("dm", "user123")

    # Group cases
    msg2 = {"room_id": "room_xyz", "from_user_id": "user123"}
    assert _guess_chat_type(msg2, "bot123") == ("group", "room_xyz")

    msg3 = {"to_user_id": "room_xyz", "msg_type": 1}
    assert _guess_chat_type(msg3, "bot123") == ("group", "room_xyz")


def test_normalize_markdown_blocks() -> None:
    """Test normalizing markdown headers for WeChat."""
    content = "# First Heading\nSome text\n## Second Heading\nMore text"
    normalized = _normalize_markdown_blocks(content)
    assert "【First Heading】" in normalized
    assert "**Second Heading**" in normalized


def test_wrap_copy_friendly_lines_for_weixin() -> None:
    """Test copy friendly line wrapping for WeChat."""
    long_line = "hello " * 30
    wrapped = _wrap_copy_friendly_lines_for_weixin(long_line)
    assert len(wrapped.splitlines()) >= 2


def test_looks_like_chatty_line_for_weixin() -> None:
    """Test checking if a line looks like chat utterance."""
    assert _looks_like_chatty_line_for_weixin("Hello how are you?") is True
    assert _looks_like_chatty_line_for_weixin("> Quoted text") is False
    assert _looks_like_chatty_line_for_weixin("- List item") is False


def test_split_text_for_weixin_delivery() -> None:
    """Test text splitting/chunking under limit."""
    short_text = "Hello world"
    chunks = _split_text_for_weixin_delivery(short_text, max_length=20)
    assert chunks == ["Hello world"]

    long_text = "a" * 50
    chunks2 = _split_text_for_weixin_delivery(long_text, max_length=20)
    assert len(chunks2) == 3
    assert chunks2[0] == "a" * 20


@pytest.mark.asyncio
async def test_init_disabled_raises_value_error() -> None:
    """Test WeChat initialization when disabled raises ValueError."""
    cfg = KesokuConfig()
    cfg.wechat = WechatConfig(enabled=False)
    gw = MagicMock(spec=Gateway)

    with patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=cfg):
        with pytest.raises(ValueError, match="WeChat chatbot is disabled"):
            WechatChatbot(chatbot_id="wechat", gateway=gw)


@pytest.mark.asyncio
async def test_init_missing_params_raises_value_error() -> None:
    """Test WeChat initialization with missing credentials raises ValueError."""
    cfg = KesokuConfig()
    cfg.wechat = WechatConfig(enabled=True, account_id="", token="")
    gw = MagicMock(spec=Gateway)

    with patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=cfg):
        with pytest.raises(ValueError, match="account_id or token is missing"):
            WechatChatbot(chatbot_id="wechat", gateway=gw)


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_wechat_chatbot_process_message(
    mock_session_cls: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test processing incoming WeChat text messages."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    # Mock ClientSession POST / GET
    mock_session.post = MagicMock()
    mock_http_method(mock_session.post)
    mock_session.get = MagicMock()
    mock_http_method(mock_session.get)

    with patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=mock_config):
        bot = WechatChatbot(chatbot_id="wechat_test", gateway=mock_gateway)
        bot._poll_session = mock_session
        bot._send_session = mock_session

        # Simulate incoming MESSAGE payload
        inbound_payload = {
            "from_user_id": "user_alice",
            "to_user_id": "test_bot_id",
            "message_id": "msg_001",
            "context_token": "ctx_tok_999",
            "item_list": [{"type": 1, "text_item": {"text": "Hello from WeChat!"}}],
        }

        # Inbound processing
        await bot._process_message(inbound_payload)

        # Verify context token was saved
        assert bot._token_store.get("test_bot_id", "user_alice") == "ctx_tok_999"

        # Verify session retrieval
        mock_gateway.db.get_session_by_channel.assert_called_once_with("wechat_test", "user_alice")

        # Verify new session creation
        mock_gateway.create_session.assert_called_once()
        create_args = mock_gateway.create_session.call_args[1]
        assert "WeChat Session" in create_args["title"]
        assert "WeChat Platforms Instructions" in create_args["custom_prompt"]

        # Verify gateway.post
        mock_gateway.post.assert_called_once()
        posted = mock_gateway.post.call_args[0][0]
        assert posted.role == MessageRole.USER
        assert posted.content == "Hello from WeChat!"
        assert posted.sender == "user_alice"
        assert posted.status == MessageStatus.PENDING_AGENT


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_wechat_chatbot_process_message_with_sys_prompt_file(
    mock_session_cls: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test custom configurable system prompt file inclusion."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.post = MagicMock()
    mock_http_method(mock_session.post)
    mock_session.get = MagicMock()
    mock_http_method(mock_session.get)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp_sys_prompt:
        tmp_sys_prompt.write("Custom wechat prompt instructions go here.")
        tmp_sys_prompt_path = tmp_sys_prompt.name

    try:
        mock_config.wechat.sys_prompt_file = tmp_sys_prompt_path

        with patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=mock_config):
            bot = WechatChatbot(chatbot_id="wechat_test", gateway=mock_gateway)
            bot._poll_session = mock_session
            bot._send_session = mock_session

            inbound_payload = {
                "from_user_id": "user_alice",
                "to_user_id": "test_bot_id",
                "message_id": "msg_001",
                "item_list": [{"type": 1, "text_item": {"text": "Hello from WeChat!"}}],
            }

            await bot._process_message(inbound_payload)

            # Verify new session custom prompt includes the custom system prompt file instructions
            mock_gateway.create_session.assert_called_once()
            create_args = mock_gateway.create_session.call_args[1]
            custom_prompt = create_args["custom_prompt"]
            assert "Custom wechat prompt instructions go here." in custom_prompt
    finally:
        if os.path.exists(tmp_sys_prompt_path):
            os.unlink(tmp_sys_prompt_path)


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_wechat_chatbot_send_text(
    mock_session_cls: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test sending assistant final response to WeChat."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.post = MagicMock()
    mock_http_method(mock_session.post, response_json='{"ret": 0, "errcode": 0}')

    with patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=mock_config):
        bot = WechatChatbot(chatbot_id="wechat_test", gateway=mock_gateway)
        bot._poll_session = mock_session
        bot._send_session = mock_session

        # Outbound message from model
        outbound_msg = Message(
            id="msg_out_999",
            session_id="sess123",
            chatbot_id="wechat_test",
            channel_id="user_alice",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Hello from assistant!",
        )

        await bot.handle_message(outbound_msg)

        # Verify EP_SEND_MESSAGE post call was made
        mock_session.post.assert_called_once()
        post_args = mock_session.post.call_args
        url = post_args[0][0]
        assert "ilink/bot/sendmessage" in url
        body = json.loads(post_args[1]["data"])
        assert body["msg"]["to_user_id"] == "user_alice"
        assert body["msg"]["item_list"][0]["text_item"]["text"] == "Hello from assistant!"

        # Verify status was updated to DELIVERED
        mock_gateway.db.update_message_status.assert_called_once_with("msg_out_999", MessageStatus.DELIVERED)


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_wechat_chatbot_slash_command_clear(
    mock_session_cls: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that /clear or /reset command deletes session and history."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.post = MagicMock()
    mock_http_method(mock_session.post, response_json='{"ret": 0, "errcode": 0}')

    with patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=mock_config):
        bot = WechatChatbot(chatbot_id="wechat_test", gateway=mock_gateway)
        bot._poll_session = mock_session
        bot._send_session = mock_session

        # Mock existing session
        mock_gateway.db.get_session_by_channel.return_value = Session(id="sess123", title="Active Session")

        inbound_payload = {
            "from_user_id": "user_alice",
            "to_user_id": "test_bot_id",
            "message_id": "msg_001",
            "item_list": [{"type": 1, "text_item": {"text": "/clear"}}],
        }

        # Handle message containing command
        await bot._process_message(inbound_payload)

        # Verify session deletion was triggered via gateway
        mock_gateway.delete_session.assert_called_once_with("sess123")

        # Verify confirmation message was sent
        mock_session.post.assert_called_once()
        body = json.loads(mock_session.post.call_args[1]["data"])
        assert "Session successfully cleared" in body["msg"]["item_list"][0]["text_item"]["text"]


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_wechat_chatbot_slash_command_status(
    mock_session_cls: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that /status command returns session metrics."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.post = MagicMock()
    mock_http_method(mock_session.post, response_json='{"ret": 0, "errcode": 0}')

    with patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=mock_config):
        bot = WechatChatbot(chatbot_id="wechat_test", gateway=mock_gateway)
        bot._poll_session = mock_session
        bot._send_session = mock_session

        mock_gateway.db.get_session_by_channel.return_value = Session(id="sess123", title="Active Session")

        # Mock history containing metrics
        mock_history = [
            Message(
                id="msg1",
                session_id="sess123",
                chatbot_id="wechat_test",
                channel_id="user_alice",
                sender="Alice",
                role=MessageRole.USER,
                type=MessageType.TEXT,
                content="hi",
            ),
            Message(
                id="msg2",
                session_id="sess123",
                chatbot_id="wechat_test",
                channel_id="user_alice",
                sender="Kesoku",
                role=MessageRole.ASSISTANT,
                type=MessageType.TEXT,
                content="hello",
                metadata={
                    "turn_metrics": {
                        "session_turns": 1,
                        "context_tokens": 3000,
                        "turn_tool_calls": 2,
                        "turn_tokens": 150,
                        "turn_time": 1.5,
                    }
                },
            ),
        ]
        mock_gateway.db.get_session_history.return_value = mock_history

        inbound_payload = {
            "from_user_id": "user_alice",
            "to_user_id": "test_bot_id",
            "message_id": "msg_001",
            "item_list": [{"type": 1, "text_item": {"text": "/status"}}],
        }

        await bot._process_message(inbound_payload)

        # Verify status reply content
        mock_session.post.assert_called_once()
        body = json.loads(mock_session.post.call_args[1]["data"])
        status_text = body["msg"]["item_list"][0]["text_item"]["text"]
        assert "Current Stats" in status_text
        assert "Session: 1 turns" in status_text
        assert "Context: 3K tokens" in status_text
        assert "Tool Calls: 2" in status_text
        assert "Time: 1.5s" in status_text


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_wechat_chatbot_trigger_cronjob(
    mock_session_cls: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that trigger_cronjob successfully creates session and posts message."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.post = MagicMock()
    mock_http_method(mock_session.post)
    mock_session.get = MagicMock()
    mock_http_method(mock_session.get)

    with patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=mock_config):
        bot = WechatChatbot(chatbot_id="wechat_test", gateway=mock_gateway)
        bot._poll_session = mock_session
        bot._send_session = mock_session

        # Run trigger_cronjob
        await bot.trigger_cronjob(
            channel_id="user_alice",
            prompt_content="Execute scheduled system check",
            mention_user_id="12345",
        )

        # Verify new session was created
        mock_gateway.create_session.assert_called_once()
        create_args = mock_gateway.create_session.call_args[1]
        assert "WeChat Scheduled Job user_alice" in create_args["title"]

        # Verify gateway.post was called
        mock_gateway.post.assert_called_once()
        posted = mock_gateway.post.call_args[0][0]
        assert posted.role == MessageRole.USER
        assert "Execute scheduled system check" in posted.content
        assert "@12345" in posted.content


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_wechat_chatbot_trigger_cronjob_auto_resolve(
    mock_session_cls: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that trigger_cronjob resolves channel_id from context store when not provided."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.post = MagicMock()
    mock_http_method(mock_session.post)
    mock_session.get = MagicMock()
    mock_http_method(mock_session.get)

    with patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=mock_config):
        bot = WechatChatbot(chatbot_id="wechat_test", gateway=mock_gateway)
        bot._poll_session = mock_session
        bot._send_session = mock_session

        # Save some channels in the token store
        bot._token_store.set("test_bot_id", "resolved_alice", "tok1")
        bot._token_store.set("test_bot_id", "resolved_bob", "tok2")
        bot._token_store.set("other_bot_id", "resolved_charlie", "tok3")

        # Run trigger_cronjob with channel_id=None
        await bot.trigger_cronjob(
            channel_id=None,
            prompt_content="Execute scheduled system check",
            mention_user_id="12345",
        )

        # Verify session creation was called for each resolved channel under test_bot_id
        # (resolved_alice and resolved_bob, but not resolved_charlie)
        assert mock_gateway.create_session.call_count == 2
        create_calls = mock_gateway.create_session.call_args_list
        titles = [call[1]["title"] for call in create_calls]
        assert "WeChat Scheduled Job resolved_alice" in titles
        assert "WeChat Scheduled Job resolved_bob" in titles

        # Verify gateway.post was called twice
        assert mock_gateway.post.call_count == 2
        posts = [call[0][0] for call in mock_gateway.post.call_args_list]
        channels = [p.channel_id for p in posts]
        assert "resolved_alice" in channels
        assert "resolved_bob" in channels
        assert "resolved_charlie" not in channels


def test_context_token_store_get_all_channels() -> None:
    """Test that ContextTokenStore.get_all_channels returns only channels for the given account."""
    from kesoku.gateway.chatbot.wechat.adapter import ContextTokenStore

    store = ContextTokenStore(persist_path=None)
    store.set("acc1", "userA", "tokA")
    store.set("acc1", "userB", "tokB")
    store.set("acc2", "userC", "tokC")

    assert sorted(store.get_all_channels("acc1")) == ["userA", "userB"]
    assert store.get_all_channels("acc2") == ["userC"]
    assert store.get_all_channels("acc3") == []


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_wechat_inbound_image_mime_sniffing(
    mock_session_cls: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Verify WeChat inbound image processing sniffs magic bytes to assign corrected mime type."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.post = MagicMock()
    mock_http_method(mock_session.post)
    mock_session.get = MagicMock()
    mock_http_method(mock_session.get)

    # Setup decrypted data with PNG magic bytes
    png_bytes = b"\x89PNG\r\n\x1a\nfake_png_content"

    from unittest.mock import mock_open

    with (
        patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=mock_config),
        patch(
            "kesoku.gateway.chatbot.wechat.adapter.WeChatMediaManager.download_and_decrypt",
            return_value=png_bytes,
        ) as mock_dl,
        patch("builtins.open", mock_open()) as mock_file_open,
    ):
        bot = WechatChatbot(chatbot_id="wechat_test", gateway=mock_gateway)
        bot._poll_session = mock_session
        bot._send_session = mock_session

        inbound_payload = {
            "from_user_id": "user_alice",
            "to_user_id": "test_bot_id",
            "message_id": "msg_002",
            "item_list": [
                {
                    "type": 2,  # ITEM_IMAGE
                    "image_item": {
                        "aeskey": "1234567890abcdef1234567890abcdef",
                        "media": {
                            "encrypt_query_param": "query_val",
                        },
                    },
                }
            ],
        }

        await bot._process_message(inbound_payload)

        # Verify that gateway.post is called with a message containing attachments metadata
        mock_gateway.post.assert_called_once()
        posted_msg = mock_gateway.post.call_args[0][0]
        assert "attachments" in posted_msg.metadata
        attachments = posted_msg.metadata["attachments"]
        assert len(attachments) == 1
        # The mime type should be correctly identified as image/png and the extension should be .png
        assert attachments[0]["mime_type"] == "image/png"
        assert attachments[0]["filename"].endswith(".png")


def test_compress_large_image() -> None:
    """Verify that _compress_image successfully reduces the size of a large image."""
    import io
    import random

    from PIL import Image, ImageDraw

    from kesoku.gateway.chatbot.wechat.adapter import _compress_image

    # Generate a large 2000x2000 RGBA image with random high-entropy lines to prevent PNG compression
    img = Image.new("RGBA", (2000, 2000), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    random.seed(42)
    for _ in range(1000):
        x0 = random.randint(0, 2000)
        y0 = random.randint(0, 2000)
        x1 = random.randint(0, 2000)
        y1 = random.randint(0, 2000)
        draw.line(
            [(x0, y0), (x1, y1)],
            fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255),
            width=15,
        )

    out = io.BytesIO()
    img.save(out, format="PNG")
    original_bytes = out.getvalue()

    assert len(original_bytes) > 500_000  # Ensure it's > 500KB

    # Compress the image using our helper to be under 200KB
    compressed_bytes = _compress_image(original_bytes, max_size=200_000)

    # Verify that it was successfully compressed to under 200KB
    assert len(compressed_bytes) < len(original_bytes)
    assert len(compressed_bytes) <= 200_000

    # Verify the compressed image is a valid JPEG
    compressed_img = Image.open(io.BytesIO(compressed_bytes))
    assert compressed_img.format == "JPEG"


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_wechat_chatbot_send_file_retry_success(
    mock_session_cls: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
    tmp_path,
) -> None:
    """Test that send_file_segment retries and succeeds if a later attempt is successful."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    # Create dummy outbound file
    dummy_file = tmp_path / "test.png"
    dummy_file.write_bytes(b"dummy_content")

    with patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=mock_config):
        bot = WechatChatbot(chatbot_id="wechat_test", gateway=mock_gateway)
        bot._poll_session = mock_session
        bot._send_session = mock_session

        # Mock _send_file to fail on first attempt and succeed on second
        call_count = 0

        async def mock_send_file_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Temporary CDN error")
            return "msg_ok"

        bot._send_file = AsyncMock(side_effect=mock_send_file_side_effect)

        outbound_msg = Message(
            id="msg_file_1",
            session_id="sess123",
            chatbot_id="wechat_test",
            channel_id="user_alice",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="image.png",
        )

        # We patch asyncio.sleep to speed up the test execution
        with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            await bot.send_file_segment("user_alice", str(dummy_file), outbound_msg)

            # Verify asyncio.sleep was called once (after first failure)
            mock_sleep.assert_called_once_with(2)

        # Verify _send_file was called twice
        assert call_count == 2


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_wechat_chatbot_send_voice_reverted(
    mock_session_cls: MagicMock,
    mock_config: KesokuConfig,
    mock_gateway: MagicMock,
) -> None:
    """Test that send_voice_segment delegates directly to send_file_segment."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    with patch("kesoku.gateway.chatbot.wechat.adapter.get_config", return_value=mock_config):
        bot = WechatChatbot(chatbot_id="wechat_test", gateway=mock_gateway)
        bot._poll_session = mock_session
        bot._send_session = mock_session

        bot.send_file_segment = AsyncMock()
        outbound_msg = Message(
            id="msg_v1",
            session_id="s1",
            chatbot_id="wechat_test",
            channel_id="u1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="voice.wav",
        )
        await bot.send_voice_segment("u1", "voice.wav", outbound_msg)
        bot.send_file_segment.assert_called_once_with("u1", "voice.wav", outbound_msg)

"""Comprehensive unit tests for v0.1.5 deferred voice UX, post-turn explicit cache, attachment LRU, and telemetry."""

import asyncio
import io
import json
import os
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from PIL import Image

from kesoku.agent.agent import Agent, SessionWorker
from kesoku.agent.llm import (
    _IMAGE_ATTACHMENT_LRU,
    _IMAGE_ATTACHMENT_LRU_LOCK,
    _IMAGE_ATTACHMENT_LRU_MAXSIZE,
    BaseLLM,
    GeminiLLM,
    LLMResponse,
    ToolCallRequest,
    _load_attachments,
    history_to_turns,
    prewarm_image_attachments,
)
from kesoku.agent.turn_executor import (
    TurnExecutor,
    discard_session_cache_state,
)
from kesoku.config import (
    DiscordChannelOverride,
    DiscordConfig,
    GeminiConfig,
    KesokuConfig,
    WorkspaceConfig,
    resolve_gemini_cache_ttl,
)
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import AsyncDatabaseManager, Message
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.chatbot.discord.adapter import MAX_TRACKED_VOICE_ANCHORS, DiscordChatbot
from kesoku.gateway.chatbot.discord.voice import VoiceFile, send_voice_message
from kesoku.gateway.chatbot.wechat.adapter import WechatChatbot
from kesoku.gateway.gateway import Gateway


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


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


async def _wait_until(predicate: Any, timeout: float = 3.0, step: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        res = predicate()
        if asyncio.iscoroutine(res):
            res = await res
        if res:
            return
        await asyncio.sleep(step)
    raise AssertionError("Timed out waiting for condition")


# ---------------------------------------------------------------------------
# 1. §1.2 Deferred Voice Zero-Jump Delivery (4s Inline Wait + Voice Anchor)
# ---------------------------------------------------------------------------


def test_voice_inline_grace_attributes() -> None:
    """Verify `_voice_inline_grace_sec` defaults: 4.0 on DiscordChatbot, 0.0 on Chatbot/WechatChatbot."""
    assert Chatbot._voice_inline_grace_sec == 0.0
    assert WechatChatbot._voice_inline_grace_sec == 0.0
    assert DiscordChatbot._voice_inline_grace_sec == 4.0


@pytest.mark.asyncio
async def test_voice_inline_grace_3_state_dispatch_ready_pending_failed(tmp_path: Any) -> None:
    """Verify 3-state dispatch after bounded inline wait: ready -> inline, pending -> tracker, failed -> notice."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="sv1", title="Voice3State", chatbot_id="discord_v015", channel_id="11")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    bot = _make_discord_bot(gw)
    bot._voice_inline_grace_sec = 0.15

    partial_msgs: dict[int, MagicMock] = {}

    def _get_partial_msg(mid: int) -> MagicMock:
        if mid not in partial_msgs:
            pm = MagicMock(id=mid)
            pm.add_reaction = AsyncMock()
            pm.remove_reaction = AsyncMock()
            pm.edit = AsyncMock()
            pm.delete = AsyncMock()
            partial_msgs[mid] = pm
        return partial_msgs[mid]

    partial_chan = MagicMock()
    partial_chan.get_partial_message = MagicMock(side_effect=_get_partial_msg)
    bot.bot.get_partial_messageable = MagicMock(return_value=partial_chan)  # type: ignore[method-assign]

    sent_texts: list[tuple[int, str, Any]] = []
    next_msg_id = 5000

    mock_channel = MagicMock()
    mock_channel.id = 11

    async def fake_channel_send(*args: Any, **kwargs: Any) -> Any:
        nonlocal next_msg_id
        next_msg_id += 1
        text = args[0] if args else kwargs.get("content", "")
        sent_texts.append((next_msg_id, text, kwargs.get("reference")))
        m = MagicMock(id=next_msg_id)
        m.channel = mock_channel
        return m

    mock_channel.send = AsyncMock(side_effect=fake_channel_send)
    bot._get_discord_channel = AsyncMock(return_value=mock_channel)  # type: ignore[method-assign]

    voice_calls: list[tuple[str, str, dict[str, Any] | None]] = []

    async def fake_send_voice_segment(
        cid: str,
        fpath: str,
        message: Message,
        *,
        message_reference: dict[str, Any] | None = None,
    ) -> None:
        _ = message
        voice_calls.append((cid, fpath, message_reference))

    bot.send_voice_segment = AsyncMock(side_effect=fake_send_voice_segment)  # type: ignore[method-assign]
    questions_sent: list[str] = []
    bot.send_question_segment = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda cid, q, choices, m, **kw: questions_sent.append(q)
    )

    # --- Case A: Ready within grace window -> inline send_voice_segment, ZERO placeholder/reaction ---
    wav_fast = os.path.join(staging_dir, "clip_fast.wav")
    _write_json(
        f"{wav_fast}.pending.json",
        {"v": 1, "job_id": "jf1", "created_at": time.time(), "timeout_sec": 180, "pid": 0},
    )

    async def _finish_fast() -> None:
        await asyncio.sleep(0.04)
        with open(wav_fast, "wb") as fh:
            fh.write(b"RIFF_FAST_WAV")
        os.unlink(f"{wav_fast}.pending.json")

    bg_fast = asyncio.create_task(_finish_fast())
    msg_fast = Message(
        id="m_fast_voice",
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="11",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"Opening line.\n[voice: {wav_fast}]\nClosing line.",
    )
    await db.save_message(msg_fast)
    await bot.render_outgoing_message(msg_fast)
    await bg_fast

    assert len(voice_calls) == 1
    assert voice_calls[0][1] == os.path.realpath(wav_fast)
    assert voice_calls[0][2] is None  # Sent inline in exact segment order!
    assert [t.strip() for _, t, _ in sent_texts] == ["Opening line.", "Closing line."]
    assert partial_msgs == {}

    # --- Case B: Grace expires (`wait_for_deferred_file` -> `(None, "生成超时")`) while still pending ---
    # MUST transition to `_start_deferred_tracking` (NOT fail!), react `🎤` on preceding text chunk,
    # send ZERO text placeholder, then deliver with `message_reference` and remove `🎤`!
    sent_texts.clear()
    voice_calls.clear()
    wav_slow = os.path.join(staging_dir, "clip_slow.wav")
    _write_json(
        f"{wav_slow}.pending.json",
        {"v": 1, "job_id": "js1", "created_at": time.time(), "timeout_sec": 180, "pid": 0},
    )
    msg_slow = Message(
        id="m_slow_voice",
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="11",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=(
            f"Spoken Japanese quote.\n[voice: {wav_slow}]\nNarrative body.\n"
            "```ui\ntype: select\nprompt: Choose action\noptions:\n  - Go left\n  - Go right\n```"
        ),
    )
    await db.save_message(msg_slow)
    await bot.render_outgoing_message(msg_slow)

    # Zero text placeholder was sent; UI prompt was delivered; `🎤` reacted on the preceding chunk!
    assert [t.strip() for _, t, _ in sent_texts] == ["Spoken Japanese quote.", "Narrative body."]
    quote_msg_id = sent_texts[0][0]
    assert questions_sent == ["Choose action"]
    partial_msgs[quote_msg_id].add_reaction.assert_awaited_once_with("🎤")

    saved_slow = await db.get_message(msg_slow.id)
    assert saved_slow is not None
    entry_slow = saved_slow.metadata["deferred_media"][0]
    assert entry_slow["state"] == "pending"
    assert entry_slow["handle"] == {
        "kind": "voice_anchor",
        "channel_id": "11",
        "message_id": str(quote_msg_id),
        "reaction": "🎤",
    }

    # Now background worker completes slow WAV
    with open(wav_slow, "wb") as fh:
        fh.write(b"RIFF_SLOW_WAV")
    os.unlink(f"{wav_slow}.pending.json")

    await _wait_until(lambda: len(voice_calls) == 1)
    assert voice_calls[0] == (
        "11",
        os.path.realpath(wav_slow),
        {"channel_id": "11", "message_id": str(quote_msg_id), "fail_if_not_exists": False},
    )
    await _wait_until(lambda: partial_msgs[quote_msg_id].remove_reaction.await_count == 1)
    assert partial_msgs[quote_msg_id].remove_reaction.await_args.args[0] == "🎤"
    partial_msgs[quote_msg_id].delete.assert_not_awaited()
    partial_msgs[quote_msg_id].edit.assert_not_awaited()

    # --- Case C: Worker writes `.error.json` during grace wait -> transitions to `failed` cleanly ---
    sent_texts.clear()
    questions_sent.clear()
    wav_err = os.path.join(staging_dir, "clip_err.wav")
    _write_json(
        f"{wav_err}.pending.json",
        {"v": 1, "job_id": "je1", "created_at": time.time(), "timeout_sec": 180, "pid": 0},
    )

    async def _fail_during_grace() -> None:
        await asyncio.sleep(0.03)
        _write_json(f"{wav_err}.error.json", {"v": 1, "job_id": "je1", "error": "TTS server 503"})
        os.unlink(f"{wav_err}.pending.json")

    bg_err = asyncio.create_task(_fail_during_grace())
    msg_err = Message(
        id="m_err_voice",
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="11",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=(
            f"Line before broken voice.\n[voice: {wav_err}]\n"
            "```ui\ntype: select\nprompt: Still rendered?\noptions:\n  - Yes\n```"
        ),
    )
    await db.save_message(msg_err)
    await bot.render_outgoing_message(msg_err)
    await bg_err

    anchor_err_id = sent_texts[0][0]
    partial_msgs[anchor_err_id].add_reaction.assert_awaited_once_with("⚠️")
    assert any("⚠️ 语音生成失败（TTS server 503）" in t for _, t, _ in sent_texts)
    assert questions_sent == ["Still rendered?"]


@pytest.mark.asyncio
async def test_voice_anchor_top_of_message_falls_back_to_inbound_user_msg_not_prior_turn(tmp_path: Any) -> None:
    """Verify top-of-message `[voice:]` anchors to triggering user `discord_message_id`, never prior turn's reply."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="sv2", title="TopVoice", chatbot_id="discord_v015", channel_id="22")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    bot = _make_discord_bot(gw)
    bot._voice_inline_grace_sec = 0.02

    partial_msgs: dict[int, MagicMock] = {}

    def _get_partial_msg(mid: int) -> MagicMock:
        if mid not in partial_msgs:
            pm = MagicMock(id=mid)
            pm.add_reaction = AsyncMock()
            pm.remove_reaction = AsyncMock()
            pm.edit = AsyncMock()
            pm.delete = AsyncMock()
            partial_msgs[mid] = pm
        return partial_msgs[mid]

    partial_chan = MagicMock()
    partial_chan.get_partial_message = MagicMock(side_effect=_get_partial_msg)
    bot.bot.get_partial_messageable = MagicMock(return_value=partial_chan)  # type: ignore[method-assign]

    mock_channel = MagicMock(id=22)
    sent_id = 6000

    async def fake_send(*_a: Any, **_kw: Any) -> Any:
        nonlocal sent_id
        sent_id += 1
        return MagicMock(id=sent_id, channel=mock_channel)

    mock_channel.send = AsyncMock(side_effect=fake_send)
    bot._get_discord_channel = AsyncMock(return_value=mock_channel)  # type: ignore[method-assign]

    # Turn 1 sends text message (id=6001)
    m_turn1 = Message(
        id="m_turn1_assistant",
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="22",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="Turn 1 reply text.",
    )
    await db.save_message(m_turn1)
    await bot.render_outgoing_message(m_turn1)
    assert "m_turn1_assistant" in bot._last_text_msg

    # Turn 2 user message arrives with `discord_message_id = "777777"`
    u_turn2 = Message(
        id="u_turn2_user",
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="22",
        sender="User",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Speak first!",
        metadata={"discord_message_id": "777777", "discord_message_channel_id": "22"},
    )
    await db.save_message(u_turn2)

    # Turn 2 assistant message has `[voice:]` AT THE VERY TOP before any text chunk!
    wav_top = os.path.join(staging_dir, "top_clip.wav")
    _write_json(
        f"{wav_top}.pending.json",
        {"v": 1, "job_id": "jtop", "created_at": time.time(), "timeout_sec": 180, "pid": 0},
    )
    m_turn2 = Message(
        id="m_turn2_assistant",
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="22",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        parent_id=u_turn2.id,
        content=f"[voice: {wav_top}]\nText after leading voice.",
    )
    await db.save_message(m_turn2)
    await bot.render_outgoing_message(m_turn2)

    # MUST anchor on inbound user message 777777, NOT Turn 1's message 6001!
    assert 6001 not in partial_msgs
    assert 777777 in partial_msgs
    partial_msgs[777777].add_reaction.assert_awaited_once_with("🎤")

    saved_t2 = await db.get_message("m_turn2_assistant")
    assert saved_t2 is not None
    assert saved_t2.metadata["deferred_media"][0]["handle"]["message_id"] == "777777"
    await bot.cancel_deferred_for_session(session.id)

    # Verify bounded LRU on `_last_text_msg` (<= 256)
    for i in range(300):
        bot._record_last_text_msg(f"out_msg_{i}", MagicMock(id=9000 + i))
    assert len(bot._last_text_msg) == MAX_TRACKED_VOICE_ANCHORS


@pytest.mark.asyncio
async def test_voice_anchor_startup_recovery_and_failure_never_edit_or_delete(tmp_path: Any) -> None:
    """Verify `_recover_deferred_media` restores `voice_anchor` dicts and NEVER calls `.edit()`/`.delete()`."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="sv_rec", title="RecVoice", chatbot_id="discord_v015", channel_id="33")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    ready_wav = os.path.join(staging_dir, "recovered_voice.wav")
    with open(ready_wav, "wb") as fh:
        fh.write(b"RIFF_RECOVERED")

    fail_wav = os.path.join(staging_dir, "failed_voice.wav")
    _write_json(f"{fail_wav}.error.json", {"v": 1, "job_id": "j_vfail", "error": "OOM killed"})

    persisted_msg = Message(
        id="m_voice_recover",
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="33",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"Hello\n[voice: {ready_wav}]\n[voice: {fail_wav}]",
        timestamp=time.time() - 10.0,
        status=MessageStatus.DELIVERED,
        metadata={
            "deferred_media": [
                {
                    "id": "j_vok",
                    "path": ready_wav,
                    "raw_path": ready_wav,
                    "kind": "voice",
                    "state": "pending",
                    "marker": {"job_id": "j_vok", "created_at": time.time() - 10.0, "timeout_sec": 180, "pid": 0},
                    "handle": {"kind": "voice_anchor", "channel_id": "33", "message_id": "88001", "reaction": "🎤"},
                },
                {
                    "id": "j_vfail",
                    "path": fail_wav,
                    "raw_path": fail_wav,
                    "kind": "voice",
                    "state": "pending",
                    "marker": {"job_id": "j_vfail", "created_at": time.time() - 10.0, "timeout_sec": 180, "pid": 0},
                    "handle": {"kind": "voice_anchor", "channel_id": "33", "message_id": "88002", "reaction": "🎤"},
                },
            ]
        },
    )
    await db.save_message(persisted_msg)

    bot = _make_discord_bot(gw)
    bot.bot.is_ready = MagicMock(return_value=True)  # type: ignore[method-assign]

    partial_msgs: dict[int, MagicMock] = {}

    def _get_partial(mid: int) -> MagicMock:
        if mid not in partial_msgs:
            pm = MagicMock(id=mid)
            pm.add_reaction = AsyncMock()
            pm.remove_reaction = AsyncMock()
            pm.edit = AsyncMock(side_effect=AssertionError("edit() MUST NEVER be called on voice_anchor!"))
            pm.delete = AsyncMock(side_effect=AssertionError("delete() MUST NEVER be called on voice_anchor!"))
            partial_msgs[mid] = pm
        return partial_msgs[mid]

    partial_chan = MagicMock()
    partial_chan.get_partial_message = MagicMock(side_effect=_get_partial)
    bot.bot.get_partial_messageable = MagicMock(return_value=partial_chan)  # type: ignore[method-assign]

    mock_channel = MagicMock(id=33)
    sent_notices: list[tuple[str, Any]] = []
    mock_channel.send = AsyncMock(
        side_effect=lambda text, **kw: sent_notices.append((text, kw.get("reference"))) or MagicMock(id=9100)
    )
    bot._get_discord_channel = AsyncMock(return_value=mock_channel)  # type: ignore[method-assign]

    delivered_voices: list[tuple[str, dict[str, Any] | None]] = []
    bot.send_voice_segment = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda cid, fpath, m, *, message_reference=None: delivered_voices.append(
            (fpath, message_reference)
        )
    )

    await bot._recover_deferred_media()

    async def _both_settled() -> bool:
        m = await db.get_message("m_voice_recover")
        if not m:
            return False
        states = [e["state"] for e in m.metadata["deferred_media"]]
        return states == ["delivered", "failed"]

    await _wait_until(_both_settled)

    assert delivered_voices == [
        (os.path.realpath(ready_wav), {"channel_id": "33", "message_id": "88001", "fail_if_not_exists": False})
    ]
    partial_msgs[88001].remove_reaction.assert_awaited_once()
    assert partial_msgs[88001].remove_reaction.await_args.args[0] == "🎤"

    partial_msgs[88002].remove_reaction.assert_awaited_once()
    assert partial_msgs[88002].remove_reaction.await_args.args[0] == "🎤"
    partial_msgs[88002].add_reaction.assert_awaited_once_with("⚠️")
    assert len(sent_notices) == 1
    assert sent_notices[0][0] == "⚠️ 语音生成失败（OOM killed）"
    assert isinstance(sent_notices[0][1], discord.MessageReference)
    assert sent_notices[0][1].fail_if_not_exists is False


@pytest.mark.asyncio
async def test_send_voice_message_http_400_retries_without_reference_using_fresh_file(tmp_path: Any) -> None:
    """Verify HTTP 400 on `flags=8192 + message_reference` retries ONCE immediately without reference & fresh File."""
    wav_in = str(tmp_path / "input.wav")
    with open(wav_in, "wb") as fh:
        fh.write(b"RIFF_DUMMY")

    mock_channel = AsyncMock(spec=discord.Thread)
    mock_channel.id = 12345
    mock_channel._state = MagicMock()
    mock_channel._state.create_message = MagicMock(return_value="created_voice_msg")

    seen_payloads: list[dict[str, Any]] = []
    seen_files_open: list[bool] = []
    seen_file_objs: list[VoiceFile] = []

    async def fake_http_send_message(channel_id: int, *, params: Any) -> dict[str, Any]:
        assert channel_id == 12345
        raw_json = params.multipart[0]["value"]
        payload = json.loads(raw_json)
        seen_payloads.append(payload)
        f_obj = params.files[0]
        seen_file_objs.append(f_obj)
        seen_files_open.append(not f_obj.fp.closed)
        if len(seen_payloads) == 1:
            resp = MagicMock(status=400, reason="Bad Request")
            raise discord.HTTPException(resp, "Cannot use message_reference with voice message flags")
        return {"id": "999111"}

    mock_channel._state.http = MagicMock()
    mock_channel._state.http.send_message = AsyncMock(side_effect=fake_http_send_message)

    def fake_subproc_run(cmd: list[str], **_kwargs: Any) -> MagicMock:
        if cmd[0] == "ffmpeg":
            with open(cmd[-1], "wb") as out_fh:
                out_fh.write(b"OggS_OPUS_DATA")
            return MagicMock(returncode=0, stdout="")
        return MagicMock(returncode=0, stdout="2.75\n")

    ref_dict = {"channel_id": "12345", "message_id": "77700", "fail_if_not_exists": False}
    with patch("subprocess.run", side_effect=fake_subproc_run) as mock_run:
        res = await send_voice_message(mock_channel, wav_in, message_reference=ref_dict)

    assert res == "created_voice_msg"
    # FFmpeg + ffprobe ran only ONCE across both HTTP attempts!
    assert mock_run.call_count == 2
    assert len(seen_payloads) == 2
    assert seen_payloads[0]["flags"] == 8192
    assert seen_payloads[0]["message_reference"] == ref_dict
    assert seen_payloads[0]["allowed_mentions"]["replied_user"] is False

    # Retry omitted message_reference cleanly and used a fresh, open VoiceFile!
    assert seen_payloads[1]["flags"] == 8192
    assert "message_reference" not in seen_payloads[1]
    assert len(seen_file_objs) == 2
    assert seen_file_objs[0] is not seen_file_objs[1]
    assert seen_files_open == [True, True]


# ---------------------------------------------------------------------------
# 2. §2.2 Post-Turn Explicit Context Cache Lifecycle
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3. §2.4 Multi-Image Attachment WebP LRU + Parallel Download + Prewarm
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 4. §2.5 Per-Round Telemetry (`ttft_ms` & `turn_metrics.rounds`)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_ttft_ms_and_turn_metrics_per_round_telemetry(tmp_path: Any) -> None:
    """Verify `LLMResponse.ttft_ms` (first non-thought chunk) and `turn_metrics.rounds[]` telemetry."""
    # 1. Direct `BaseLLM.generate` streaming `ttft_ms` ignores thought chunks and measures first text delta
    mock_client = MagicMock()

    def _delayed_stream(**_kw: Any) -> Any:
        time.sleep(0.01)
        p_thought = MagicMock(text="thinking...", thought=True, function_call=None, thought_signature=None)
        yield MagicMock(
            candidates=[MagicMock(content=MagicMock(parts=[p_thought]), finish_reason=None, safety_ratings=None)],
            usage_metadata=None,
            prompt_feedback=None,
        )
        time.sleep(0.04)
        p_first = MagicMock(text="First visible", thought=False, function_call=None, thought_signature=None)
        yield MagicMock(
            candidates=[MagicMock(content=MagicMock(parts=[p_first]), finish_reason=None, safety_ratings=None)],
            usage_metadata=None,
            prompt_feedback=None,
        )
        time.sleep(0.08)
        p_tail = MagicMock(text=" tail.", thought=False, function_call=None, thought_signature=None)
        yield MagicMock(
            candidates=[MagicMock(content=MagicMock(parts=[p_tail]), finish_reason="STOP", safety_ratings=None)],
            usage_metadata=MagicMock(
                prompt_token_count=100,
                candidates_token_count=10,
                total_token_count=110,
                cached_content_token_count=80,
            ),
            prompt_feedback=None,
        )

    mock_client.models.generate_content_stream.side_effect = _delayed_stream
    with patch("google.genai.Client", return_value=mock_client):
        g_llm = GeminiLLM(GeminiConfig(thinking_level="low", stream=True))
        stream_res = await g_llm.generate(prompt="Hi", stream=True)
    assert stream_res.ttft_ms is not None
    assert 35.0 <= stream_res.ttft_ms < 115.0  # After thought (~50ms), strictly before tail finish (~130ms)!

    # 2. Multi-round `TurnExecutor` populates `turn_metrics` extra fields + `rounds[]`
    cfg = KesokuConfig(
        agent_working_dir=str(tmp_path),
        workspace=WorkspaceConfig(sessions_dir=str(tmp_path / "sessions"), db_path=str(tmp_path / "tel.db")),
        gemini=GeminiConfig(
            model_name="gemini-3.8-flash",
            thinking_level="low",
            stream=True,
            context_caching=True,
            context_caching_mode="explicit",
        ),
    )
    gw, db = await _make_gateway(tmp_path, cfg=cfg)
    gw.register_stream_handler("discord_tel", AsyncMock())
    await gw.create_session("s_tel", title="Telemetry", chatbot_id="discord_tel", channel_id="ch_tel")
    staging_dir = str(tmp_path / "sessions" / "s_tel")
    os.makedirs(staging_dir, exist_ok=True)

    # Seed Turn 0 message so explicit cache prefix has length 1 < Turn 1's length 2
    seed_msg = Message(
        id="u_seed",
        session_id="s_tel",
        chatbot_id="discord_tel",
        channel_id="ch_tel",
        sender="u",
        role=MessageRole.USER,
        content="Seed history message",
        status=MessageStatus.PROCESSED,
        timestamp=time.time() - 20.0,
    )
    await gw.post(seed_msg)

    class TelemetryGeminiLLM(BaseLLM):
        model_name = "gemini-3.8-flash"

        def __init__(self) -> None:
            self.config = cfg.gemini
            self.calls = 0

        async def generate(self, **_kwargs: Any) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="",
                    thought="Calling two tools",
                    tool_calls=[
                        ToolCallRequest(name="ok_tool", arguments={}, tool_call_id="tc1"),
                        ToolCallRequest(name="bad_tool", arguments={}, tool_call_id="tc2"),
                    ],
                    prompt_tokens=6000,
                    cached_tokens=5200,
                    candidates_tokens=45,
                    total_tokens=6045,
                    ttft_ms=88.5,
                )
            return LLMResponse(
                content="All done!",
                prompt_tokens=6400,
                cached_tokens=5200,
                candidates_tokens=60,
                total_tokens=6460,
                ttft_ms=42.25,
            )

    TelemetryGeminiLLM.__name__ = "GeminiLLM"
    llm = TelemetryGeminiLLM()
    ctx = KesokuContext(config=cfg, db=gw.context.sync_db, llm=llm)
    gw.context = ctx

    tool_runner = MagicMock()
    tool_runner.tool_registry.get_tools_list.return_value = []
    tool_runner.tool_context = MagicMock(transitioned_to_session=None)

    async def _exec_tool(call: Any, tc_msg: Message, **_kw: Any) -> Message:
        await asyncio.sleep(0.02)
        meta: dict[str, Any] = {"tool_name": call.name}
        if call.name == "bad_tool":
            meta["tool_error"] = "exit code 2"
        else:
            meta["tool_result"] = "ok"
        return Message(
            session_id=tc_msg.session_id,
            chatbot_id=tc_msg.chatbot_id,
            channel_id=tc_msg.channel_id,
            sender=call.name,
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="out",
            status=MessageStatus.RESPONDED,
            parent_id=tc_msg.id,
            metadata=meta,
        )

    tool_runner.execute_tool.side_effect = _exec_tool

    worker = SessionWorker("s_tel", gw, context=ctx)
    worker.active_cache_name = "projects/p/locations/global/cachedContents/seed_c"
    worker.active_cache_llm = llm
    worker.cached_messages_len = 1

    u_active = Message(
        id="u_tel_1",
        session_id="s_tel",
        chatbot_id="discord_tel",
        channel_id="ch_tel",
        sender="u",
        role=MessageRole.USER,
        content="Run tools please",
        status=MessageStatus.PENDING_AGENT,
        timestamp=time.time(),
    )
    await gw.post(u_active)

    ex = TurnExecutor("s_tel", gw, tool_runner, context=ctx)
    worker._running = True
    with patch("kesoku.agent.compressor.HistoryCompressor.auto_compact_session", return_value=False):
        await ex.process_turn(current_msg=u_active, worker=worker, session_staging_dir=staging_dir)

    hist = await db.get_session_history("s_tel")
    final_assistant = [m for m in hist if m.role == MessageRole.ASSISTANT and m.type == MessageType.TEXT][-1]
    tm = final_assistant.metadata["turn_metrics"]

    # Legacy fields intact
    assert tm["status"] == "finished"
    assert tm["turn_tool_calls"] == 2
    assert tm["context_tokens"] == 6400
    assert tm["cached_tokens"] == 5200
    assert tm["turn_tokens"] == 6045 + 6460

    # New v0.1.5 telemetry fields
    assert tm["model_name"] == "gemini-3.8-flash"
    assert tm["thinking_level"] == "low"
    assert tm["stream"] is True
    assert tm["cache_mode"] == "explicit"
    assert tm["used_explicit_cache"] is True
    assert len(tm["rounds"]) == 2

    r1, r2 = tm["rounds"]
    assert r1["round"] == 1
    assert r1["prompt_tokens"] == 6000
    assert r1["cached_tokens"] == 5200
    assert r1["output_tokens"] == 45
    assert r1["ttft_ms"] == 88.5
    assert r1["tool_count"] == 2
    assert r1["tool_errors"] == 1
    assert r1["tool_ms"] >= 15.0

    assert r2["round"] == 2
    assert r2["prompt_tokens"] == 6400
    assert r2["cached_tokens"] == 5200
    assert r2["output_tokens"] == 60
    assert r2["ttft_ms"] == 42.25
    assert r2["tool_count"] == 0
    assert r2["tool_errors"] == 0
    assert r2["tool_ms"] == 0.0
    discard_session_cache_state("s_tel")


@pytest.mark.asyncio
async def test_multi_voice_same_message_anchors_and_remaining_timeout_cap(tmp_path: Any) -> None:
    """Verify two `[voice:]` tags in one message anchor to their respective preceding chunks & cap remaining timeout."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="sv_m", title="MultiVoice", chatbot_id="discord_v015", channel_id="55")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    bot = _make_discord_bot(gw)
    bot._voice_inline_grace_sec = 4.0  # Full 4.0s grace, but remaining marker timeout is ~0.05s!

    seen_caps: list[float] = []

    async def _spy_wait(candidates: list[str], **kwargs: Any) -> tuple[str | None, str | None]:
        _ = candidates
        seen_caps.append(float(kwargs["max_wait_cap"]))
        return (None, "生成超时")

    mock_channel = MagicMock(id=55)
    seq_id = 7000

    async def _send_chunk(*_a: Any, **_kw: Any) -> Any:
        nonlocal seq_id
        seq_id += 1
        return MagicMock(id=seq_id, channel=mock_channel)

    mock_channel.send = AsyncMock(side_effect=_send_chunk)
    bot._get_discord_channel = AsyncMock(return_value=mock_channel)  # type: ignore[method-assign]
    partial_chan = MagicMock()
    partial_chan.get_partial_message = MagicMock(
        side_effect=lambda mid: MagicMock(id=mid, add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    )
    bot.bot.get_partial_messageable = MagicMock(return_value=partial_chan)  # type: ignore[method-assign]

    now = time.time()
    w1 = os.path.join(staging_dir, "line1.wav")
    w2 = os.path.join(staging_dir, "line2.wav")
    # Marker created 9.92s ago with 10s clamped timeout -> remaining timeout ~0.08s << 4.0s!
    _write_json(f"{w1}.pending.json", {"v": 1, "job_id": "jw1", "created_at": now - 9.92, "timeout_sec": 10, "pid": 0})
    _write_json(f"{w2}.pending.json", {"v": 1, "job_id": "jw2", "created_at": now - 9.92, "timeout_sec": 10, "pid": 0})

    msg = Message(
        id="m_two_voices",
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="55",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"First line.\n[voice: {w1}]\nSecond line.\n[voice: {w2}]",
    )
    await db.save_message(msg)
    with patch("kesoku.gateway.chatbot.base.wait_for_deferred_file", side_effect=_spy_wait):
        await bot.render_outgoing_message(msg)

    assert len(seen_caps) == 2
    assert all(cap < 0.5 for cap in seen_caps)  # Capped by remaining marker timeout (< 4.0s)!

    saved = await db.get_message("m_two_voices")
    assert saved is not None
    entries = saved.metadata["deferred_media"]
    assert entries[0]["handle"]["message_id"] == "7001"
    assert entries[1]["handle"]["message_id"] == "7002"
    await bot.cancel_deferred_for_session(session.id)


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


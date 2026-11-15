"""Unit tests for the generic deferred media attachment protocol."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from kesoku.agent.llm import (
    LLMResponse,
)
from kesoku.agent.state_snapshot import is_internal_staging_name, render_staging_artifacts
from kesoku.agent.tools.media import analyze_media
from kesoku.agent.tools.registry import ToolContext
from kesoku.agent.tools.shell import run_shell_command
from kesoku.agent.turn_executor import (
    TurnExecutor,
)
from kesoku.config import (
    DiscordConfig,
    KesokuConfig,
    WechatConfig,
    WorkspaceConfig,
)
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import AsyncDatabaseManager, DatabaseManager, Message
from kesoku.gateway.chatbot.base import (
    Chatbot,
)
from kesoku.gateway.chatbot.cli_bot import CLIChatbot
from kesoku.gateway.chatbot.cronjob import CronjobChatbot
from kesoku.gateway.chatbot.deferred_media import (
    DEFAULT_DEFERRED_PLACEHOLDER,
    clamp_timeout_sec,
    is_pid_alive,
    resolve_deferred_sync,
    sanitize_error_reason,
    sanitize_placeholder,
)
from kesoku.gateway.chatbot.discord.adapter import MAX_TRACKED_VOICE_ANCHORS, DiscordChatbot
from kesoku.gateway.chatbot.discord.voice import VoiceFile, send_voice_message
from kesoku.gateway.chatbot.google_chat.adapter import GoogleChatChatbot
from kesoku.gateway.chatbot.wechat.adapter import WechatChatbot
from kesoku.gateway.gateway import Gateway


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


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


def _make_discord_bot(gateway: Gateway, chatbot_id: str = "discord_test") -> DiscordChatbot:
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


@pytest.mark.asyncio
async def test_discord_deferred_pending_to_ready_and_fresh_file_on_retry(tmp_path: Any) -> None:
    """Verify text-first delivery, placeholder isolation, fresh discord.File retry, and metadata persistence."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="s1", title="Test", chatbot_id="discord_test", channel_id="1001")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    # Pre-existing sibling file that would score >0.90 under fuzzy matching if deferred check did not run first.
    sibling_path = os.path.join(staging_dir, "chart_11.png")
    with open(sibling_path, "wb") as fh:
        fh.write(b"OLD_SIBLING_PNG")

    target_path = os.path.join(staging_dir, "chart_12.png")
    created_at = time.time()
    _write_json(
        f"{target_path}.pending.json",
        {
            "v": 1,
            "kind": "deferred_media",
            "target": target_path,
            "job_id": "job_chart_12",
            "created_at": created_at,
            "timeout_sec": 120,
            "pid": 0,
            "placeholder": "📊 正在绘制图表…",
        },
    )

    bot = _make_discord_bot(gw)
    sent_events: list[tuple[str, Any]] = []

    placeholder_msg = MagicMock()
    placeholder_msg.id = 55501
    placeholder_msg.channel = MagicMock()
    placeholder_msg.channel.id = 1001
    placeholder_msg.delete = AsyncMock()

    seen_files: list[discord.File] = []
    seen_open_states: list[bool] = []
    seen_edit_kwargs: list[dict[str, Any]] = []

    async def fake_placeholder_edit(**kwargs: Any) -> None:
        seen_edit_kwargs.append(kwargs)
        attachments = kwargs.get("attachments") or []
        if attachments:
            f_obj = attachments[0]
            seen_files.append(f_obj)
            seen_open_states.append(bool(getattr(f_obj.fp, "closed", False)))
            if len(seen_files) == 1:
                raise TimeoutError("simulated gateway upload timeout")

    placeholder_msg.edit = AsyncMock(side_effect=fake_placeholder_edit)

    mock_channel = MagicMock()
    mock_channel.id = 1001

    async def fake_channel_send(*args: Any, **kwargs: Any) -> Any:
        if "file" in kwargs:
            sent_events.append(("file", kwargs["file"].filename))
            m = MagicMock(id=999)
            return m
        text = args[0] if args else kwargs.get("content", "")
        sent_events.append(("text", text))
        if text == "📊 正在绘制图表…":
            return placeholder_msg
        return MagicMock(id=888)

    mock_channel.send = AsyncMock(side_effect=fake_channel_send)
    bot._get_discord_channel = AsyncMock(return_value=mock_channel)  # type: ignore[method-assign]

    msg = Message(
        id="m_deferred_1",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="1001",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"Intro paragraph.\n[file: {target_path}]\nOutro paragraph.",
    )
    await db.save_message(msg)

    await bot.render_outgoing_message(msg)

    # Text before, placeholder, and text after were delivered in exact order without waiting for PNG.
    assert [ev[0] for ev in sent_events] == ["text", "text", "text"]
    assert "Intro paragraph." in sent_events[0][1]
    assert sent_events[1][1] == "📊 正在绘制图表…"
    assert "Outro paragraph." in sent_events[2][1]

    # Placeholder was NOT placed in _intermediate_messages or _turn_special_msg, so on_message_delivered kept it.
    placeholder_msg.delete.assert_not_awaited()

    # Pending metadata was persisted to SQLite before watcher completed.
    saved_before = await db.get_message(msg.id)
    assert saved_before is not None
    assert saved_before.metadata["deferred_media"][0]["state"] == "pending"
    assert saved_before.metadata["deferred_media"][0]["handle"] == {
        "channel_id": "1001",
        "message_id": "55501",
    }

    # Now producer atomically finishes PNG and removes .pending.json.
    tmp_part = os.path.join(staging_dir, ".deferred-tmp.part")
    with open(tmp_part, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\nFRESH_CHART_12")
    os.replace(tmp_part, target_path)
    os.unlink(f"{target_path}.pending.json")

    async def _is_delivered() -> bool:
        m = await db.get_message(msg.id)
        return bool(m and m.metadata["deferred_media"][0]["state"] == "delivered")

    await _wait_until(_is_delivered, timeout=4.0)

    assert len(seen_files) == 2
    assert seen_files[0] is not seen_files[1]
    assert seen_open_states == [False, False]
    for kw in seen_edit_kwargs:
        assert kw.get("content") is None
        assert "view" not in kw
    # Sibling chart_11.png was never sent!
    assert ("file", "chart_11.png") not in sent_events


@pytest.mark.asyncio
async def test_discord_deferred_notfound_and_permanent_edit_fallbacks_and_voice(tmp_path: Any) -> None:
    """Verify Discord NotFound -> send_file_segment, 403 -> reference reply, and voice delete+send."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="s2", title="Test2", chatbot_id="discord_test", channel_id="1002")
    bot = _make_discord_bot(gw)

    png_path = str(tmp_path / "result.png")
    with open(png_path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\nDATA")

    msg = Message(
        id="m_fb",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="1002",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="hi",
    )

    # 1. NotFound on placeholder.edit -> send_file_segment
    not_found_resp = MagicMock(status=404, reason="Not Found")
    handle_nf = MagicMock()
    handle_nf.edit = AsyncMock(side_effect=discord.NotFound(not_found_resp, "Unknown Message"))
    bot.send_file_segment = AsyncMock()  # type: ignore[method-assign]
    await bot.deliver_deferred_attachment("1002", png_path, msg, handle_nf, "file")
    bot.send_file_segment.assert_awaited_once_with("1002", png_path, msg)

    # 2. Permanent 403 error on placeholder.edit -> channel.send(file=..., reference=handle, mention_author=False)
    forbidden_resp = MagicMock(status=403, reason="Forbidden")
    handle_403 = MagicMock()
    handle_403.edit = AsyncMock(side_effect=discord.Forbidden(forbidden_resp, "Cannot edit"))
    mock_channel = MagicMock()
    mock_channel.send = AsyncMock()
    bot._get_discord_channel_with_abort = AsyncMock(return_value=mock_channel)  # type: ignore[method-assign]
    await bot.deliver_deferred_attachment("1002", png_path, msg, handle_403, "file")
    mock_channel.send.assert_awaited_once()
    send_kwargs = mock_channel.send.await_args.kwargs
    assert send_kwargs["reference"] is handle_403
    assert send_kwargs["mention_author"] is False
    assert isinstance(send_kwargs["file"], discord.File)

    # 3. Voice attachment: send_voice_segment first, then delete placeholder
    ogg_path = str(tmp_path / "speech.ogg")
    with open(ogg_path, "wb") as fh:
        fh.write(b"OggS")
    call_order: list[str] = []

    async def fake_voice(cid: str, fpath: str, m: Message) -> None:
        _ = (cid, fpath, m)
        call_order.append("send_voice")

    async def fake_del() -> None:
        call_order.append("delete_placeholder")

    bot.send_voice_segment = AsyncMock(side_effect=fake_voice)  # type: ignore[method-assign]
    handle_voice = MagicMock()
    handle_voice.delete = AsyncMock(side_effect=fake_del)
    await bot.deliver_deferred_attachment("1002", ogg_path, msg, handle_voice, "voice")
    assert call_order == ["send_voice", "delete_placeholder"]


@pytest.mark.asyncio
async def test_discord_discard_deferred_placeholder(tmp_path: Any) -> None:
    """Verify `discard_deferred_placeholder` removes `voice_anchor`, deletes message handles, and swallows errors."""
    gw, _ = await _make_gateway(tmp_path)
    session = await gw.create_session(
        session_id="s_disc", title="Discard", chatbot_id="discord_test", channel_id="1002"
    )
    bot = _make_discord_bot(gw)
    bot.bot.is_ready = MagicMock(return_value=True)  # type: ignore[method-assign]

    partial_msg = MagicMock(id=7788)
    partial_msg.remove_reaction = AsyncMock()
    partial_msg.delete = AsyncMock()
    partial_chan = MagicMock()
    partial_chan.get_partial_message = MagicMock(return_value=partial_msg)
    bot.bot.get_partial_messageable = MagicMock(return_value=partial_chan)  # type: ignore[method-assign]

    msg = Message(
        id="m_disc",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="1002",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="hi",
    )
    seg = {"type": "voice", "path": "clip.wav"}

    # 1. voice_anchor dict -> remove_reaction awaited once, delete never called
    anchor_handle = {"kind": "voice_anchor", "channel_id": "1002", "message_id": "7788", "reaction": "🎤"}
    await bot.discard_deferred_placeholder(anchor_handle, msg, seg)
    partial_msg.remove_reaction.assert_awaited_once()
    assert partial_msg.remove_reaction.await_args.args[0] == "🎤"
    partial_msg.delete.assert_not_awaited()

    # 2. Message/PartialMessage handle with delete -> delete awaited once
    msg_handle = MagicMock()
    msg_handle.delete = AsyncMock()
    await bot.discard_deferred_placeholder(msg_handle, msg, seg)
    msg_handle.delete.assert_awaited_once()

    # 3. Exceptions swallowed cleanly (both on delete and remove_reaction)
    bad_handle = MagicMock()
    bad_handle.delete = AsyncMock(side_effect=RuntimeError("delete boom"))
    await bot.discard_deferred_placeholder(bad_handle, msg, seg)
    bad_handle.delete.assert_awaited_once()

    partial_msg.remove_reaction = AsyncMock(side_effect=RuntimeError("reaction boom"))
    await bot._safe_discard_placeholder(anchor_handle, msg, seg)
    partial_msg.remove_reaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_wechat_deferred_send_later_and_failure_text(tmp_path: Any) -> None:
    """Verify WeChat omits placeholder spam, appends file when ready, and sends text on failure."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="sw", title="WeChat", chatbot_id="wechat_test", channel_id="wx_u1")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    wcfg = WechatConfig(chatbot_id="wechat_test", enabled=True, account_id="bot_acc", token="tok_123")
    with patch("kesoku.gateway.chatbot.wechat.adapter.get_config") as mock_cfg:
        mock_cfg.return_value = KesokuConfig(agent_working_dir=str(tmp_path))
        wbot = WechatChatbot("wechat_test", gw, wechat_config=wcfg)
    wbot._deferred_poll_initial_sec = 0.02
    wbot._deferred_poll_max_sec = 0.04
    wbot._deferred_dead_grace_sec = 0.08

    sent_texts: list[str] = []
    sent_files: list[str] = []
    wbot.send_text_chunks = AsyncMock(side_effect=lambda cid, chunks, m: sent_texts.extend(chunks))  # type: ignore[method-assign]
    wbot.send_file_segment = AsyncMock(side_effect=lambda cid, fpath, m: sent_files.append(fpath))  # type: ignore[method-assign]

    img_ok = os.path.join(staging_dir, "summary_card.png")
    _write_json(
        f"{img_ok}.pending.json",
        {"v": 1, "job_id": "wx1", "created_at": time.time(), "timeout_sec": 60, "pid": 0},
    )
    msg = Message(
        id="m_wx1",
        session_id=session.id,
        chatbot_id="wechat_test",
        channel_id="wx_u1",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"Hello WeChat!\n[file: {img_ok}]",
    )
    await db.save_message(msg)
    await wbot.render_outgoing_message(msg)

    # Only main text was sent — zero placeholder text!
    assert sent_texts == ["Hello WeChat!"]
    assert sent_files == []

    with open(img_ok, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\nWX")
    os.unlink(f"{img_ok}.pending.json")

    await _wait_until(lambda: len(sent_files) == 1)
    assert sent_files == [os.path.realpath(img_ok)]

    # Now test failure path on WeChat
    img_err = os.path.join(staging_dir, "failed_card.png")
    _write_json(
        f"{img_err}.pending.json",
        {"v": 1, "job_id": "wx2", "created_at": time.time(), "timeout_sec": 60, "pid": 0},
    )
    msg2 = Message(
        id="m_wx2",
        session_id=session.id,
        chatbot_id="wechat_test",
        channel_id="wx_u1",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"[file: {img_err}]",
    )
    await db.save_message(msg2)
    await wbot.render_outgoing_message(msg2)

    os.unlink(f"{img_err}.pending.json")
    _write_json(f"{img_err}.error.json", {"v": 1, "job_id": "wx2", "error": "quota exceeded"})

    await _wait_until(lambda: any("附件生成失败 (failed_card.png): quota exceeded" in t for t in sent_texts))

    # Inline in-order voice on WeChat (`_voice_inline_wait_sec > 0`): waits inline and delivers before trailing text!
    wx_events: list[str] = []
    wbot._voice_inline_wait_sec = 2.0
    wbot._voice_inline_anchor_delay_sec = 0.04
    wbot.send_text_chunks = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda cid, chunks, m: [wx_events.append(f"text:{c.strip()}") for c in chunks]
    )
    wbot.send_file_segment = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda cid, fpath, m: wx_events.append(f"file:{fpath}")
    )

    wav_wx = os.path.join(staging_dir, "wx_clip.wav")
    _write_json(
        f"{wav_wx}.pending.json",
        {"v": 1, "job_id": "wx_v1", "created_at": time.time(), "timeout_sec": 60, "pid": 0},
    )

    async def _finish_wx_voice() -> None:
        await asyncio.sleep(0.10)
        with open(wav_wx, "wb") as fh:
            fh.write(b"RIFF_WX_VOICE")
        os.unlink(f"{wav_wx}.pending.json")

    bg_wx_v = asyncio.create_task(_finish_wx_voice())
    msg3 = Message(
        id="m_wx3_voice",
        session_id=session.id,
        chatbot_id="wechat_test",
        channel_id="wx_u1",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"双语行\n[voice: {wav_wx}]\n后文",
    )
    await db.save_message(msg3)
    await wbot.render_outgoing_message(msg3)
    await bg_wx_v

    assert wx_events == ["text:双语行", f"file:{os.path.realpath(wav_wx)}", "text:后文"]
    saved_wx_v = await db.get_message("m_wx3_voice")
    assert saved_wx_v is not None
    assert "deferred_media" not in (saved_wx_v.metadata or {})


@pytest.mark.asyncio
async def test_ready_before_render_and_stale_mtime_gate(tmp_path: Any) -> None:
    """Verify ready-before-render sends synchronously while stale reused filenames wait for fresh mtime."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="s_mtime", title="Mtime", chatbot_id="discord_test", channel_id="1003")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)
    bot = _make_discord_bot(gw)

    sent_files: list[bytes] = []
    placeholders: list[str] = []

    async def fake_send_file(cid: str, fpath: str, m: Message) -> None:
        _ = (cid, m)
        with open(fpath, "rb") as fh:
            sent_files.append(fh.read())

    async def fake_placeholder(cid: str, seg: dict[str, Any], m: Message, marker: dict[str, Any]) -> Any:
        _ = (cid, seg, m)
        placeholders.append(sanitize_placeholder(marker))
        h = MagicMock(id=777, channel=MagicMock(id=1003))
        h.edit = AsyncMock(side_effect=lambda **kw: sent_files.append(kw["attachments"][0].fp.read()))
        return h

    bot.send_file_segment = AsyncMock(side_effect=fake_send_file)  # type: ignore[method-assign]
    bot.send_deferred_placeholder = AsyncMock(side_effect=fake_placeholder)  # type: ignore[method-assign]

    # Case A: target already ready with current mtime while .pending.json still exists
    fast_target = os.path.join(staging_dir, "fast.png")
    now = time.time()
    with open(fast_target, "wb") as fh:
        fh.write(b"FAST_READY")
    _write_json(
        f"{fast_target}.pending.json",
        {"v": 1, "job_id": "j_fast", "created_at": now - 1.0, "timeout_sec": 60, "pid": 0},
    )
    msg_a = Message(
        id="ma",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="1003",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"[file: {fast_target}]",
    )
    await db.save_message(msg_a)
    await bot.render_outgoing_message(msg_a)

    assert placeholders == []
    assert sent_files == [b"FAST_READY"]
    assert not os.path.exists(f"{fast_target}.pending.json")

    # Case B: reused filename has old mtime < created_at - 0.05 -> must wait for fresh write!
    reused_target = os.path.join(staging_dir, "reused.png")
    with open(reused_target, "wb") as fh:
        fh.write(b"STALE_TURN_1")
    old_ts = time.time() - 20.0
    os.utime(reused_target, (old_ts, old_ts))
    new_created = time.time()
    _write_json(
        f"{reused_target}.pending.json",
        {"v": 1, "job_id": "j_reused", "created_at": new_created, "timeout_sec": 60, "pid": 0},
    )

    msg_b = Message(
        id="mb",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="1003",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"[file: {reused_target}]",
    )
    await db.save_message(msg_b)
    await bot.render_outgoing_message(msg_b)

    # Must NOT have delivered STALE_TURN_1 synchronously; placeholder sent instead.
    assert placeholders == [DEFAULT_DEFERRED_PLACEHOLDER]
    assert sent_files == [b"FAST_READY"]

    await asyncio.sleep(0.06)
    assert sent_files == [b"FAST_READY"]

    # Now producer writes fresh TURN_2 file and removes .pending.json
    with open(reused_target, "wb") as fh:
        fh.write(b"FRESH_TURN_2")
    os.unlink(f"{reused_target}.pending.json")

    await _wait_until(lambda: len(sent_files) == 2)
    assert sent_files[1] == b"FRESH_TURN_2"


@pytest.mark.asyncio
async def test_fast_error_before_render_preserves_trailing_ui_and_sanitizes(tmp_path: Any) -> None:
    """Verify [C-2] & Pitfall #14: fast .error.json skips fuzzy match/self-healing and delivers trailing UI."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="s_err", title="Err", chatbot_id="discord_test", channel_id="1004")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    # Sibling file that would fuzzy-match figure_12.png if .error.json were not checked first.
    with open(os.path.join(staging_dir, "figure_11.png"), "wb") as fh:
        fh.write(b"WRONG_FIGURE_11")

    failed_target = os.path.join(staging_dir, "figure_12.png")
    raw_secret_traceback = (
        "Traceback (most recent call last):\n"
        '  File "/app/worker.py", line 42, in run\n'
        "RuntimeError: upstream 429 GEMINI_API_KEY=AIzaSySecretKeyValue1234567890 "
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 " + ("x" * 260)
    )
    _write_json(
        f"{failed_target}.error.json",
        {"v": 1, "job_id": "j_err", "target": failed_target, "exit_code": 1, "error": raw_secret_traceback},
    )

    bot = _make_discord_bot(gw)
    sent_texts: list[str] = []
    questions_sent: list[str] = []
    bot.send_text_chunks = AsyncMock(side_effect=lambda cid, chunks, m: sent_texts.extend(chunks))  # type: ignore[method-assign]
    bot.send_file_segment = AsyncMock()  # type: ignore[method-assign]
    bot.send_question_segment = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda cid, q, choices, m, **kw: questions_sent.append(q)
    )
    bot._inject_self_healing_notification = AsyncMock()  # type: ignore[method-assign]

    msg = Message(
        id="m_fast_err",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="1004",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=(
            f"Summary text.\n[file: {failed_target}]\n"
            "```ui\ntype: select\nprompt: Next step?\noptions:\n  - Retry\n  - Continue\n```"
        ),
    )
    await db.save_message(msg)
    await bot.render_outgoing_message(msg)

    bot.send_file_segment.assert_not_awaited()
    bot._inject_self_healing_notification.assert_not_awaited()
    assert not os.path.exists(f"{failed_target}.error.json")
    assert questions_sent == ["Next step?"]

    failure_notices = [t for t in sent_texts if "⚠️ 附件生成失败 (figure_12.png):" in t]
    assert len(failure_notices) == 1
    notice = failure_notices[0]
    assert "Traceback" not in notice
    assert "AIzaSySecretKeyValue" not in notice
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in notice
    assert "[REDACTED]" in notice

    saved = await db.get_message(msg.id)
    assert saved is not None and saved.status == MessageStatus.DELIVERED

    # Direct unit checks on placeholder & error bounds
    assert len(sanitize_error_reason("E" * 500)) <= 200
    assert len(sanitize_placeholder({"placeholder": "P" * 300})) <= 120
    assert sanitize_placeholder({}) == DEFAULT_DEFERRED_PLACEHOLDER
    assert sanitize_placeholder({"placeholder": "   "}) == DEFAULT_DEFERRED_PLACEHOLDER


@pytest.mark.asyncio
async def test_superseded_job_id_and_stale_error_json_ignored(tmp_path: Any) -> None:
    """Verify [C-R2.1] stale .error.json job_id filtering and superseded .pending.json handling."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="s_sup", title="Sup", chatbot_id="discord_test", channel_id="1005")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)
    bot = _make_discord_bot(gw)

    # Subcase A: stale .error.json with old job_id is ignored while active job finishes
    t1 = os.path.join(staging_dir, "plot.png")
    _write_json(f"{t1}.pending.json", {"v": 1, "job_id": "new_job", "created_at": time.time(), "pid": 0})
    _write_json(f"{t1}.error.json", {"v": 1, "job_id": "stale_old_job", "error": "old error"})

    h1 = MagicMock(id=11, channel=MagicMock(id=1005))
    h1.edit = AsyncMock()
    bot.send_deferred_placeholder = AsyncMock(return_value=h1)  # type: ignore[method-assign]

    m1 = Message(
        id="m_sup_1",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="1005",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"[file: {t1}]",
    )
    await db.save_message(m1)
    await bot.render_outgoing_message(m1)
    await asyncio.sleep(0.06)
    h1.edit.assert_not_awaited()

    with open(t1, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\nOK")
    os.unlink(f"{t1}.pending.json")
    os.unlink(f"{t1}.error.json")
    await _wait_until(lambda: h1.edit.await_count == 1)
    assert h1.edit.await_args.kwargs.get("content") is None

    # Subcase B: .pending.json replaced by a newer job_id -> old watcher fails without deleting new pending
    t2 = os.path.join(staging_dir, "overwrite.png")
    _write_json(f"{t2}.pending.json", {"v": 1, "job_id": "job_v1", "created_at": time.time(), "pid": 0})
    h2 = MagicMock(id=22, channel=MagicMock(id=1005))
    h2.edit = AsyncMock()
    bot.send_deferred_placeholder = AsyncMock(return_value=h2)  # type: ignore[method-assign]

    m2 = Message(
        id="m_sup_2",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="1005",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"[file: {t2}]",
    )
    await db.save_message(m2)
    await bot.render_outgoing_message(m2)

    _write_json(f"{t2}.pending.json", {"v": 1, "job_id": "job_v2", "created_at": time.time(), "pid": 0})
    await _wait_until(lambda: h2.edit.await_count == 1)
    assert "已被新任务取代" in h2.edit.await_args.kwargs["content"]
    assert os.path.exists(f"{t2}.pending.json")
    with open(f"{t2}.pending.json", encoding="utf-8") as fh:
        assert json.load(fh)["job_id"] == "job_v2"


@pytest.mark.asyncio
async def test_dead_worker_pid_backfill_and_grace_period(tmp_path: Any) -> None:
    """Verify pid=0 is tolerated initially, and a backfilled dead PID fails after grace period."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="s_pid", title="PID", chatbot_id="discord_test", channel_id="1006")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)
    bot = _make_discord_bot(gw)

    target = os.path.join(staging_dir, "crash_render.png")
    _write_json(
        f"{target}.pending.json",
        {"v": 1, "job_id": "j_pid", "created_at": time.time(), "timeout_sec": 120, "pid": 0},
    )
    handle = MagicMock(id=33, channel=MagicMock(id=1006))
    handle.edit = AsyncMock()
    bot.send_deferred_placeholder = AsyncMock(return_value=handle)  # type: ignore[method-assign]

    msg = Message(
        id="m_pid",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="1006",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"[file: {target}]",
    )
    await db.save_message(msg)
    await bot.render_outgoing_message(msg)

    # pid=0 stays pending without false-positive dead detection
    await asyncio.sleep(0.12)
    handle.edit.assert_not_awaited()

    # Spawn a real short-lived subprocess and wait for it to exit so we have a genuinely dead PID
    dead_proc = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_proc.wait(timeout=5)
    assert not is_pid_alive(dead_proc.pid)

    _write_json(
        f"{target}.pending.json",
        {"v": 1, "job_id": "j_pid", "created_at": time.time(), "timeout_sec": 120, "pid": dead_proc.pid},
    )

    await _wait_until(lambda: handle.edit.await_count == 1)
    assert "后台生成进程异常退出" in handle.edit.await_args.kwargs["content"]
    saved = await db.get_message(msg.id)
    assert saved is not None and saved.metadata["deferred_media"][0]["state"] == "failed"


@pytest.mark.asyncio
async def test_timeout_clamping_and_worker_termination(tmp_path: Any) -> None:
    """Verify timeout clamping [10, 600] and SIGTERM termination on expired worker."""
    assert clamp_timeout_sec({"timeout_sec": 1}) == 10.0
    assert clamp_timeout_sec({"timeout_sec": 9999}) == 600.0
    assert clamp_timeout_sec({}) == 240.0

    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="s_to", title="Timeout", chatbot_id="discord_test", channel_id="1007")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)
    bot = _make_discord_bot(gw)

    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        target = os.path.join(staging_dir, "slow.png")
        _write_json(
            f"{target}.pending.json",
            {
                "v": 1,
                "job_id": "j_slow",
                "created_at": time.time() - 25.0,
                "timeout_sec": 10,
                "pid": sleeper.pid,
            },
        )
        handle = MagicMock(id=44, channel=MagicMock(id=1007))
        handle.edit = AsyncMock()
        bot.send_deferred_placeholder = AsyncMock(return_value=handle)  # type: ignore[method-assign]

        msg = Message(
            id="m_to",
            session_id=session.id,
            chatbot_id="discord_test",
            channel_id="1007",
            sender="Assistant",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content=f"[file: {target}]",
        )
        await db.save_message(msg)
        await bot.render_outgoing_message(msg)

        await _wait_until(lambda: handle.edit.await_count == 1)
        assert "生成超时" in handle.edit.await_args.kwargs["content"]
        sleeper.wait(timeout=5)
        assert not is_pid_alive(sleeper.pid)
        assert not os.path.exists(f"{target}.pending.json")
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait(timeout=5)


@pytest.mark.asyncio
async def test_cancellation_split_stop_vs_clear_and_delete_session(tmp_path: Any) -> None:
    """Verify [C-8]/[C-R2.2]: stop() preserves worker & pending state, while clear/delete_session kills worker."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(
        session_id="s_cancel", title="Cancel", chatbot_id="discord_test", channel_id="1008"
    )
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)
    bot = _make_discord_bot(gw)
    bot.bot.close = AsyncMock()  # type: ignore[method-assign]

    worker1 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    worker2 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        t1 = os.path.join(staging_dir, "keep_on_stop.png")
        _write_json(
            f"{t1}.pending.json",
            {"v": 1, "job_id": "j_stop", "created_at": time.time(), "timeout_sec": 300, "pid": worker1.pid},
        )
        h1 = MagicMock(id=51, channel=MagicMock(id=1008))
        h1.edit = AsyncMock()
        bot.send_deferred_placeholder = AsyncMock(return_value=h1)  # type: ignore[method-assign]

        m1 = Message(
            id="m_cancel_1",
            session_id=session.id,
            chatbot_id="discord_test",
            channel_id="1008",
            sender="Assistant",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content=f"[file: {t1}]",
        )
        await db.save_message(m1)
        await bot.render_outgoing_message(m1)
        await asyncio.sleep(0.05)

        # 1. Chatbot.stop() cancels ONLY in-memory task; worker1 stays alive and DB state remains "pending"
        bot.stop()
        await asyncio.sleep(0.05)
        assert worker1.poll() is None
        saved = await db.get_message(m1.id)
        assert saved is not None and saved.metadata["deferred_media"][0]["state"] == "pending"

        # 2. clear_session_by_channel kills worker1 and marks metadata "cancelled"
        await bot.clear_session_by_channel("1008")
        worker1.wait(timeout=5)
        saved_after_clear = await db.get_message(m1.id)
        assert saved_after_clear is not None
        assert saved_after_clear.metadata["deferred_media"][0]["state"] == "cancelled"

        # 3. Gateway.delete_session kills worker2 BEFORE removing workspace_dir
        sess2 = await gw.create_session(session_id="s_del", title="Del", chatbot_id="discord_test", channel_id="1009")
        staging2 = os.path.join(gw.workspace_config.sessions_dir, sess2.workspace_name)
        os.makedirs(staging2, exist_ok=True)
        t2 = os.path.join(staging2, "delete_me.png")
        # Store pid=0 in DB metadata and real PID only in on-disk .pending.json to verify pre-rmtree read!
        _write_json(
            f"{t2}.pending.json",
            {"v": 1, "job_id": "j_del", "created_at": time.time(), "timeout_sec": 300, "pid": worker2.pid},
        )
        m2 = Message(
            id="m_cancel_2",
            session_id=sess2.id,
            chatbot_id="discord_test",
            channel_id="1009",
            sender="Assistant",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content=f"[file: {t2}]",
            metadata={
                "deferred_media": [
                    {
                        "id": "j_del",
                        "path": t2,
                        "kind": "file",
                        "state": "pending",
                        "marker": {"job_id": "j_del", "pid": 0, "created_at": time.time()},
                        "handle": None,
                    }
                ]
            },
        )
        await db.save_message(m2)
        await gw.delete_session(sess2.id)
        worker2.wait(timeout=5)
        assert not os.path.exists(staging2)
    finally:
        for p in (worker1, worker2):
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)


@pytest.mark.asyncio
async def test_startup_recovery_partial_message_after_wait_until_ready(tmp_path: Any) -> None:
    """Verify [C-3]/[C-R2.3] startup recovery awaits wait_until_ready and edits PartialMessage in place."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="s_rec", title="Rec", chatbot_id="discord_test", channel_id="2001")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    ready_img = os.path.join(staging_dir, "recovered.png")
    with open(ready_img, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\nRECOVERED")

    recent_msg = Message(
        id="m_recent",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="2001",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"[file: {ready_img}]",
        timestamp=time.time() - 60.0,
        status=MessageStatus.DELIVERED,
        metadata={
            "deferred_media": [
                {
                    "id": "j_rec",
                    "path": ready_img,
                    "raw_path": ready_img,
                    "kind": "file",
                    "state": "pending",
                    "marker": {"job_id": "j_rec", "created_at": time.time() - 60.0, "timeout_sec": 240, "pid": 0},
                    "handle": {"channel_id": "2001", "message_id": "88888"},
                }
            ]
        },
    )
    stale_old_msg = Message(
        id="m_older_than_24h",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="2001",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="[file: ancient.png]",
        timestamp=time.time() - 90000.0,
        status=MessageStatus.DELIVERED,
        metadata={
            "deferred_media": [
                {
                    "id": "j_old",
                    "path": os.path.join(staging_dir, "ancient.png"),
                    "kind": "file",
                    "state": "pending",
                    "marker": {"job_id": "j_old", "created_at": time.time() - 90000.0},
                    "handle": {"channel_id": "2001", "message_id": "11111"},
                }
            ]
        },
    )
    await db.save_message(recent_msg)
    await db.save_message(stale_old_msg)

    bot = _make_discord_bot(gw)
    ready_flag = False

    async def fake_wait_ready() -> None:
        nonlocal ready_flag
        ready_flag = True

    bot.bot.is_ready = MagicMock(side_effect=lambda: ready_flag)  # type: ignore[method-assign]
    bot.bot.wait_until_ready = AsyncMock(side_effect=fake_wait_ready)  # type: ignore[method-assign]

    partial_msg = MagicMock()
    partial_msg.edit = AsyncMock()
    partial_chan = MagicMock()
    partial_chan.get_partial_message = MagicMock(return_value=partial_msg)
    bot.bot.get_partial_messageable = MagicMock(return_value=partial_chan)  # type: ignore[method-assign]

    await bot._recover_deferred_media()

    async def _recovered() -> bool:
        m = await db.get_message("m_recent")
        return bool(m and m.metadata["deferred_media"][0]["state"] == "delivered")

    await _wait_until(_recovered)
    bot.bot.wait_until_ready.assert_awaited()
    bot.bot.get_partial_messageable.assert_called_once_with(2001)
    partial_chan.get_partial_message.assert_called_once_with(88888)
    partial_msg.edit.assert_awaited_once()
    assert partial_msg.edit.await_args.kwargs["content"] is None

    # >24h message was ignored by startup scan
    old_after = await db.get_message("m_older_than_24h")
    assert old_after is not None and old_after.metadata["deferred_media"][0]["state"] == "pending"


@pytest.mark.asyncio
async def test_multi_attachment_deepcopy_subdir_lookup_and_sidecar_exclusions(tmp_path: Any) -> None:
    """Verify [C-6] basename subdir lookup, [C-9] concurrent deepcopy updates, and sidecar exclusions."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="s_multi", title="Multi", chatbot_id="discord_test", channel_id="3001")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    sub_dir = os.path.join(staging_dir, "nested_charts")
    os.makedirs(sub_dir, exist_ok=True)

    # Create sidecar/temp files in staging_dir to test exclusion filters
    sub_target = os.path.join(sub_dir, "figure_sub.png")
    direct_target = os.path.join(staging_dir, "figure_top.png")
    _write_json(f"{sub_target}.pending.json", {"v": 1, "job_id": "j_sub", "created_at": time.time(), "pid": 0})
    _write_json(f"{direct_target}.pending.json", {"v": 1, "job_id": "j_top", "created_at": time.time(), "pid": 0})
    _write_json(os.path.join(staging_dir, "x.error.json"), {"error": "err"})
    _write_json(os.path.join(staging_dir, "x.job.json"), {"job": 1})
    with open(os.path.join(staging_dir, ".deferred-abc.part"), "wb") as fh:
        fh.write(b"PART")
    with open(os.path.join(staging_dir, "keep_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("visible artifact")

    for excluded in ("a.pending.json", "a.error.json", "a.job.json", "tmp.part"):
        assert is_internal_staging_name(excluded)
    snap = render_staging_artifacts(staging_dir)
    assert "keep_report.txt" in snap
    for suffix in (".pending.json", ".error.json", ".job.json", ".part"):
        assert suffix not in snap

    bot = _make_discord_bot(gw)
    listing = await bot._get_staging_files_list(session.id)
    assert "keep_report.txt" in listing
    assert all(not item.endswith((".pending.json", ".error.json", ".job.json", ".part")) for item in listing)

    handles = [MagicMock(id=101, channel=MagicMock(id=3001)), MagicMock(id=102, channel=MagicMock(id=3001))]
    for h in handles:
        h.edit = AsyncMock()
    idx_counter = 0

    async def next_handle(*_a: Any, **_kw: Any) -> Any:
        nonlocal idx_counter
        h = handles[idx_counter]
        idx_counter += 1
        return h

    bot.send_deferred_placeholder = AsyncMock(side_effect=next_handle)  # type: ignore[method-assign]

    persisted_snapshots: list[dict[str, Any]] = []
    orig_update = db.update_message_metadata

    async def spy_update(mid: str, metadata: dict[str, Any]) -> None:
        persisted_snapshots.append(metadata)
        await orig_update(mid, metadata)

    db.update_message_metadata = spy_update  # type: ignore[method-assign]

    # Note: figure_sub.png is referenced ONLY by basename while living in nested_charts/ [C-6]!
    msg = Message(
        id="m_multi",
        session_id=session.id,
        chatbot_id="discord_test",
        channel_id="3001",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=f"[file: figure_sub.png]\n[file: {direct_target}]",
    )
    await db.save_message(msg)
    await bot.render_outgoing_message(msg)

    # Complete both files simultaneously
    for p in (sub_target, direct_target):
        with open(p, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\nMULTI")
        os.unlink(f"{p}.pending.json")

    async def _both_delivered() -> bool:
        m = await db.get_message(msg.id)
        if not m:
            return False
        states = [e["state"] for e in m.metadata.get("deferred_media", [])]
        return states == ["delivered", "delivered"]

    await _wait_until(_both_delivered)
    assert len(persisted_snapshots) >= 4
    # Verify deep-copied snapshots did not mutate earlier captured dicts in place
    assert persisted_snapshots[0]["deferred_media"][0]["state"] == "pending"


@pytest.mark.asyncio
async def test_analyze_media_waits_for_pending_and_aborts_on_error_sidecar(tmp_path: Any) -> None:
    """Verify [C-5] analyze_media waits for in-progress .pending.json and exits cleanly on .error.json."""
    gw, _ = await _make_gateway(tmp_path)
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=LLMResponse(content="Detailed chart description."))
    gw.context.get_llm = MagicMock(return_value=mock_llm)  # type: ignore[method-assign]

    ctx = ToolContext(session_id="s_media", session_workspace="ws_m", gateway=gw)
    pending_img = str(tmp_path / "delayed_chart.png")
    _write_json(
        f"{pending_img}.pending.json",
        {"v": 1, "job_id": "jm1", "created_at": time.time(), "timeout_sec": 30, "pid": 0},
    )

    async def _producer() -> None:
        await asyncio.sleep(0.12)
        with open(pending_img, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\nREADY")
        os.unlink(f"{pending_img}.pending.json")

    prod_task = asyncio.create_task(_producer())
    res = await analyze_media(path=pending_img, prompt="Inspect chart", context=ctx)
    await prod_task
    assert res == "Detailed chart description."

    # Error sidecar case
    broken_img = str(tmp_path / "broken_chart.png")
    _write_json(f"{broken_img}.error.json", {"v": 1, "job_id": "jm2", "error": "upstream 500"})
    err_res = await analyze_media(path=broken_img, prompt="Inspect broken", context=ctx)
    assert "failed to generate" in err_res and "upstream 500" in err_res


@pytest.mark.asyncio
async def test_capability_gating_shell_env_and_turn_executor_pivot_refresh(tmp_path: Any) -> None:
    """Verify [C-4] capability flags, KESOKU_DEFERRED_MEDIA env gating, and TurnExecutor pivot refresh."""
    assert Chatbot.supports_deferred_media is False
    assert DiscordChatbot.supports_deferred_media is True
    assert WechatChatbot.supports_deferred_media is True
    assert CLIChatbot.supports_deferred_media is False
    assert GoogleChatChatbot.supports_deferred_media is False
    assert CronjobChatbot.supports_deferred_media is False

    gw, _ = await _make_gateway(tmp_path)
    gw.register_deferred_media("discord_primary", True)
    gw.register_deferred_media("cli", False)
    assert gw.supports_deferred_media("discord_primary") is True
    assert gw.supports_deferred_media("cli") is False
    assert gw.supports_deferred_media("unregistered") is False

    # 1. Shell env injection when supports_deferred_media=True and foreground_only=False
    ctx_enabled = ToolContext(
        session_id="s_env",
        session_workspace="ws_env",
        gateway=gw,
        active_jobs=gw.context.active_jobs,
        supports_deferred_media=True,
        foreground_only=False,
    )
    out_enabled = await run_shell_command(
        f"{sys.executable} -c \"import os; print('VAL=' + os.environ.get('KESOKU_DEFERRED_MEDIA', 'NONE'))\"",
        context=ctx_enabled,
    )
    assert "VAL=1" in out_enabled

    # 2. Subagent foreground_only=True MUST NOT receive KESOKU_DEFERRED_MEDIA=1
    ctx_subagent = ToolContext(
        session_id="s_env",
        session_workspace="ws_env",
        gateway=gw,
        active_jobs=gw.context.active_jobs,
        supports_deferred_media=True,
        foreground_only=True,
    )
    with patch.dict(os.environ, {"KESOKU_DEFERRED_MEDIA": "1"}):
        out_sub = await run_shell_command(
            f"{sys.executable} -c \"import os; print('VAL=' + os.environ.get('KESOKU_DEFERRED_MEDIA', 'NONE'))\"",
            context=ctx_subagent,
        )
    assert "VAL=NONE" in out_sub

    # 3. CLI / unsupported chatbot MUST NOT receive KESOKU_DEFERRED_MEDIA=1
    ctx_disabled = ToolContext(
        session_id="s_env",
        session_workspace="ws_env",
        gateway=gw,
        active_jobs=gw.context.active_jobs,
        supports_deferred_media=False,
        foreground_only=False,
    )
    with patch.dict(os.environ, {"KESOKU_DEFERRED_MEDIA": "1"}):
        out_dis = await run_shell_command(
            f"{sys.executable} -c \"import os; print('VAL=' + os.environ.get('KESOKU_DEFERRED_MEDIA', 'NONE'))\"",
            context=ctx_disabled,
        )
    assert "VAL=NONE" in out_dis

    # 4. TurnExecutor resolves capability at turn start AND refreshes after drain_queue_and_pivot
    await gw.create_session("s_pivot", title="Pivot", chatbot_id="cli", channel_id="c_cli")
    m_cli = Message(
        id="u_cli",
        session_id="s_pivot",
        chatbot_id="cli",
        channel_id="c_cli",
        sender="user",
        role=MessageRole.USER,
        content="first from cli",
        status=MessageStatus.PENDING_AGENT,
    )
    m_discord = Message(
        id="u_disc",
        session_id="s_pivot",
        chatbot_id="discord_primary",
        channel_id="c_disc",
        sender="user",
        role=MessageRole.USER,
        content="pivoted from discord",
        status=MessageStatus.PENDING_AGENT,
    )
    await gw.post(m_cli)
    await gw.post(m_discord)

    observed_flags: list[bool] = []
    tool_ctx = ToolContext(session_id="s_pivot", session_workspace="s_pivot", gateway=gw)

    mock_llm = MagicMock()

    async def fake_gen(**_kw: Any) -> LLMResponse:
        observed_flags.append(tool_ctx.supports_deferred_media)
        if len(observed_flags) == 1:
            return LLMResponse(content="")
        return LLMResponse(content="done")

    mock_llm.generate = AsyncMock(side_effect=fake_gen)
    gw.context._llm = mock_llm

    tool_runner = MagicMock()
    tool_runner.tool_context = tool_ctx
    tool_runner.tool_registry.get_tools_list.return_value = []
    executor = TurnExecutor(session_id="s_pivot", gateway=gw, tool_runner=tool_runner, context=gw.context)
    assert executor._resolve_supports_deferred_media("discord_primary") is True
    assert executor._resolve_supports_deferred_media("cli") is False

    pivot_calls = 0
    worker = MagicMock()
    worker.running = True
    worker.queue_empty.return_value = True

    async def fake_pivot(m: Message) -> Message:
        nonlocal pivot_calls
        pivot_calls += 1
        if pivot_calls == 1:
            return m
        return m_discord

    worker.drain_queue_and_pivot = AsyncMock(side_effect=fake_pivot)

    await executor.process_turn(
        current_msg=m_cli,
        worker=worker,
        session_staging_dir=str(tmp_path),
    )
    assert observed_flags == [False, True]
    assert tool_ctx.supports_deferred_media is True


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


# origin: tests/test_round3_fixes.py K7
@pytest.mark.asyncio
async def test_k7_resolve_deferred_fuzzy_pending_wins_over_stale_and_staging_list(tmp_path: Path) -> None:
    db_path = str(tmp_path / "k7.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    sessions_dir = tmp_path / "sessions"
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(sessions_dir)))
    gw = Gateway(context=KesokuContext(config=cfg, db=db))
    sess = await gw.create_session("s_k7", title="K7")
    bot = _DummyChatbot("discord_test", gw)
    staging_dir = Path(bot.get_session_staging_dir(sess.workspace_name))
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Stale completed file from an earlier turn
    stale = staging_dir / "shot_final_old.png"
    stale.write_bytes(b"OLD")
    old_ts = time.time() - 600
    os.utime(stale, (old_ts, old_ts))

    # Active in-flight deferred worker for shotA_final.png
    pending_sidecar = staging_dir / "shotA_final.png.pending.json"
    pending_sidecar.write_text(
        json.dumps({"job_id": "j1", "created_at": time.time(), "timeout_sec": 240}),
        encoding="utf-8",
    )

    # Noise files and excluded directories
    (staging_dir / "background_job_123.stdout").write_text("out", encoding="utf-8")
    (staging_dir / "cmd_output_456.txt").write_text("big", encoding="utf-8")
    (staging_dir / "hist_001.txt").write_text("hist", encoding="utf-8")
    (staging_dir / ".hidden_note").write_text("secret", encoding="utf-8")
    (staging_dir / "context").mkdir()
    (staging_dir / "context" / "80-lesson.md").write_text("ctx", encoding="utf-8")
    (staging_dir / "game").mkdir()
    (staging_dir / "game" / "scene.json").write_text("{}", encoding="utf-8")

    # Typo'd tag [file: shot_final.png] fuzzy-matches active shotA_final.png.pending.json (>= 0.80)
    res = resolve_deferred_sync("shot_final.png", str(staging_dir))
    assert res is not None
    kind, target, marker = res
    assert kind == "pending"
    assert os.path.basename(target) == "shotA_final.png"
    assert marker.get("job_id") == "j1"

    # Staging list shows "(生成中)" first (newer mtime), keeps real artifacts, drops all noise & cap <= 20
    listing = await bot._get_staging_files_list(sess.id)
    assert listing[0] == "shotA_final.png (生成中)"
    assert "shot_final_old.png" in listing
    for forbidden in ("background_job_", "cmd_output_", "hist_", ".hidden", "context", "game", ".pending.json"):
        assert all(forbidden not in entry for entry in listing)


# origin: tests/test_ttft_async_tts_v015.py
def test_voice_inline_wait_attributes() -> None:
    """Verify `_voice_inline_wait_sec` (60.0) & `_voice_inline_anchor_delay_sec` (1.0) defaults, no legacy grace."""
    assert Chatbot._voice_inline_wait_sec == 60.0
    assert WechatChatbot._voice_inline_wait_sec == 60.0
    assert DiscordChatbot._voice_inline_wait_sec == 60.0
    assert Chatbot._voice_inline_anchor_delay_sec == 1.0
    assert WechatChatbot._voice_inline_anchor_delay_sec == 1.0
    assert DiscordChatbot._voice_inline_anchor_delay_sec == 1.0
    assert not hasattr(Chatbot, "_voice_inline_grace_sec")
    assert not hasattr(WechatChatbot, "_voice_inline_grace_sec")
    assert not hasattr(DiscordChatbot, "_voice_inline_grace_sec")


# origin: tests/test_ttft_async_tts_v015.py
@pytest.mark.asyncio
async def test_voice_inline_grace_3_state_dispatch_ready_pending_failed(tmp_path: Any) -> None:
    """Verify 3-state dispatch after bounded inline wait: ready -> inline, pending -> tracker, failed -> notice."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="sv1", title="Voice3State", chatbot_id="discord_v015", channel_id="11")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    bot = _make_discord_bot(gw, chatbot_id="discord_v015")
    bot._voice_inline_wait_sec = 0.4
    bot._voice_inline_anchor_delay_sec = 0.15

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


# origin: tests/test_ttft_async_tts_v015.py
@pytest.mark.asyncio
async def test_voice_inline_waits_in_order_and_handles_phase_b_failures(tmp_path: Any) -> None:
    """Verify Cases D/E/F: Phase-B in-order voice delivery, Phase-B `.error.json`, and inline dead-PID failure (R1)."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(
        session_id="sv_r5", title="VoiceRound5", chatbot_id="discord_v015", channel_id="12"
    )
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    bot = _make_discord_bot(gw, chatbot_id="discord_v015")
    bot.bot.is_ready = MagicMock(return_value=True)  # type: ignore[method-assign]
    bot._voice_inline_wait_sec = 3.0
    bot._voice_inline_anchor_delay_sec = 0.05
    bot._deferred_dead_grace_sec = 0.08

    events: list[str] = []
    partial_msgs: dict[int, MagicMock] = {}

    def _get_partial_msg(mid: int) -> MagicMock:
        if mid not in partial_msgs:
            pm = MagicMock(id=mid)

            async def _on_add(emoji: str, _mid: int = mid) -> None:
                events.append(f"react_add:{_mid}:{emoji}")

            async def _on_rem(emoji: str, _member: Any = None, _mid: int = mid) -> None:
                events.append(f"react_rem:{_mid}:{emoji}")

            pm.add_reaction = AsyncMock(side_effect=_on_add)
            pm.remove_reaction = AsyncMock(side_effect=_on_rem)
            pm.edit = AsyncMock()
            pm.delete = AsyncMock()
            partial_msgs[mid] = pm
        return partial_msgs[mid]

    partial_chan = MagicMock()
    partial_chan.get_partial_message = MagicMock(side_effect=_get_partial_msg)
    bot.bot.get_partial_messageable = MagicMock(return_value=partial_chan)  # type: ignore[method-assign]

    sent_texts: list[tuple[int, str, Any]] = []
    next_msg_id = 5500
    mock_channel = MagicMock(id=12)

    async def fake_channel_send(*args: Any, **kwargs: Any) -> Any:
        nonlocal next_msg_id
        next_msg_id += 1
        text = (args[0] if args else kwargs.get("content", "")).strip()
        sent_texts.append((next_msg_id, text, kwargs.get("reference")))
        events.append(f"text:{text}")
        return MagicMock(id=next_msg_id, channel=mock_channel)

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
        events.append("voice")

    async def fake_send_file_segment(cid: str, fpath: str, message: Message) -> None:
        _ = (cid, message)
        events.append(f"file:{os.path.basename(fpath)}")

    questions_sent: list[str] = []

    async def fake_send_question_segment(cid: str, q: str, choices: list[str], m: Message, **_kw: Any) -> None:
        _ = (cid, choices, m)
        questions_sent.append(q)
        events.append("question")

    bot.send_voice_segment = AsyncMock(side_effect=fake_send_voice_segment)  # type: ignore[method-assign]
    bot.send_file_segment = AsyncMock(side_effect=fake_send_file_segment)  # type: ignore[method-assign]
    bot.send_question_segment = AsyncMock(side_effect=fake_send_question_segment)  # type: ignore[method-assign]

    # --- Case D: Round-5 regression — voice takes 0.35s (> anchor_delay, < wait_sec), delivered IN ORDER ---
    sticker_gif = os.path.join(staging_dir, "04_没问题.gif")
    with open(sticker_gif, "wb") as fh:
        fh.write(b"GIF89a_STICKER")

    wav_d = os.path.join(staging_dir, "win11_bypass.wav")
    _write_json(
        f"{wav_d}.pending.json",
        {"v": 1, "job_id": "jd1", "created_at": time.time(), "timeout_sec": 180, "pid": 0},
    )

    async def _finish_d() -> None:
        await asyncio.sleep(0.35)
        with open(wav_d, "wb") as fh:
            fh.write(b"RIFF_CASE_D")
        os.unlink(f"{wav_d}.pending.json")

    bg_d = asyncio.create_task(_finish_d())
    msg_d = Message(
        id="m_case_d",
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="12",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=(
            f"> 日文\n<spoiler>中文</spoiler>\n[voice: {wav_d}]\n正文段落。\n[file: {sticker_gif}]\n"
            "```ui\ntype: select\nprompt: 下一步？\noptions:\n  - 继续\n```"
        ),
    )
    await db.save_message(msg_d)
    await bot.render_outgoing_message(msg_d)
    await bg_d

    bilingual_d_id = sent_texts[0][0]
    delivery_events_d = [ev for ev in events if not ev.startswith("react_")]
    assert delivery_events_d == [
        "text:> 日文\n||中文||",
        "voice",
        "text:正文段落。",
        "file:04_没问题.gif",
        "question",
    ]
    assert voice_calls == [("12", os.path.realpath(wav_d), None)]
    partial_msgs[bilingual_d_id].add_reaction.assert_awaited_once_with("🎤")
    partial_msgs[bilingual_d_id].remove_reaction.assert_awaited_once()
    assert partial_msgs[bilingual_d_id].remove_reaction.await_args.args[0] == "🎤"
    assert events.index(f"react_add:{bilingual_d_id}:🎤") < events.index("voice")
    assert events.index("voice") < events.index(f"react_rem:{bilingual_d_id}:🎤")
    assert events.index(f"react_rem:{bilingual_d_id}:🎤") < events.index("text:正文段落。")

    saved_d = await db.get_message("m_case_d")
    assert saved_d is not None
    assert "deferred_media" not in (saved_d.metadata or {})
    assert saved_d.status == MessageStatus.DELIVERED

    # --- Case E: Worker writes `.error.json` during Phase B -> 🎤->⚠️ + notice BEFORE trailing text ---
    events.clear()
    sent_texts.clear()
    voice_calls.clear()
    questions_sent.clear()

    wav_e = os.path.join(staging_dir, "phase_b_err.wav")
    _write_json(
        f"{wav_e}.pending.json",
        {"v": 1, "job_id": "je_b", "created_at": time.time(), "timeout_sec": 180, "pid": 0},
    )

    async def _fail_e() -> None:
        await asyncio.sleep(0.25)
        _write_json(f"{wav_e}.error.json", {"v": 1, "job_id": "je_b", "error": "CUDA OOM"})
        os.unlink(f"{wav_e}.pending.json")

    bg_e = asyncio.create_task(_fail_e())
    msg_e = Message(
        id="m_case_e",
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="12",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=(
            f"> 日文\n<spoiler>中文</spoiler>\n[voice: {wav_e}]\n正文段落。\n"
            "```ui\ntype: select\nprompt: 报错后继续？\noptions:\n  - 好的\n```"
        ),
    )
    await db.save_message(msg_e)
    await bot.render_outgoing_message(msg_e)
    await bg_e

    bilingual_e_id = sent_texts[0][0]
    partial_msgs[bilingual_e_id].remove_reaction.assert_awaited_once()
    assert partial_msgs[bilingual_e_id].remove_reaction.await_args.args[0] == "🎤"
    assert [c.args[0] for c in partial_msgs[bilingual_e_id].add_reaction.await_args_list] == ["🎤", "⚠️"]
    texts_e = [t for _, t, _ in sent_texts]
    assert texts_e == ["> 日文\n||中文||", "⚠️ 语音生成失败（CUDA OOM）", "正文段落。"]
    assert questions_sent == ["报错后继续？"]
    saved_e = await db.get_message("m_case_e")
    assert saved_e is not None
    assert "deferred_media" not in (saved_e.metadata or {})

    # --- Case F (R1): Dead worker PID during inline wait -> unlinks pending + fails inline IN ORDER ---
    events.clear()
    sent_texts.clear()
    voice_calls.clear()
    questions_sent.clear()

    wav_f = os.path.join(staging_dir, "dead_pid_voice.wav")
    _write_json(
        f"{wav_f}.pending.json",
        {"v": 1, "job_id": "jf_dead", "created_at": time.time(), "timeout_sec": 180, "pid": 999999},
    )
    msg_f = Message(
        id="m_case_f",
        session_id=session.id,
        chatbot_id="discord_v015",
        channel_id="12",
        sender="Assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=(
            f"> 日文\n<spoiler>中文</spoiler>\n[voice: {wav_f}]\n正文段落。\n"
            "```ui\ntype: select\nprompt: 进程异常后继续？\noptions:\n  - 好的\n```"
        ),
    )
    await db.save_message(msg_f)
    with patch("kesoku.gateway.chatbot.deferred_media.is_pid_alive", return_value=False):
        await bot.render_outgoing_message(msg_f)

    assert not os.path.exists(f"{wav_f}.pending.json")
    bilingual_f_id = sent_texts[0][0]
    partial_msgs[bilingual_f_id].remove_reaction.assert_awaited_once()
    assert partial_msgs[bilingual_f_id].remove_reaction.await_args.args[0] == "🎤"
    assert [c.args[0] for c in partial_msgs[bilingual_f_id].add_reaction.await_args_list] == ["🎤", "⚠️"]
    texts_f = [t for _, t, _ in sent_texts]
    assert texts_f == ["> 日文\n||中文||", "⚠️ 语音生成失败（后台生成进程异常退出）", "正文段落。"]
    assert questions_sent == ["进程异常后继续？"]
    saved_f = await db.get_message("m_case_f")
    assert saved_f is not None
    assert "deferred_media" not in (saved_f.metadata or {})


# origin: tests/test_ttft_async_tts_v015.py
@pytest.mark.asyncio
async def test_voice_anchor_top_of_message_falls_back_to_inbound_user_msg_not_prior_turn(tmp_path: Any) -> None:
    """Verify top-of-message `[voice:]` anchors to triggering user `discord_message_id`, never prior turn's reply."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="sv2", title="TopVoice", chatbot_id="discord_v015", channel_id="22")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    bot = _make_discord_bot(gw, chatbot_id="discord_v015")
    bot._voice_inline_wait_sec = 0.02
    bot._voice_inline_anchor_delay_sec = 0.0

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


# origin: tests/test_ttft_async_tts_v015.py
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

    bot = _make_discord_bot(gw, chatbot_id="discord_v015")
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


# origin: tests/test_ttft_async_tts_v015.py
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


# origin: tests/test_ttft_async_tts_v015.py
@pytest.mark.asyncio
async def test_multi_voice_same_message_anchors_and_remaining_timeout_cap(tmp_path: Any) -> None:
    """Verify two `[voice:]` tags in one message anchor to their respective preceding chunks & cap remaining timeout."""
    gw, db = await _make_gateway(tmp_path)
    session = await gw.create_session(session_id="sv_m", title="MultiVoice", chatbot_id="discord_v015", channel_id="55")
    staging_dir = os.path.join(gw.workspace_config.sessions_dir, session.workspace_name)
    os.makedirs(staging_dir, exist_ok=True)

    bot = _make_discord_bot(gw, chatbot_id="discord_v015")
    bot._voice_inline_wait_sec = 60.0  # Full 60.0s inline wait, capped by remaining marker timeout (< 0.5s)!

    seen_caps: list[float] = []

    async def _spy_wait(candidates: list[str], **kwargs: Any) -> tuple[str | None, str | None]:
        _ = candidates
        cap = float(kwargs["max_wait_cap"])
        seen_caps.append(cap)
        await asyncio.sleep(min(cap, 0.2))
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
    # Markers near their 10s clamped timeout -> remaining timeouts (~0.15s, ~0.19s) << 60.0s and < 1.0s!
    _write_json(f"{w1}.pending.json", {"v": 1, "job_id": "jw1", "created_at": now - 9.85, "timeout_sec": 10, "pid": 0})
    _write_json(f"{w2}.pending.json", {"v": 1, "job_id": "jw2", "created_at": now - 9.65, "timeout_sec": 10, "pid": 0})

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

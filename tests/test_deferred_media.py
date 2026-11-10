"""Unit tests for the generic deferred media attachment protocol."""

import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from kesoku.agent.llm import LLMResponse
from kesoku.agent.state_snapshot import is_internal_staging_name, render_staging_artifacts
from kesoku.agent.tools.media import analyze_media
from kesoku.agent.tools.registry import ToolContext
from kesoku.agent.tools.shell import run_shell_command
from kesoku.agent.turn_executor import TurnExecutor
from kesoku.config import DiscordConfig, KesokuConfig, WechatConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import AsyncDatabaseManager, Message
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.chatbot.cli_bot import CLIChatbot
from kesoku.gateway.chatbot.cronjob import CronjobChatbot
from kesoku.gateway.chatbot.deferred_media import (
    DEFAULT_DEFERRED_PLACEHOLDER,
    clamp_timeout_sec,
    is_pid_alive,
    sanitize_error_reason,
    sanitize_placeholder,
)
from kesoku.gateway.chatbot.discord.adapter import DiscordChatbot
from kesoku.gateway.chatbot.google_chat.adapter import GoogleChatChatbot
from kesoku.gateway.chatbot.wechat.adapter import WechatChatbot
from kesoku.gateway.gateway import Gateway


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


async def _make_gateway(tmp_path: Any) -> tuple[Gateway, AsyncDatabaseManager]:
    sessions_dir = str(tmp_path / "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    db_path = str(tmp_path / "kesoku.db")
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

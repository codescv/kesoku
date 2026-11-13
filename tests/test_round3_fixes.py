"""Comprehensive unit and regression tests for Round 3 Kesoku fixes (K1-K10)."""

import asyncio
import json
import os
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import aiohttp
import discord
import pytest
from google.genai import types

from kesoku.agent.compressor import HistoryCompressor
from kesoku.agent.history import build_history
from kesoku.agent.llm import BaseLLM, GeminiLLM, LLMResponse, _drain_blocking_stream
from kesoku.agent.prompt import build_sys_prompt
from kesoku.agent.tools import ActiveJobsRegistry
from kesoku.agent.tools.file import _resolve_role_sync
from kesoku.agent.tools.media import analyze_media
from kesoku.agent.tools.memory import _resolve_role
from kesoku.agent.tools.registry import ToolContext
from kesoku.agent.tools.shell import run_shell_command
from kesoku.agent.turn_executor import TurnExecutor
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import (
    DiscordConfig,
    GeminiConfig,
    GoogleChatConfig,
    KesokuConfig,
    WechatConfig,
    WorkspaceConfig,
)
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, Session
from kesoku.gateway.chatbot.base import (
    ChannelUnavailableError,
    Chatbot,
    DeliveryAbortedError,
    InboundMessageAttachment,
    InboundMessageDTO,
)
from kesoku.gateway.chatbot.deferred_media import resolve_deferred_sync
from kesoku.gateway.chatbot.discord import DiscordChatbot
from kesoku.gateway.chatbot.discord import persistent_ui as pui
from kesoku.gateway.gateway import Gateway


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


def _make_discord_http_exc(status: int, code: int = 0, text: str = "err") -> discord.HTTPException:
    resp = MagicMock()
    resp.status = status
    resp.reason = text
    exc = discord.HTTPException(resp, text)
    exc.status = status
    exc.code = code
    return exc


def _make_discord_not_found(code: int = 10062, text: str = "Unknown interaction") -> discord.NotFound:
    resp = MagicMock()
    resp.status = 404
    resp.reason = "Not Found"
    exc = discord.NotFound(resp, {"code": code, "message": text})
    exc.code = code
    return exc


def _make_discord_server_error(status: int = 503, text: str = "Service Unavailable") -> discord.DiscordServerError:
    resp = MagicMock()
    resp.status = status
    resp.reason = text
    exc = discord.DiscordServerError(resp, text)
    exc.status = status
    return exc


# ---------------------------------------------------------------------------
# K1: Input-blocked prompts (detect, don't retry, filter history, unbrick)
# ---------------------------------------------------------------------------


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
    from kesoku.agent.llm import ToolCallRequest

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

    with patch("kesoku.agent.llm.get_llm", return_value=SummaryLLM()):
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

    with patch("kesoku.agent.llm.get_llm", return_value=SummaryLLM()):
        reply_all = await bot.manual_compact_session_by_channel("chan_all", "all")
    assert reply_all.startswith("🔄 Compacted 2 turn(s) into ")
    assert "including recent turns" in reply_all
    assert cfg.agent.protect_front_turns == 3
    assert cfg.agent.protect_tail_turns == 6


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


# ---------------------------------------------------------------------------
# K2: Transient Discord transport errors & attachment classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_k2_discord_send_file_retries_503_twice_with_fresh_file(tmp_path: Path) -> None:
    db_path = str(tmp_path / "k2.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(tmp_path / "sessions")),
        discord=DiscordConfig(enabled=True, bot_token="tok", chatbot_id="discord_test"),
    )
    gw = Gateway(context=KesokuContext(config=cfg, db=db))
    sess = await gw.create_session("s_k2", title="K2", chatbot_id="discord_test", channel_id="3001")
    bot = DiscordChatbot(chatbot_id="discord_test", gateway=gw, bot_token="tok", discord_config=cfg.discord)

    img_path = tmp_path / "photo.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nDATA")

    channel = MagicMock()
    send_attempts = 0

    async def flaky_send(*args: Any, **kwargs: Any) -> Any:
        nonlocal send_attempts
        send_attempts += 1
        f_obj = kwargs.get("file")
        assert f_obj is not None
        assert not f_obj.fp.closed
        if send_attempts <= 2:
            raise _make_discord_server_error(503)
        return MagicMock(id=999)

    channel.send = AsyncMock(side_effect=flaky_send)
    bot.bot.get_channel = MagicMock(return_value=channel)

    opened_files: list[discord.File] = []
    real_file_cls = discord.File

    def spy_file(*args: Any, **kwargs: Any) -> discord.File:
        inst = real_file_cls(*args, **kwargs)
        opened_files.append(inst)
        return inst

    msg = Message(
        id="m_k2_ok",
        session_id=sess.id,
        chatbot_id="discord_test",
        channel_id="3001",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content=f"[file: {img_path}]",
    )
    await gw.post(msg)

    with (
        patch("kesoku.gateway.chatbot.discord.adapter.asyncio.sleep", new=AsyncMock()),
        patch("kesoku.gateway.chatbot.discord.adapter.discord.File", side_effect=spy_file),
    ):
        await bot.render_outgoing_message(msg)

    assert send_attempts == 3
    assert len(opened_files) == 3
    assert all(f.fp.closed for f in opened_files)
    raw = await gw.db.get_session_messages(sess.id)
    assert not any(m.metadata.get("is_self_healing_retry") for m in raw)


@pytest.mark.asyncio
async def test_k2_existing_file_503x3_and_timeout_render_ui_without_self_healing(tmp_path: Path) -> None:
    db_path = str(tmp_path / "k2_ui.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(tmp_path / "sessions")),
        discord=DiscordConfig(enabled=True, bot_token="tok", chatbot_id="discord_test"),
    )
    gw = Gateway(context=KesokuContext(config=cfg, db=db))
    sess = await gw.create_session("s_k2_ui", title="K2 UI", chatbot_id="discord_test", channel_id="3002")
    bot = DiscordChatbot(chatbot_id="discord_test", gateway=gw, bot_token="tok", discord_config=cfg.discord)

    img_path = tmp_path / "panel.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nPANEL")

    sent_texts: list[str] = []
    sent_views: list[Any] = []
    channel = MagicMock()

    async def send_503_on_file(*args: Any, **kwargs: Any) -> Any:
        if "file" in kwargs:
            raise _make_discord_server_error(503)
        if "view" in kwargs or "embed" in kwargs:
            sent_views.append(kwargs)
            return MagicMock(id=501)
        if args:
            sent_texts.append(str(args[0]))
        return MagicMock(id=500)

    channel.send = AsyncMock(side_effect=send_503_on_file)
    bot.bot.get_channel = MagicMock(return_value=channel)

    reply_body = (
        f"Scene text\n[file: {img_path}]\n"
        "```ui\ntype: select\nprompt: Pick next action\noptions:\n  - Inspect\n  - Leave\n```"
    )
    msg1 = Message(
        id="m_503x3",
        session_id=sess.id,
        chatbot_id="discord_test",
        channel_id="3002",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content=reply_body,
    )
    await gw.post(msg1)

    with patch("kesoku.gateway.chatbot.discord.adapter.asyncio.sleep", new=AsyncMock()):
        await bot.render_outgoing_message(msg1)

    assert any("⚠️ 附件发送失败（网络错误，已重试）: panel.png" in t for t in sent_texts)
    assert len(sent_views) == 1
    assert not any(m.metadata.get("is_self_healing_retry") for m in await gw.db.get_session_messages(sess.id))

    # Subcase B: DeliveryAbortedError (timeout exhaustion) on an existing file also delivers ⚠️ + renders UI
    dummy_bot = _DummyChatbot("discord_test", gw)
    dummy_bot.send_file_segment = AsyncMock(side_effect=DeliveryAbortedError("Timeout sending file"))  # type: ignore[method-assign]
    msg2 = Message(
        id="m_timeout",
        session_id=sess.id,
        chatbot_id="discord_test",
        channel_id="3002",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content=reply_body,
    )
    await gw.post(msg2)
    await dummy_bot.render_outgoing_message(msg2)

    flat_chunks = [c for grp in dummy_bot.sent_chunks for c in grp]
    assert any("⚠️ 附件发送失败（网络错误，已重试）: panel.png" in c for c in flat_chunks)
    assert len(dummy_bot.sent_questions) == 1
    assert not any(m.metadata.get("is_self_healing_retry") for m in await gw.db.get_session_messages(sess.id))


@pytest.mark.asyncio
async def test_k2_missing_file_self_heals_with_ui_note_and_channel_unavailable_aborts(tmp_path: Path) -> None:
    db_path = str(tmp_path / "k2_abort.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(tmp_path / "sessions")),
        discord=DiscordConfig(enabled=True, bot_token="tok", chatbot_id="discord_test"),
    )
    gw = Gateway(context=KesokuContext(config=cfg, db=db))
    sess = await gw.create_session("s_k2_ab", title="K2 Abort", chatbot_id="discord_test", channel_id="3003")

    # 1. Missing file triggers self-healing containing the ```ui reminder
    dummy_bot = _DummyChatbot("discord_test", gw)
    dummy_bot.send_file_segment = AsyncMock(side_effect=FileNotFoundError("no such file"))  # type: ignore[method-assign]
    msg_missing = Message(
        id="m_missing",
        session_id=sess.id,
        chatbot_id="discord_test",
        channel_id="3003",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="[file: nonexistent.png]\n```ui\ntype: select\nprompt: Q\noptions:\n  - A\n```",
    )
    await gw.post(msg_missing)
    await dummy_bot.render_outgoing_message(msg_missing)
    assert dummy_bot.sent_questions == []
    healing_rows = [
        m for m in await gw.db.get_session_messages(sess.id) if m.metadata.get("is_self_healing_retry")
    ]
    assert len(healing_rows) == 1
    assert "Re-send the full reply including any ```ui block of the failed message." in healing_rows[0].content

    # 2. ChannelUnavailableError on an existing file aborts the whole message (no ⚠️ text, no UI)
    existing = tmp_path / "ok.png"
    existing.write_bytes(b"PNG")
    bot = DiscordChatbot(chatbot_id="discord_test", gateway=gw, bot_token="tok", discord_config=cfg.discord)
    bot.bot.get_channel = MagicMock(return_value=None)
    bot.bot.fetch_channel = AsyncMock(side_effect=_make_discord_not_found(10003, "Unknown Channel"))

    msg_gone = Message(
        id="m_gone",
        session_id=sess.id,
        chatbot_id="discord_test",
        channel_id="3003",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content=f"[file: {existing}]\n```ui\ntype: select\nprompt: Q\noptions:\n  - A\n```",
    )
    await gw.post(msg_gone)
    with pytest.raises(ChannelUnavailableError):
        await bot._get_discord_channel_with_abort(msg_gone)
    await bot.render_outgoing_message(msg_gone)
    # Still only the 1 earlier healing row
    healing_after = [
        m for m in await gw.db.get_session_messages(sess.id) if m.metadata.get("is_self_healing_retry")
    ]
    assert len(healing_after) == 1


@pytest.mark.asyncio
async def test_k2_render_outgoing_message_text_failure_marks_error_and_sends_fallback(tmp_path: Path) -> None:
    db_path = str(tmp_path / "k2_txt.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(tmp_path / "sessions")))
    gw = Gateway(context=KesokuContext(config=cfg, db=db))
    sess = await gw.create_session("s_txt", title="Txt", chatbot_id="mock_bot", channel_id="c_txt")

    bot = _DummyChatbot("mock_bot", gw)
    calls: list[list[str]] = []

    async def fail_first(channel_id: str, chunks: list[str], message: Message) -> None:
        calls.append(list(chunks))
        if len(calls) == 1:
            raise RuntimeError("network reset")

    bot.send_text_chunks = fail_first  # type: ignore[method-assign]
    msg = Message(
        id="m_txt_fail",
        session_id=sess.id,
        chatbot_id="mock_bot",
        channel_id="c_txt",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Long reply text",
        status=MessageStatus.PENDING,
    )
    await gw.post(msg)
    await bot.render_outgoing_message(msg)

    persisted = await gw.db.get_message("m_txt_fail")
    assert persisted is not None
    assert persisted.status == MessageStatus.ERROR
    assert calls == [["Long reply text"], ["⚠️ 回复发送失败，请再说一次"]]


@pytest.mark.asyncio
async def test_k2_deferred_placeholder_and_reply_ref_fallback_retry_transient_errors(tmp_path: Path) -> None:
    db_path = str(tmp_path / "k2_def.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(tmp_path / "sessions")),
        discord=DiscordConfig(enabled=True, bot_token="tok", chatbot_id="discord_test"),
    )
    gw = Gateway(context=KesokuContext(config=cfg, db=db))
    sess = await gw.create_session("s_def", title="Def", chatbot_id="discord_test", channel_id="3009")
    bot = DiscordChatbot(chatbot_id="discord_test", gateway=gw, bot_token="tok", discord_config=cfg.discord)
    bot.bot.is_ready = MagicMock(return_value=True)

    channel = MagicMock()
    placeholder_calls = 0

    async def flaky_placeholder(*args: Any, **kwargs: Any) -> Any:
        nonlocal placeholder_calls
        placeholder_calls += 1
        if placeholder_calls == 1:
            raise aiohttp.ClientConnectionError("reset")
        return MagicMock(id=888, channel=MagicMock(id=3009))

    channel.send = AsyncMock(side_effect=flaky_placeholder)
    bot.bot.get_channel = MagicMock(return_value=channel)

    msg = Message(
        id="m_def",
        session_id=sess.id,
        chatbot_id="discord_test",
        channel_id="3009",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="hi",
    )
    with patch("kesoku.gateway.chatbot.discord.adapter.asyncio.sleep", new=AsyncMock()):
        handle = await bot.send_deferred_placeholder("3009", {"type": "file", "path": "a.png"}, msg, {})
    assert handle is not None
    assert placeholder_calls == 2

    # deliver_deferred_attachment: permanent edit error (400) -> reply-reference fallback retries 429 once then succeeds
    img = tmp_path / "done.png"
    img.write_bytes(b"PNG")
    handle.edit = AsyncMock(side_effect=_make_discord_http_exc(400, text="Bad Request"))
    ref_calls = 0

    async def flaky_ref(*args: Any, **kwargs: Any) -> Any:
        nonlocal ref_calls
        ref_calls += 1
        if ref_calls == 1:
            raise _make_discord_http_exc(429, text="Rate limited")
        return MagicMock(id=889)

    channel.send = AsyncMock(side_effect=flaky_ref)
    with patch("kesoku.gateway.chatbot.discord.adapter.asyncio.sleep", new=AsyncMock()):
        await bot.deliver_deferred_attachment("3009", str(img), msg, handle, "file")
    assert ref_calls == 2


# ---------------------------------------------------------------------------
# K3: Undelivered assistant replies marked in build_history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_k3_build_history_prefixes_errored_assistant_rows_on_copy(tmp_path: Path) -> None:
    db_path = str(tmp_path / "k3.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path))
    gw = Gateway(context=KesokuContext(config=cfg, db=db))
    await gw.create_session("s_k3", title="K3")

    m_err = Message(
        id="a_err",
        session_id="s_k3",
        chatbot_id="discord",
        channel_id="c1",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Invisible reply",
        status=MessageStatus.ERROR,
        timestamp=10.0,
    )
    m_ok = Message(
        id="a_ok",
        session_id="s_k3",
        chatbot_id="discord",
        channel_id="c1",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="Visible reply",
        status=MessageStatus.DELIVERED,
        timestamp=20.0,
    )
    await gw.post(m_err)
    await gw.post(m_ok)

    hist = await build_history(gw, "s_k3", heal_orphans=False)
    by_id = {m.id: m for m in hist}
    assert by_id["a_err"].content == "[未送达：以下这条回复因发送失败，用户没有看到]\nInvisible reply"
    assert by_id["a_ok"].content == "Visible reply"

    # DB row itself must remain unmutated
    persisted = await gw.db.get_message("a_err")
    assert persisted is not None
    assert persisted.content == "Invisible reply"


# ---------------------------------------------------------------------------
# K4: Per-chatbot default_role & TOML multi-bot channel overrides
# ---------------------------------------------------------------------------


def test_k4_toml_parse_two_discord_bots_with_default_role_and_five_channel_blocks() -> None:
    raw_toml = """
[[chatbots.discord]]
enabled = true
chatbot_id = "discord_primary"
bot_token = "tok_1"

[[chatbots.discord.channels]]
channels = ["claude"]
llm = "claude"

[[chatbots.discord.channels]]
channels = ["opus"]
llm = "claude"
model_name = "claude-opus-4-6"

[[chatbots.discord.channels]]
channels = ["argon"]
llm = "gemini"
thinking_level = "high"

[[chatbots.discord.channels]]
channels = ["flash", "general"]
llm = "gemini"
thinking_level = "medium"

[[chatbots.discord.channels]]
channels = ["lite", "quick"]
llm = "gemini"
thinking_level = "minimal"

[[chatbots.discord]]
enabled = true
chatbot_id = "discord_persona"
bot_token = "tok_2"
default_role = "mentor"

[[chatbots.discord.channels]]
channels = ["claude"]
llm = "claude"

[[chatbots.discord.channels]]
channels = ["opus"]
llm = "claude"
model_name = "claude-opus-4-6"

[[chatbots.discord.channels]]
channels = ["argon"]
llm = "gemini"
thinking_level = "high"

[[chatbots.discord.channels]]
channels = ["flash", "general"]
llm = "gemini"
thinking_level = "medium"

[[chatbots.discord.channels]]
channels = ["lite", "quick"]
llm = "gemini"
thinking_level = "minimal"
"""
    cfg = KesokuConfig.model_validate(tomllib.loads(raw_toml))
    assert cfg.get_chatbot_default_role("discord_primary") == "default"
    assert cfg.get_chatbot_default_role("discord_persona") == "mentor"
    assert cfg.get_chatbot_default_role("unknown_bot") == "default"

    bot2 = cfg.get_discord_config("discord_persona")
    assert bot2 is not None
    assert bot2.default_role == "mentor"
    assert len(bot2.channels) == 5
    assert [c.channels for c in bot2.channels] == [
        ["claude"],
        ["opus"],
        ["argon"],
        ["flash", "general"],
        ["lite", "quick"],
    ]

    # Also verify WeChat & GoogleChat default_role resolution
    cfg.chatbots.wechat.append(WechatConfig(chatbot_id="wc_persona", default_role="tutor"))
    cfg.chatbots.google_chat.append(GoogleChatConfig(chatbot_id="gc_persona", default_role="coach"))
    assert cfg.get_chatbot_default_role("wc_persona") == "tutor"
    assert cfg.get_chatbot_default_role("gc_persona") == "coach"


@pytest.mark.asyncio
async def test_k4_role_inheritance_new_thread_clear_explicit_override_and_resolvers(tmp_path: Path) -> None:
    db_path = str(tmp_path / "k4.db")
    roles_dir = tmp_path / "roles"
    (roles_dir / "mentor").mkdir(parents=True)
    (roles_dir / "mentor" / "intro.md").write_text("You are Mentor Persona.", encoding="utf-8")
    (roles_dir / "default").mkdir(parents=True)
    (roles_dir / "default" / "intro.md").write_text("You are Default Persona.", encoding="utf-8")

    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(
            db_path=db_path,
            roles_dir=str(roles_dir),
            sessions_dir=str(tmp_path / "sessions"),
        ),
        agent_working_dir=str(tmp_path),
    )
    cfg.chatbots.discord.append(
        DiscordConfig(enabled=True, bot_token="tok", chatbot_id="discord_persona", default_role="mentor")
    )
    cfg.chatbots.discord.append(
        DiscordConfig(enabled=True, bot_token="tok2", chatbot_id="discord_plain")
    )
    ctx = KesokuContext(config=cfg, db=db)
    gw = Gateway(context=ctx)

    # 1. New Discord thread (no DB binding) on chatbot with default_role="mentor" -> session role "mentor"
    sess_thread = await gw.create_session(
        None,
        title="New Thread",
        chatbot_id="discord_persona",
        channel_id="thread_9001",
    )
    assert sess_thread.role_name == "mentor"
    persisted_thread = await gw.db.get_session(sess_thread.id)
    assert persisted_thread is not None and persisted_thread.role_name == "mentor"

    # 2. /clear on that channel re-creates session with "mentor"
    bot = _DummyChatbot("discord_persona", gw)
    await bot.clear_session_by_channel("thread_9001")
    cleared_sess = await gw.db.get_session_by_channel("discord_persona", "thread_9001")
    assert cleared_sess is not None
    assert cleared_sess.id != sess_thread.id
    assert cleared_sess.role_name == "mentor"

    # 3. Resolvers across TurnExecutor, prompt.py, file.py, and memory.py all resolve "mentor"
    # Even if a legacy session row had role_name="default", channel inheritance resolves "mentor"
    legacy_sess = Session(id="s_legacy", title="Legacy", role_name="default")
    await gw.db.create_session(legacy_sess)
    await gw.db.set_active_session_for_channel("discord_persona", "chan_legacy", legacy_sess.id)

    u_msg = Message(
        id="u_leg",
        session_id="s_legacy",
        chatbot_id="discord_persona",
        channel_id="chan_legacy",
        sender="u",
        role=MessageRole.USER,
        content="hi",
    )
    await gw.post(u_msg)

    tool_ctx = ToolContext(gateway=gw, session_id="s_legacy", original_msg_id="u_leg")
    assert _resolve_role_sync(tool_ctx) == "mentor"
    assert await _resolve_role(None, tool_ctx) == "mentor"

    tr = MagicMock()
    tr.tool_registry.get_tools_list.return_value = []
    tr.tool_context = tool_ctx
    executor = TurnExecutor("s_legacy", gw, tr, MagicMock(spec=TurnLogger), context=ctx)
    assert await executor._resolve_active_role(u_msg) == "mentor"

    with patch("kesoku.agent.prompt.get_config", return_value=cfg):
        sys_prompt = build_sys_prompt(session=Session(id="s_legacy", title="Legacy", role_name="default"))
    assert "You are Mentor Persona." in sys_prompt

    # 4. /role custom_scholar wins over default_role
    await gw.db.set_channel_role("discord_persona", "thread_9001", "custom_scholar")
    assert await gw.db.get_channel_role_with_inheritance("discord_persona", "thread_9001") == "custom_scholar"

    # 5. Chatbot without default_role falls back to "default"
    sess_plain = await gw.create_session(
        None,
        title="Plain Thread",
        chatbot_id="discord_plain",
        channel_id="thread_plain",
    )
    assert sess_plain.role_name == "default"


# ---------------------------------------------------------------------------
# K5: Preempted turns synthesize cancelled TOOL_RESULT rows; neutral orphans
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_k5_preemption_posts_cancelled_tool_results_and_zero_orphans(tmp_path: Path) -> None:
    from kesoku.agent.llm import ToolCallRequest

    db_path = str(tmp_path / "k5.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path))
    ctx = KesokuContext(config=cfg, db=db)
    gw = Gateway(context=ctx)
    await gw.create_session("s_k5", title="K5")

    u_msg = Message(
        id="u_k5",
        session_id="s_k5",
        chatbot_id="discord",
        channel_id="c1",
        sender="u",
        role=MessageRole.USER,
        content="run tools",
        status=MessageStatus.PROCESSING,
    )
    await gw.post(u_msg)

    tool_runner = MagicMock()
    tool_runner.tool_context = ToolContext(gateway=gw, session_id="s_k5")

    async def never_runs(*args: Any, **kwargs: Any) -> Message:
        raise AssertionError("execute_tool must not run when preempted")

    tool_runner.execute_tool = never_runs
    executor = TurnExecutor("s_k5", gw, tool_runner, MagicMock(spec=TurnLogger), context=ctx)

    worker = MagicMock()
    worker.queue_empty.return_value = False  # Preempted!

    res = LLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(name="run_shell_command", arguments={"command": "echo 1"}, tool_call_id="tc_1"),
            ToolCallRequest(name="read_file", arguments={"file_name": "a.txt"}, tool_call_id="tc_2"),
        ],
    )
    should_continue = await executor._execute_tool_calls(res, u_msg, worker, str(tmp_path))
    assert should_continue is False

    # Zero orphans in DB!
    orphans = await gw.db.get_orphaned_tool_calls("s_k5")
    assert orphans == []

    msgs = await gw.db.get_session_messages("s_k5")
    results = [m for m in msgs if m.type == MessageType.TOOL_RESULT]
    assert len(results) == 2
    assert {r.sender for r in results} == {"run_shell_command", "read_file"}
    for r in results:
        assert r.metadata.get("tool_error") == "cancelled"
        assert "was cancelled before it ran because the user sent a new message." in r.content


# ---------------------------------------------------------------------------
# K6: Expired Discord interaction (10062) still records choice & advances turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_k6_expired_interaction_10062_persists_edits_and_posts_choice() -> None:
    client = object()
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    class MemDB:
        async def save_ui_prompt(self, sid: str, mid: str, spec: dict[str, Any]) -> None:
            rows[(sid, mid)] = dict(spec)

        async def get_ui_prompt(self, sid: str, mid: str) -> dict[str, Any] | None:
            r = rows.get((sid, mid))
            return dict(r) if r else None

    mem_db = MemDB()
    spec = pui.build_question_spec("Where next?", ["Option Alpha", "Option Beta"], multi=False, max_choices=1)
    await mem_db.save_ui_prompt("s_ui", "m_ui", spec)

    chatbot = MagicMock()
    chatbot.chatbot_id = "discord_test"
    chatbot.gateway.db = mem_db
    chatbot.gateway.post = AsyncMock()
    chatbot._typing_tasks = {"chan_ui": MagicMock()}
    pui.register_chatbot(client, chatbot)  # type: ignore[arg-type]

    interaction = MagicMock()
    interaction.client = client
    interaction.response.defer = AsyncMock(side_effect=_make_discord_not_found(10062))
    interaction.followup.send = AsyncMock()
    interaction.edit_original_response = AsyncMock()
    interaction.channel_id = "chan_ui"
    interaction.user.id = 99
    interaction.user.display_name = "Player"
    sent_msg = MagicMock(id=321)
    sent_msg.created_at.timestamp.return_value = 500.0
    interaction.channel.send = AsyncMock(return_value=sent_msg)
    interaction.message.edit = AsyncMock()

    await pui.handle_question_answer(interaction, "s_ui", "m_ui", [1])

    saved = await mem_db.get_ui_prompt("s_ui", "m_ui")
    assert saved is not None and saved.get("answered") == "Option Beta"
    interaction.edit_original_response.assert_not_awaited()
    interaction.message.edit.assert_awaited_once()
    chatbot.gateway.post.assert_awaited_once()
    posted_msg = chatbot.gateway.post.call_args[0][0]
    assert posted_msg.content == "Option Beta"
    assert posted_msg.role == MessageRole.USER

    # Re-clicking the same option when expired (10062) notifies via channel.send instead of followup.send
    interaction.channel.send.reset_mock()
    await pui.handle_question_answer(interaction, "s_ui", "m_ui", [1])
    interaction.followup.send.assert_not_awaited()
    interaction.channel.send.assert_awaited_once()
    assert "Already answered: Option Beta" in interaction.channel.send.call_args[0][0]

    # Non-10062 NotFound (e.g. 10008 Unknown Message) does NOT post a choice
    await mem_db.save_ui_prompt("s_ui2", "m_ui2", pui.build_question_spec("Q2", ["A", "B"], False, 1))
    chatbot.gateway.post.reset_mock()
    interaction2 = MagicMock()
    interaction2.client = client
    interaction2.response.defer = AsyncMock(side_effect=_make_discord_not_found(10008, "Unknown Message"))
    await pui.handle_question_answer(interaction2, "s_ui2", "m_ui2", [0])
    chatbot.gateway.post.assert_not_awaited()


# ---------------------------------------------------------------------------
# K7: Deferred media fuzzy match [C-7] & clean staging file listing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# K8: Foreground shell log cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_k8_foreground_shell_unlinks_background_job_logs_and_keeps_bg(tmp_path: Path) -> None:
    db_path = str(tmp_path / "k8.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    sessions_dir = tmp_path / "sessions"
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(sessions_dir)))
    cfg.shell.mode = "blocklist"
    cfg.shell.background_threshold_seconds = 5.0
    gw = Gateway(context=KesokuContext(config=cfg, db=db))
    sess = await gw.create_session("s_k8", title="K8")

    from kesoku.utils.path import PathResolver

    tctx = ToolContext(
        gateway=gw,
        session_id=sess.id,
        session_workspace=sess.workspace_name,
        active_jobs=ActiveJobsRegistry(),
        kesoku_context=gw.context,
    )

    with (
        patch("kesoku.agent.tools.shell.get_config", return_value=cfg),
        patch("kesoku.utils.path.get_config", return_value=cfg),
    ):
        staging_dir = Path(PathResolver.get_session_staging_dir(sess.workspace_name))
        out = await run_shell_command("echo hello_k8", context=tctx)
        assert "hello_k8" in out
        assert [f for f in os.listdir(staging_dir) if f.startswith("background_job_")] == []

        # Truncated foreground output keeps cmd_output_*.txt while still unlinking background_job_*
        long_out = await run_shell_command(
            "python3 -c \"print('X' * 2500)\"", max_output_chars=1200, context=tctx
        )
        assert "cmd_output_" in long_out
        assert [f for f in os.listdir(staging_dir) if f.startswith("background_job_")] == []
        assert len([f for f in os.listdir(staging_dir) if f.startswith("cmd_output_")]) == 1

        # Backgrounded command retains its background_job_*.stdout/.stderr while running
        cfg.shell.background_threshold_seconds = 0.2
        bg_out = await run_shell_command("sleep 2", context=tctx)
        assert "transitioned to background execution" in bg_out
        bg_files = [f for f in os.listdir(staging_dir) if f.startswith("background_job_")]
        assert len(bg_files) == 2
        await tctx.active_jobs.stop_all_for_session(sess.id)
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# K9: Vertex / Gemini request timeout (config + HttpOptions + non-stream + stream)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# K10: Inbound attachment inline-visibility hint & analyze_media docstring
# ---------------------------------------------------------------------------


def test_k10_format_inbound_content_and_analyze_media_docstring(tmp_path: Path) -> None:
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=str(tmp_path / "k10.db")))
    DatabaseManager(cfg.workspace.db_path).init_tables()
    gw = Gateway(context=KesokuContext(config=cfg))
    bot = _DummyChatbot("discord_test", gw)

    dto = InboundMessageDTO(
        sender_id="u1",
        channel_id="c1",
        text="Check these files",
        attachments=[
            InboundMessageAttachment(path="/tmp/a.jpg", mime_type="image/jpeg", filename="a.jpg"),
            InboundMessageAttachment(path="/tmp/b.ogg", mime_type="audio/ogg", filename="b.ogg"),
            InboundMessageAttachment(path="/tmp/c.pdf", mime_type="application/pdf", filename="c.pdf"),
            InboundMessageAttachment(path="/tmp/d.txt", mime_type="text/plain", filename="d.txt"),
        ],
    )
    formatted = bot._format_inbound_content(dto)
    lines = formatted.splitlines()
    img_line = next(line for line in lines if "a.jpg" in line)
    aud_line = next(line for line in lines if "b.ogg" in line)
    pdf_line = next(line for line in lines if "c.pdf" in line)
    txt_line = next(line for line in lines if "d.txt" in line)

    expected_clause = (
        "— image/audio attachments are already visible to you in this message; do not call analyze_media on them"
    )
    assert expected_clause in img_line
    assert expected_clause in aud_line
    assert expected_clause not in pdf_line
    assert expected_clause not in txt_line

    assert analyze_media.__doc__ is not None
    assert (
        "Use only for files that are not part of the current user message and that you did not just generate yourself."
        in analyze_media.__doc__
    )

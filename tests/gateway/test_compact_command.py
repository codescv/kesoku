"""Tests for the `/compact [instructions]` command plumbing (P1-4): routing, focus directive, busy refusal."""

from typing import Any

import pytest

from kesoku.agent.compressor import HistoryCompressor
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message
from kesoku.gateway.chatbot.base import Chatbot
from kesoku.gateway.gateway import Gateway


class _Bot(Chatbot):
    async def handle_message(self, message: Message) -> None:
        return None


async def _setup(tmp_path: Any, n_turns: int = 6, status: str = MessageStatus.PROCESSED) -> tuple[Gateway, _Bot]:
    db_path = str(tmp_path / "compact.db")
    DatabaseManager(db_path).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, sessions_dir=str(tmp_path / "sessions")))
    gw = Gateway(context=KesokuContext(config=cfg))
    await gw.create_session("sess_c", title="c", chatbot_id="cli", channel_id="ch")
    for i in range(n_turns):
        await gw.post(Message(session_id="sess_c", chatbot_id="cli", channel_id="ch", sender="u",
                              role=MessageRole.USER, type=MessageType.TEXT, content=f"question {i}",
                              status=status if i == n_turns - 1 else MessageStatus.PROCESSED, timestamp=100.0 + i * 10))
        await gw.post(Message(session_id="sess_c", chatbot_id="cli", channel_id="ch", sender="Kesoku",
                              role=MessageRole.ASSISTANT, type=MessageType.TEXT, content=f"answer {i}",
                              status=MessageStatus.DELIVERED, timestamp=101.0 + i * 10))
    return gw, _Bot("cli", gw)


async def _run(bot: _Bot, text: str) -> str:
    replies: list[str] = []

    async def reply(content: str, *args: Any, **kwargs: Any) -> None:
        replies.append(content)

    await bot.execute_command_from_text(text, reply_func=reply, channel_id="ch")
    return "\n".join(replies)


@pytest.mark.asyncio
async def test_compact_passes_focus_instructions(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    gw, bot = await _setup(tmp_path)
    seen: dict[str, Any] = {}

    async def fake_force(self: HistoryCompressor, session_id: str, history: list, llm: Any, config: Any, *,
                         directive: str | None = None, staging_dir: str | None = None) -> int:
        seen.update(session_id=session_id, directive=directive, staging_dir=staging_dir)
        for msg in history[2:6]:  # pretend turns 2-3 were compacted
            msg.summary_node_id = "n1"
        return 1

    monkeypatch.setattr(HistoryCompressor, "force_compact_session", fake_force, raising=False)
    monkeypatch.setattr("kesoku.agent.llm.get_llm", lambda *a, **k: object())
    reply = await _run(bot, "/compact  keep only Subsystem Y   signatures ")
    assert seen["session_id"] == "sess_c"
    assert seen["directive"] == "keep only Subsystem Y   signatures"
    assert seen["staging_dir"] and seen["staging_dir"].endswith("sess_c")
    assert reply.startswith("🔄 Compacted 2 turn(s) into 1 summary node(s).")
    assert 'Focus: "keep only Subsystem Y   signatures"' in reply


@pytest.mark.asyncio
async def test_bare_compact_has_no_directive_and_reports_nothing_to_do(tmp_path: Any,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    gw, bot = await _setup(tmp_path)
    seen: dict[str, Any] = {}

    async def fake_force(self: HistoryCompressor, session_id: str, history: list, llm: Any, config: Any, *,
                         directive: str | None = None, staging_dir: str | None = None) -> int:
        seen["directive"] = directive
        return 0

    monkeypatch.setattr(HistoryCompressor, "force_compact_session", fake_force, raising=False)
    monkeypatch.setattr("kesoku.agent.llm.get_llm", lambda *a, **k: object())
    reply = await _run(bot, "/compact")
    assert seen["directive"] is None
    assert reply.startswith("ℹ️ Nothing to compact")
    assert "protected head/tail" in reply


@pytest.mark.asyncio
async def test_compact_refuses_while_turn_in_progress(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    gw, bot = await _setup(tmp_path, status=MessageStatus.PROCESSING)
    called = False

    async def fake_force(*args: Any, **kwargs: Any) -> int:
        nonlocal called
        called = True
        return 1

    monkeypatch.setattr(HistoryCompressor, "force_compact_session", fake_force, raising=False)
    reply = await _run(bot, "/compact keep Y")
    assert reply == "⏳ A turn is in progress; run /compact after it finishes."
    assert not called


@pytest.mark.asyncio
async def test_compact_refuses_while_worker_busy(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    gw, bot = await _setup(tmp_path)

    class _Worker:
        _processing_turn = True

    class _Agent:
        workers = {"sess_c": _Worker()}

    gw.agent = _Agent()
    reply = await _run(bot, "/compact")
    assert reply.startswith("⏳ A turn is in progress")


@pytest.mark.asyncio
async def test_compact_requires_channel(tmp_path: Any) -> None:
    gw, bot = await _setup(tmp_path)
    replies: list[str] = []

    async def reply(content: str, *args: Any, **kwargs: Any) -> None:
        replies.append(content)

    await bot.execute_command_from_text("/compact x", reply_func=reply, channel_id=None)
    assert replies == ["⚠️ Channel ID is required for this command."]

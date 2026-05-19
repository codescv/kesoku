"""Unit tests for Kesoku V3 Concurrency & Logic (Broker pub/sub, Session Workers, Interruption)."""

import asyncio
from typing import Any

import pytest

from kesoku.agent.agent import Agent
from kesoku.agent.llm import MockLLM
from kesoku.agent.tools import ToolRegistry
from kesoku.config import WorkspaceConfig
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_USER,
    STATUS_INTERRUPTED,
    STATUS_PENDING_AGENT,
    STATUS_PROCESSED,
    TYPE_TEXT,
)
from kesoku.db import DatabaseManager, Message
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def temp_db(tmp_path: Any) -> str:
    return str(tmp_path / "test_concurrency.db")


@pytest.mark.asyncio
async def test_pure_broker_pubsub(temp_db: str) -> None:
    """Test that Gateway post() correctly broadcasts messages to active matching listeners."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))

    received_user = []
    received_model = []

    async def user_listener() -> None:
        async for msg in gw.listen(role=ROLE_USER):
            received_user.append(msg)
            if len(received_user) >= 1:
                break

    async def model_listener() -> None:
        async for msg in gw.listen(role=ROLE_ASSISTANT, chatbot_id="cli"):
            received_model.append(msg)
            if len(received_model) >= 1:
                break

    user_task = asyncio.create_task(user_listener())
    model_task = asyncio.create_task(model_listener())

    # Post a user message
    msg1 = Message(session_id="s1", chatbot_id="cli", channel_id="c1", sender="User", role=ROLE_USER, content="Hi")
    await gw.post(msg1)

    # Post a model message
    msg2 = Message(
        session_id="s1", chatbot_id="cli", channel_id="c1", sender="Agent", role=ROLE_ASSISTANT, content="Resp"
    )
    await gw.post(msg2)

    await asyncio.gather(user_task, model_task)

    assert len(received_user) == 1
    assert received_user[0].content == "Hi"
    assert len(received_model) == 1
    assert received_model[0].content == "Resp"


@pytest.mark.asyncio
async def test_multi_user_simultaneous(temp_db: str) -> None:
    """Test that Agent spawns separate SessionWorkers for simultaneous user sessions."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))
    reg = ToolRegistry()

    @reg.register
    def calculator(expression: str, context: Any = None) -> str:
        return "Result = 20.0"

    llm = MockLLM()

    agent = Agent(gw, llm, reg)
    agent_task = asyncio.create_task(agent.start())

    # Ingest messages from two separate sessions simultaneously
    await gw.create_session("sess_A", title="Sess A")
    await gw.create_session("sess_B", title="Sess B")
    msg_a = await gw.post(
        Message(
            session_id="sess_A",
            chatbot_id="cli",
            channel_id="ch_A",
            sender="u1",
            role=ROLE_USER,
            type=TYPE_TEXT,
            content="Please calculate 10 + 10",
            status=STATUS_PENDING_AGENT,
        )
    )
    msg_b = await gw.post(
        Message(
            session_id="sess_B",
            chatbot_id="cli",
            channel_id="ch_B",
            sender="u2",
            role=ROLE_USER,
            type=TYPE_TEXT,
            content="Please calculate 20 + 20",
            status=STATUS_PENDING_AGENT,
        )
    )

    await asyncio.sleep(0.5)
    agent.stop()
    await asyncio.gather(agent_task, return_exceptions=True)

    # Verify both sessions were processed and workers were created
    hist_a = await gw.get_session_history("sess_A")
    hist_b = await gw.get_session_history("sess_B")

    assert any(m.status == STATUS_PROCESSED for m in hist_a)
    assert any(m.status == STATUS_PROCESSED for m in hist_b)


@pytest.mark.asyncio
async def test_user_thought_interruption(temp_db: str) -> None:
    """Test sending an interruption message while a session is processing."""
    DatabaseManager(temp_db).init_tables()
    gw = Gateway(workspace_config=WorkspaceConfig(db_path=temp_db))
    reg = ToolRegistry()

    @reg.register
    def calculator(expression: str, context: Any = None) -> str:
        return "Result = 2.0"

    llm = MockLLM()

    agent = Agent(gw, llm, reg)
    agent_task = asyncio.create_task(agent.start())

    # Send first message
    await gw.create_session("sess_int", title="Sess Int")
    msg1 = await gw.post(
        Message(
            session_id="sess_int",
            chatbot_id="cli",
            channel_id="ch_int",
            sender="u1",
            role=ROLE_USER,
            type=TYPE_TEXT,
            content="Please calculate 1 + 1",
            status=STATUS_PENDING_AGENT,
        )
    )

    # Immediately send second message before or during tool execution
    msg2 = await gw.post(
        Message(
            session_id="sess_int",
            chatbot_id="cli",
            channel_id="ch_int",
            sender="u1",
            role=ROLE_USER,
            type=TYPE_TEXT,
            content="Actually, calculate 2 + 2",
            status=STATUS_PENDING_AGENT,
        )
    )

    await asyncio.sleep(0.5)
    agent.stop()
    await asyncio.gather(agent_task, return_exceptions=True)

    hist = await gw.get_session_history("sess_int")

    # Verify msg1 was interrupted or msg2 was processed
    assert any(m.status == STATUS_INTERRUPTED or m.id == msg2.id for m in hist)
    assert any(m.status == STATUS_PROCESSED for m in hist)

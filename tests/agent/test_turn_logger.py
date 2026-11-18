"""Unit tests for TurnLogger class."""

import asyncio
import os
import tempfile
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from kesoku.agent.agent import SessionWorker
from kesoku.agent.llm import (
    BaseLLM,
    GeminiLLM,
    LLMResponse,
    ToolCallRequest,
)
from kesoku.agent.turn_executor import (
    TurnExecutor,
    discard_session_cache_state,
)
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import (
    GeminiConfig,
    KesokuConfig,
    WorkspaceConfig,
)
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import AsyncDatabaseManager, Message
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def temp_staging_dir() -> str:
    """Create a temporary staging directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_get_next_llm_turn_idx_empty(temp_staging_dir: str) -> None:
    """Verify that turn index starts at 1 when directory is empty."""
    logger = TurnLogger("sess_1", temp_staging_dir)
    assert logger._get_next_llm_turn_idx() == 1


def test_get_next_llm_turn_idx_existing(temp_staging_dir: str) -> None:
    """Verify that turn index detects the highest existing index and returns max + 1."""
    logger = TurnLogger("sess_1", temp_staging_dir)

    # Create dummy log files
    open(os.path.join(temp_staging_dir, "llm-turn-1.log.yaml"), "w").close()
    open(os.path.join(temp_staging_dir, "llm-turn-5.log.yaml"), "w").close()
    open(os.path.join(temp_staging_dir, "llm-turn-2.log.yaml"), "w").close()
    open(os.path.join(temp_staging_dir, "other-file.txt"), "w").close()

    assert logger._get_next_llm_turn_idx() == 6


def test_log_llm_turn_serialization(temp_staging_dir: str) -> None:
    """Verify that log_llm_turn serializes metadata, history, tools, and responses correctly."""
    logger = TurnLogger("sess_1", temp_staging_dir)

    history = [
        Message(
            session_id="sess_1",
            chatbot_id="cli",
            channel_id="ch1",
            sender="u1",
            role="user",
            content="Compute 5 + 5",
        )
    ]

    def dummy_calculator(x: int, y: int) -> int:
        """Dummy tool docstring."""
        return x + y

    tools: list[Any] = [dummy_calculator]

    response = LLMResponse(
        content="The result is 10",
        thought="Calculated the sum",
        tool_calls=[
            ToolCallRequest(
                name="dummy_calculator",
                arguments={"x": 5, "y": 5},
                thought_signature="sum_sig",
                tool_call_id="tc_1",
            )
        ],
        prompt_tokens=10,
        candidates_tokens=5,
        total_tokens=15,
    )

    logger.log_llm_turn(
        llm_provider="TestLLM",
        history=history,
        tools=tools,
        response=response,
    )

    # Verify file creation
    log_path = os.path.join(temp_staging_dir, ".kesoku_logs", "llm-turn-1.log.yaml")
    assert os.path.exists(log_path)

    # Verify file content
    with open(log_path, encoding="utf-8") as f:
        log_data = yaml.safe_load(f)

    # Assert metadata
    assert log_data["metadata"]["session_id"] == "sess_1"
    assert log_data["metadata"]["turn_index"] == 1
    assert log_data["metadata"]["llm_provider"] == "TestLLM"
    assert "timestamp_iso" in log_data["metadata"]

    # Assert history
    assert len(log_data["history"]) == 1
    assert log_data["history"][0]["role"] == "user"
    assert log_data["history"][0]["content"] == "Compute 5 + 5"

    # Assert tools
    assert len(log_data["tools"]) == 1
    assert log_data["tools"][0]["name"] == "dummy_calculator"
    assert log_data["tools"][0]["description"] == "Dummy tool docstring."
    assert log_data["tools"][0]["parameters"] == {"x": "<class 'int'>", "y": "<class 'int'>"}

    # Assert response
    assert log_data["response"]["content"] == "The result is 10"
    assert log_data["response"]["thought"] == "Calculated the sum"
    assert len(log_data["response"]["tool_calls"]) == 1
    assert log_data["response"]["tool_calls"][0]["name"] == "dummy_calculator"
    assert log_data["response"]["tool_calls"][0]["arguments"] == {"x": 5, "y": 5}
    assert log_data["response"]["tool_calls"][0]["tool_call_id"] == "tc_1"
    assert log_data["response"]["metrics"] == {
        "prompt_tokens": 10,
        "candidates_tokens": 5,
        "cached_tokens": None,
        "total_tokens": 15,
    }


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


# origin: tests/test_ttft_async_tts_v015.py
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
    # K1/K3 structural telemetry fields are present on every round (test doubles report no part data).
    for r in (r1, r2):
        assert r["thought_tokens"] is None
        assert r["parts"] is None
        assert r["reply_marker"] == "off"
        assert r["pre_reply_chars"] == 0
    discard_session_cache_state("s_tel")

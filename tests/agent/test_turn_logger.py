"""Unit tests for TurnLogger class."""

import os
import tempfile
from typing import Any

import pytest
import yaml

from kesoku.agent.llm import LLMResponse, ToolCallRequest
from kesoku.agent.turn_logger import TurnLogger
from kesoku.db import Message


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
    log_path = os.path.join(temp_staging_dir, "llm-turn-1.log.yaml")
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
        "total_tokens": 15,
    }

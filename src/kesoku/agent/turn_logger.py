"""Handles logging raw LLM turn inputs, outputs, and metrics to session staging directory."""

import datetime
import inspect
import os
import re
import time
from collections.abc import Callable
from typing import Any

import yaml

from kesoku.agent.llm import LLMResponse
from kesoku.db import Message
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class TurnLogger:
    """Handles logging raw LLM turn inputs, outputs, and metrics to session staging directory."""

    def __init__(self, session_id: str, session_staging_dir: str) -> None:
        """Initialize TurnLogger.

        Args:
            session_id: Unique conversational session identifier.
            session_staging_dir: Target directory path for staging logs.
        """
        self.session_id = session_id
        self.session_staging_dir = session_staging_dir

    def _get_next_llm_turn_idx(self) -> int:
        """Determine the next turn index by scanning the staging directory for existing log.yaml files.

        Returns:
            The next turn index starting from 1.
        """
        if not os.path.exists(self.session_staging_dir):
            return 1
        max_idx = 0
        try:
            # Scan directory and find the maximum existing turn index
            for filename in os.listdir(self.session_staging_dir):
                match = re.match(r"llm-turn-(\d+)\.log\.yaml", filename)
                if match:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
        except Exception as e:
            logger.warning(f"Error scanning staging directory for turn logs: {e}")
        return max_idx + 1

    def log_llm_turn(
        self,
        llm_provider: str,
        history: list[Message],
        tools: list[Callable[..., Any]],
        response: LLMResponse,
    ) -> None:
        """Log the raw LLM turn inputs and outputs to a YAML file in the session staging directory.

        Args:
            llm_provider: Name of the LLM provider.
            history: Conversational history sent to the LLM.
            tools: List of tools passed to the LLM.
            response: LLMResponse object returned by the LLM.
        """
        os.makedirs(self.session_staging_dir, exist_ok=True)
        idx = self._get_next_llm_turn_idx()
        log_filename = f"llm-turn-{idx}.log.yaml"
        log_filepath = os.path.join(self.session_staging_dir, log_filename)

        # Format history messages into dict representation
        formatted_history = []
        for msg in history:
            formatted_history.append({
                "id": msg.id,
                "role": msg.role,
                "sender": msg.sender,
                "type": msg.type,
                "content": msg.content,
                "metadata": msg.metadata,
                "timestamp": msg.timestamp,
                "status": msg.status,
                "parent_id": msg.parent_id,
            })

        # Format tools with signature details
        formatted_tools = []
        for func in tools:
            doc = inspect.getdoc(func) or ""
            description = doc.split("\n\n")[0] if doc else ""
            sig = inspect.signature(func)
            parameters = {
                p_name: str(p.annotation)
                for p_name, p in sig.parameters.items()
                if p_name != "context"
            }
            formatted_tools.append({
                "name": func.__name__,
                "description": description,
                "parameters": parameters,
            })

        # Format response tool calls
        formatted_tool_calls = []
        for tc in response.tool_calls:
            formatted_tool_calls.append({
                "name": tc.name,
                "arguments": tc.arguments,
                "thought_signature": tc.thought_signature,
                "tool_call_id": tc.tool_call_id,
            })

        now = datetime.datetime.fromtimestamp(time.time(), datetime.UTC)

        log_data = {
            "metadata": {
                "timestamp": time.time(),
                "timestamp_iso": now.isoformat(),
                "session_id": self.session_id,
                "turn_index": idx,
                "llm_provider": llm_provider,
            },
            "history": formatted_history,
            "tools": formatted_tools,
            "response": {
                "content": response.content,
                "thought": response.thought,
                "tool_calls": formatted_tool_calls,
                "raw_response": response.raw_response,
                "metrics": {
                    "prompt_tokens": response.prompt_tokens,
                    "candidates_tokens": response.candidates_tokens,
                    "total_tokens": response.total_tokens,
                },
            },
        }

        # Write to target YAML file
        with open(log_filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(log_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logger.info(f"Logged raw LLM turn {idx} to {log_filepath}")

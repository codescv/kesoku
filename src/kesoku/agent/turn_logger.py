"""Handles logging raw LLM turn inputs, outputs, and metrics to session staging directory.

Logs are written as `llm-turn-<N>.log.yaml` under `$STAGING_DIR/.kesoku_logs/` (legacy logs in the staging root are
still honoured when computing the next index). The heavy part (YAML serialization + file write) runs off the event
loop via `alog_llm_turn`; the log payload is snapshotted on the calling thread first so later in-place mutations of
`Message` objects can never race with the background writer.
"""

import asyncio
import datetime
import inspect
import json
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any

import yaml

from kesoku.agent.llm import LLMResponse
from kesoku.db import Message
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

# Subdirectory (inside a session staging dir) holding kesoku-internal logs, kept out of the staging root so
# `os.listdir`-based paths (artifact listings, index scans) stay cheap and the agent's workspace stays clean.
INTERNAL_LOGS_DIRNAME = ".kesoku_logs"

_LLM_TURN_RE = re.compile(r"llm-turn-(\d+)\.log\.yaml")

try:  # LibYAML-backed dumper is ~8x faster than the pure-Python SafeDumper.
    _YamlDumper: Any = yaml.CSafeDumper
except AttributeError:  # pragma: no cover - depends on the PyYAML build
    _YamlDumper = yaml.SafeDumper

# Process-wide next-index reservations keyed by log directory: the main worker and sub-agents each create their own
# TurnLogger for the same session directory, so the counter must be shared to avoid index collisions.
_NEXT_IDX: dict[str, int] = {}
_NEXT_IDX_LOCK = threading.Lock()


def internal_logs_dir(session_staging_dir: str) -> str:
    """Return the kesoku-internal log directory for `session_staging_dir`."""
    return os.path.join(session_staging_dir, INTERNAL_LOGS_DIRNAME)


def _scan_max_llm_turn_idx(directory: str) -> int:
    """Return the largest `llm-turn-<N>` index found directly inside `directory` (0 if none or missing)."""
    max_idx = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                match = _LLM_TURN_RE.match(entry.name)
                if match:
                    max_idx = max(max_idx, int(match.group(1)))
    except FileNotFoundError:
        return 0
    except Exception as e:
        logger.warning(f"Error scanning {directory} for turn logs: {e}")
    return max_idx


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

    @property
    def logs_dir(self) -> str:
        """Directory receiving new `llm-turn-*.log.yaml` files."""
        return internal_logs_dir(self.session_staging_dir)

    def _get_next_llm_turn_idx(self) -> int:
        """Determine the next turn index by scanning the staging root and the internal logs directory.

        Returns:
            The next turn index starting from 1.
        """
        if not os.path.exists(self.session_staging_dir):
            return 1
        return max(_scan_max_llm_turn_idx(self.session_staging_dir), _scan_max_llm_turn_idx(self.logs_dir)) + 1

    def _idx_key(self) -> str:
        """Key of this logger's directory in the process-wide index registry."""
        return os.path.abspath(self.session_staging_dir)

    def _reserve_llm_turn_idx(self) -> int:
        """Reserve the next turn index, scanning the filesystem only once per directory per process."""
        key = self._idx_key()
        with _NEXT_IDX_LOCK:
            idx = _NEXT_IDX.get(key)
            if idx is None:
                idx = self._get_next_llm_turn_idx()
            _NEXT_IDX[key] = idx + 1
            return idx

    def build_log_data(
        self,
        llm_provider: str,
        history: list[Message],
        tools: list[Callable[..., Any]],
        response: LLMResponse,
        system_prompt: str | None = None,
        extra_metrics: dict[str, Any] | None = None,
        idx: int = 0,
    ) -> dict[str, Any]:
        """Build an independent, JSON-safe snapshot of the log payload (deep-copied via a JSON round-trip)."""
        formatted_history = []
        if system_prompt:
            formatted_history.append(
                {
                    "id": f"sys_{self.session_id}",
                    "role": "system",
                    "sender": "System",
                    "type": "text",
                    "content": system_prompt,
                    "metadata": {},
                    "timestamp": time.time() - 1000.0,
                    "status": "responded",
                    "parent_id": None,
                }
            )

        for msg in history:
            formatted_history.append(
                {
                    "id": msg.id,
                    "role": str(msg.role),
                    "sender": msg.sender,
                    "type": str(msg.type),
                    "content": msg.content,
                    "metadata": msg.metadata,
                    "timestamp": msg.timestamp,
                    "status": str(msg.status),
                    "parent_id": msg.parent_id,
                }
            )

        formatted_tools = []
        for func in tools:
            doc = inspect.getdoc(func) or ""
            description = doc.split("\n\n")[0] if doc else ""
            sig = inspect.signature(func)
            parameters = {p_name: str(p.annotation) for p_name, p in sig.parameters.items() if p_name != "context"}
            formatted_tools.append({"name": func.__name__, "description": description, "parameters": parameters})

        formatted_tool_calls = [
            {
                "name": tc.name,
                "arguments": tc.arguments,
                "thought_signature": tc.thought_signature,
                "tool_call_id": tc.tool_call_id,
            }
            for tc in response.tool_calls
        ]

        now_ts = time.time()
        log_data = {
            "metadata": {
                "timestamp": now_ts,
                "timestamp_iso": datetime.datetime.fromtimestamp(now_ts, datetime.UTC).isoformat(),
                "session_id": self.session_id,
                "turn_index": idx,
                "llm_provider": llm_provider,
            },
            "history": formatted_history,
            "tools": formatted_tools,
            # `raw_response` (a multi-KB SDK repr duplicating content/thought/usage) is intentionally not logged.
            "response": {
                "content": response.content,
                "thought": response.thought,
                "tool_calls": formatted_tool_calls,
                "metrics": {
                    "prompt_tokens": response.prompt_tokens,
                    "candidates_tokens": response.candidates_tokens,
                    "cached_tokens": response.cached_tokens,
                    "total_tokens": response.total_tokens,
                    **(extra_metrics or {}),
                },
            },
        }
        # C-accelerated JSON round-trip: a cheap deep copy that also coerces non-JSON values (enums, objects) to str.
        return json.loads(json.dumps(log_data, ensure_ascii=False, default=str))

    def _write_log(self, log_data: dict[str, Any], idx: int) -> str:
        """Serialize `log_data` to `llm-turn-<idx>.log.yaml` (blocking; call from a worker thread)."""
        logs_dir = self.logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        log_filepath = os.path.join(logs_dir, f"llm-turn-{idx}.log.yaml")
        with open(log_filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                log_data, f, Dumper=_YamlDumper, allow_unicode=True, default_flow_style=False, sort_keys=False
            )
        logger.debug(f"Logged raw LLM turn {idx} to {log_filepath}")
        return log_filepath

    def log_llm_turn(
        self,
        llm_provider: str,
        history: list[Message],
        tools: list[Callable[..., Any]],
        response: LLMResponse,
        system_prompt: str | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> str:
        """Synchronously log the raw LLM turn inputs and outputs to a YAML file (blocking).

        Prefer `alog_llm_turn` from async code so serialization and disk I/O stay off the event loop.

        Returns:
            Path of the written log file.
        """
        idx = self._reserve_llm_turn_idx()
        log_data = self.build_log_data(llm_provider, history, tools, response, system_prompt, extra_metrics, idx)
        return self._write_log(log_data, idx)

    async def alog_llm_turn(
        self,
        llm_provider: str,
        history: list[Message],
        tools: list[Callable[..., Any]],
        response: LLMResponse,
        system_prompt: str | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> str:
        """Log the raw LLM turn with the snapshot built on the caller's thread and the write offloaded to a thread.

        Returns:
            Path of the written log file.
        """
        # Snapshot first (synchronously), so no later mutation of the live messages can leak into the log.
        log_data = self.build_log_data(llm_provider, history, tools, response, system_prompt, extra_metrics, 0)
        if self._idx_key() in _NEXT_IDX:
            idx = self._reserve_llm_turn_idx()
        else:
            # First log for this directory in this process: the index scan may list a large legacy staging dir.
            idx = await asyncio.to_thread(self._reserve_llm_turn_idx)
        log_data["metadata"]["turn_index"] = idx
        return await asyncio.to_thread(self._write_log, log_data, idx)

"""Tests for generic per-message overrides in `TurnExecutor` (llm_override, mode_prefs, ignore_context_globs)."""

from typing import Any

import pytest

from kesoku.agent.context_dir import ContextEntry
from kesoku.agent.turn_executor import (
    TurnExecutor,
    _StreamPreviewPublisher,
    filter_ignored_context_entries,
    render_mode_prefs_block,
    validate_llm_override,
)
from kesoku.config import AppConfig, KesokuConfig
from kesoku.constants import MessageRole
from kesoku.db import Message


def test_validate_llm_override_whitelist() -> None:
    allowed = ["m-1", "m-2"]
    assert validate_llm_override({"model_name": "m-1", "thinking_level": "low", "stream": True}, allowed) == {
        "model_name": "m-1",
        "thinking_level": "low",
        "stream": True,
    }
    assert validate_llm_override({"llm": "Claude"}, allowed) == {"llm": "claude"}
    assert validate_llm_override({"model_name": "m-3"}, allowed) is None
    assert validate_llm_override({"model_name": "m-1"}, []) is None
    assert validate_llm_override({"llm": "other"}, allowed) is None
    assert validate_llm_override({"thinking_level": "ultra"}, allowed) is None
    assert validate_llm_override({"stream": "yes"}, allowed) is None
    assert validate_llm_override({"temperature": 2}, allowed) is None
    assert validate_llm_override({}, allowed) is None
    assert validate_llm_override("m-1", allowed) is None


def test_render_mode_prefs_block() -> None:
    block = render_mode_prefs_block({"b": 1, "a": "x</context><evil>"})
    assert block.startswith('<context name="app-prefs">\n')
    assert '"a":"x<\\/context><evil>"' in block and block.count("</context>") == 1
    assert render_mode_prefs_block({}) == ""
    assert render_mode_prefs_block(["x"]) == ""
    assert render_mode_prefs_block({"k": "v" * 3000}) == ""


def test_filter_ignored_context_entries() -> None:
    entries = [ContextEntry(name="notes", content="n"), ContextEntry(name="game-quiz#1", content="g"),
               ContextEntry(name="scene", content="s")]
    assert [e.name for e in filter_ignored_context_entries(entries, ["game-*"])] == ["notes", "scene"]
    assert filter_ignored_context_entries(entries, None) == entries
    assert filter_ignored_context_entries(entries, "game-*") == entries
    assert filter_ignored_context_entries(entries, ["", 3]) == entries


class _Ctx:
    def __init__(self, cfg: KesokuConfig) -> None:
        self.config = cfg
        self.calls: list[dict[str, Any]] = []

    def get_llm(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return type("L", (), {"config": type("C", (), {"stream": False})()})()


class _Gw:
    def __init__(self) -> None:
        self.intervals = {"app": 0.05}

    def has_stream_handler(self, chatbot_id: str) -> bool:
        return True

    def get_stream_min_interval(self, chatbot_id: str) -> float:
        return self.intervals.get(chatbot_id, 1.5)


def _executor(cfg: KesokuConfig) -> tuple[TurnExecutor, _Ctx]:
    ctx = _Ctx(cfg)
    tr = type("TR", (), {"tool_context": type("TC", (), {})()})()
    return TurnExecutor("s1", _Gw(), tr, context=ctx), ctx  # type: ignore[arg-type]


def _msg(chatbot_id: str, metadata: dict[str, Any]) -> Message:
    return Message(session_id="s1", chatbot_id=chatbot_id, channel_id="c", sender="u", role=MessageRole.USER,
                   content="hi", metadata=metadata)


def test_resolve_llm_and_stream_honour_message_override_only_for_allowed_models() -> None:
    cfg = KesokuConfig()
    cfg.chatbots.app = [AppConfig(chatbot_id="app", allowed_models=["m-1"])]
    te, ctx = _executor(cfg)
    te._resolve_llm(_msg("app", {"llm_override": {"model_name": "m-1", "thinking_level": "high"}}))
    assert ctx.calls[-1] == {"model_name": "m-1", "thinking_level": "high"}
    te._resolve_llm(_msg("app", {"llm_override": {"model_name": "not-allowed"}}))
    assert ctx.calls[-1] == {}
    # Channels without an app config never accept per-message model overrides.
    te._resolve_llm(_msg("discord", {"llm_override": {"model_name": "m-1"}}))
    assert ctx.calls[-1] == {}
    llm = ctx.get_llm()
    assert te._resolve_stream(_msg("app", {"llm_override": {"stream": True}}), llm) is True
    assert te._resolve_stream(_msg("app", {}), llm) is False


def test_stream_interval_resolution() -> None:
    te, _ = _executor(KesokuConfig())
    assert te._resolve_stream_min_interval("app") == pytest.approx(0.05)
    assert te._resolve_stream_min_interval("discord") == pytest.approx(1.5)
    pub = _StreamPreviewPublisher(gateway=None, chatbot_id="app", channel_id="c", session_id="s", turn_id="t",  # type: ignore[arg-type]
                                  min_interval=0.05)
    assert pub.min_interval == pytest.approx(0.05)

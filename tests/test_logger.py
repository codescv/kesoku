"""Tests for environment-dependent log handler selection (`kesoku.logger`)."""

import io
import logging

import pytest
from rich.console import Console
from rich.logging import RichHandler

from kesoku import logger as logger_mod


class _Stream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("KESOKU_PLAIN_LOGS", raising=False)
    monkeypatch.delenv("KESOKU_RICH_TRACEBACKS", raising=False)
    return monkeypatch


def _set_streams(mp, stderr_tty: bool, stdout_tty: bool = True) -> _Stream:
    out = _Stream(tty=stdout_tty)
    mp.setattr(logger_mod.sys, "stderr", _Stream(tty=stderr_tty))
    mp.setattr(logger_mod.sys, "stdout", out)
    return out


def test_non_tty_uses_plain_one_line_handler(clean_env) -> None:
    stream = _set_streams(clean_env, stderr_tty=False)
    handler = logger_mod.make_log_handler()
    assert isinstance(handler, logging.StreamHandler) and not isinstance(handler, RichHandler)
    assert handler.formatter is not None and handler.formatter._fmt == logger_mod.PLAIN_LOG_FORMAT
    assert handler.stream is stream

    log = logging.getLogger("kesoku.test_logger.plain")
    log.handlers.clear()
    try:
        returned = logger_mod.setup_logger("kesoku.test_logger.plain")
        assert returned is log and not log.propagate
        try:
            raise ValueError("bad")
        except ValueError:
            log.exception("one line %s", "msg")
    finally:
        log.handlers.clear()
    out = stream.getvalue()
    assert " ERROR kesoku.test_logger.plain: one line msg\n" in out
    assert "Traceback (most recent call last)" in out  # plain stdlib traceback, no Rich rendering


def test_piped_stdout_also_selects_plain_handler(clean_env) -> None:
    _set_streams(clean_env, stderr_tty=True, stdout_tty=False)
    assert isinstance(logger_mod.make_log_handler(), logger_mod.PlainStreamHandler)


def test_plain_handler_follows_stdout_swaps(clean_env) -> None:
    first = _set_streams(clean_env, stderr_tty=False)
    handler = logger_mod.make_log_handler()
    second = _Stream(tty=False)
    clean_env.setattr(logger_mod.sys, "stdout", second)
    handler.handle(logging.makeLogRecord({"name": "x", "levelno": logging.INFO, "levelname": "INFO", "msg": "hi"}))
    assert first.getvalue() == "" and "INFO x: hi" in second.getvalue()


def test_env_forces_plain_logs_on_tty(clean_env) -> None:
    _set_streams(clean_env, stderr_tty=True)
    clean_env.setenv("KESOKU_PLAIN_LOGS", "1")
    assert not isinstance(logger_mod.make_log_handler(), RichHandler)


def test_tty_uses_rich_without_tracebacks_unless_opted_in(clean_env) -> None:
    _set_streams(clean_env, stderr_tty=True)
    handler = logger_mod.make_log_handler()
    assert isinstance(handler, RichHandler) and handler.rich_tracebacks is False
    assert handler.console is logger_mod.console
    clean_env.setenv("KESOKU_RICH_TRACEBACKS", "1")
    rich_handler = logger_mod.make_log_handler()
    assert isinstance(rich_handler, RichHandler) and rich_handler.rich_tracebacks is True


def test_configure_logging_replaces_root_handlers(clean_env) -> None:
    _set_streams(clean_env, stderr_tty=False)
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        root.addHandler(logging.NullHandler())
        logger_mod.configure_logging(logging.DEBUG)
        assert root.level == logging.DEBUG and len(root.handlers) == 1
        assert isinstance(root.handlers[0], logger_mod.PlainStreamHandler)
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_shared_console_export() -> None:
    assert isinstance(logger_mod.console, Console)

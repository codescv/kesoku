"""Logging configuration and console loggers for Kesoku.

Interactive terminals get Rich console logging (timestamps, levels, colors). When stderr or stdout is not a TTY
(systemd / journald, pipes, test runners) or `KESOKU_PLAIN_LOGS=1`, a plain one-line `logging.StreamHandler` is used
instead: Rich's syntax-highlighted tracebacks cost ~250 ms per record on the event loop and wrap lines at 80 columns
in the journal. Rich tracebacks are opt-in via `KESOKU_RICH_TRACEBACKS=1`.

Both handlers write to the *current* `sys.stdout` (like the shared Rich `console`), so CLI commands that report
through `logger.info` keep printing to stdout.
"""

import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler

# Global shared console used for both status spinners and Rich logging
console = Console()

PLAIN_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_TRUTHY = ("1", "true", "yes", "on")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _is_tty(stream: object) -> bool:
    try:
        return bool(stream is not None and stream.isatty())  # type: ignore[attr-defined]
    except (AttributeError, ValueError, OSError):
        return False


def use_plain_logs() -> bool:
    """Return True for the plain one-line handler: non-TTY stderr/stdout (systemd) or `KESOKU_PLAIN_LOGS=1`."""
    return _env_flag("KESOKU_PLAIN_LOGS") or not (_is_tty(sys.stderr) and _is_tty(sys.stdout))


class PlainStreamHandler(logging.StreamHandler):
    """`StreamHandler` that always writes to the *current* `sys.stdout` (survives stream swaps, e.g. CLI runners)."""

    def __init__(self) -> None:
        """Create the handler with the concise one-line formatter."""
        super().__init__()
        self.setFormatter(logging.Formatter(PLAIN_LOG_FORMAT))

    @property  # type: ignore[override]
    def stream(self):  # noqa: ANN201 - mirrors `logging.StreamHandler.stream`
        """The live `sys.stdout` (same destination as the shared Rich `console`)."""
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:  # noqa: ANN001 - assignments from `StreamHandler.__init__` are ignored
        pass


def make_log_handler() -> logging.Handler:
    """Build the log handler for the current environment (plain `StreamHandler` or `RichHandler`)."""
    if use_plain_logs():
        return PlainStreamHandler()
    return RichHandler(console=console, rich_tracebacks=_env_flag("KESOKU_RICH_TRACEBACKS"), markup=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with the environment-appropriate handler for all logging output across Kesoku.

    Args:
        level: Logging severity level threshold.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.addHandler(make_log_handler())


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up and return an individual logger with the environment-appropriate handler.

    Args:
        name: Module logger name.
        level: Logging severity level threshold.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(make_log_handler())
    # Prevent duplication if root logger is also configured
    logger.propagate = False
    return logger

"""Logging configuration and Rich console loggers for Kesoku.

Provides standardized Rich console loggers with tracebacks, timestamps, and levels.
"""

import logging
from rich.console import Console
from rich.logging import RichHandler

# Global shared console used for both status spinners and Rich logging
console = Console()


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger to apply RichHandler to all logging output across Kesoku.

    Args:
        level: Logging severity level threshold.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing uncolored handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = RichHandler(console=console, rich_tracebacks=True, markup=False)
    root_logger.addHandler(handler)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up and return an individual logger with RichHandler.

    Args:
        name: Module logger name.
        level: Logging severity level threshold.

    Returns:
        Configured logging.Logger instance.
    """
    handler = RichHandler(console=console, rich_tracebacks=True, markup=False)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    # Prevent duplication if root logger is also configured
    logger.propagate = False
    return logger

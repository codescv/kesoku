"""Logging configuration and ANSI colored formatting for Kesoku.

Provides standardized colored console loggers with filename and line numbers.
"""

import logging


class ColorFormatter(logging.Formatter):
    """ANSI colored logging formatter displaying timestamps, process, thread, filename, line numbers, and levels."""

    BOLD = "\033[1m"
    RESET = "\033[0m"
    CYAN = "\033[36m"
    COLORS = {
        logging.DEBUG: "\033[34m",  # Blue
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }

    def __init__(self) -> None:
        """Initialize the ColorFormatter with format string including filename and lineno."""
        super().__init__(
            "%(asctime)s - [PID:%(process)d|T:%(thread)d] - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
        )

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Format timestamp with bold cyan styling.

        Args:
            record: LogRecord instance.
            datefmt: Optional format string.

        Returns:
            Colored timestamp string.
        """
        time_str = super().formatTime(record, datefmt)
        return f"{self.BOLD}{self.CYAN}{time_str}{self.RESET}"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record levelname with corresponding ANSI color.

        Args:
            record: LogRecord instance.

        Returns:
            Fully formatted log string.
        """
        orig_levelname = record.levelname
        level_color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{self.BOLD}{level_color}{orig_levelname}{self.RESET}"

        result = super().format(record)

        record.levelname = orig_levelname
        return result


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger to apply ColorFormatter to all logging output across Kesoku.

    Args:
        level: Logging severity level threshold.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing uncolored handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    root_logger.addHandler(handler)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up and return an individual logger with ColorFormatter.

    Args:
        name: Module logger name.
        level: Logging severity level threshold.

    Returns:
        Configured logging.Logger instance.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    # Prevent duplication if root logger is also configured
    logger.propagate = False
    return logger

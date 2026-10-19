"""Word-boundary text truncation helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def truncate_words(text: str, max_words: int, suffix: str = "...") -> str:
    """Truncate text to at most `max_words` whitespace-delimited words.

    Args:
        text: Input string to truncate.
        max_words: Maximum number of words to retain.
        suffix: Trailing marker appended when truncation occurs.

    Returns:
        Truncated string with `suffix` appended if word count exceeded `max_words`.
    """
    if max_words <= 0:
        logger.debug("truncate_words called with non-positive max_words=%d", max_words)
        return ""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + suffix

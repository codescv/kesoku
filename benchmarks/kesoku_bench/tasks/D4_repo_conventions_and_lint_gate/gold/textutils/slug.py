"""URL-safe slug generation helpers."""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


def slugify(
    text: str,
    max_length: int | None = None,
    allow_unicode: bool = False,
    separator: str = "-",
) -> str:
    """Convert text into a normalized URL-safe slug.

    Args:
        text: Input string to convert into a slug.
        max_length: Optional maximum character length of the returned slug.
        allow_unicode: If True, retain non-ASCII alphanumeric characters (NFKC).
        separator: Separator string used between words (defaults to '-').

    Returns:
        Normalized slug string with consecutive and boundary separators removed.
    """
    if not text:
        return ""

    if allow_unicode:
        normalized = unicodedata.normalize("NFKC", text).lower()
        chars: list[str] = []
        for ch in normalized:
            cat = unicodedata.category(ch)
            if ch.isalnum() and not cat.startswith(("P", "S", "Z")):
                chars.append(ch)
            else:
                chars.append(separator)
        raw = "".join(chars)
    else:
        decomposed = unicodedata.normalize("NFKD", text)
        stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()
        raw = "".join(
            ch if ("a" <= ch <= "z" or "0" <= ch <= "9") else separator
            for ch in stripped
        )

    if separator:
        collapsed = re.sub(rf"{re.escape(separator)}+", separator, raw).strip(separator)
    else:
        collapsed = raw

    if max_length is not None and max_length > 0:
        collapsed = collapsed[:max_length]
        if separator:
            collapsed = collapsed.strip(separator)

    if not collapsed:
        logger.debug("empty slug for input: %r", text)

    return collapsed

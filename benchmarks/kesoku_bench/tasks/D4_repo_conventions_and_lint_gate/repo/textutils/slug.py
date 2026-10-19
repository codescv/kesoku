"""URL-safe slug generation helpers."""

from __future__ import annotations

import unicodedata


def slugify(
    text: str,
    max_length: int | None = None,
    allow_unicode: bool = False,
    separator: str = "-",
) -> str:
    """Convert text into a normalized URL slug."""
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

    cleaned = raw.strip(separator)
    collapsed = cleaned.replace(separator + separator, separator)

    if max_length is not None and max_length > 0:
        collapsed = collapsed[:max_length]

    if not collapsed:
        print(f"[textutils.slug] empty slug for input: {text!r}")

    return collapsed

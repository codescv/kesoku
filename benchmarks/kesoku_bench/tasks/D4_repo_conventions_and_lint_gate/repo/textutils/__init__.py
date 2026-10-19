"""Text normalization, slugify, and word-boundary truncation utilities."""

from textutils.slug import slugify
from textutils.truncate import truncate_words

__all__ = ["slugify", "truncate_words"]

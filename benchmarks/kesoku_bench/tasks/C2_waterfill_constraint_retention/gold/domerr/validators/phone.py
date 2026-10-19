"""Phone validator (E.164)."""

import re
from typing import Any

from domerr.codes import ErrorCode
from domerr.errors import DomainError
from domerr.registry import validator

__all__ = ["validate_phone"]

_SEPARATORS = re.compile(r"[\s\-.()]")
_E164 = re.compile(r"\+\d{8,15}")


@validator("phone")
def validate_phone(value: Any) -> str:
    """Normalise a phone number to E.164.

    Args:
        value: Raw phone value.

    Returns:
        ``+`` followed by 8-15 digits.

    Raises:
        DomainError: ``E_DOM_PHONE_001`` if missing/not a string, ``E_DOM_PHONE_002`` if not E.164.
    """
    if not isinstance(value, str) or not value.strip():
        raise DomainError(ErrorCode.E_DOM_PHONE_001, "phone is required", field_path="phone")
    text = _SEPARATORS.sub("", value.strip())
    if text.startswith("00"):
        text = "+" + text[2:]
    if not _E164.fullmatch(text):
        raise DomainError(ErrorCode.E_DOM_PHONE_002, "phone must be E.164", field_path="phone")
    return text

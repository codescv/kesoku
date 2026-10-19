"""Email validator."""

from typing import Any

from domerr.codes import ErrorCode
from domerr.errors import DomainError
from domerr.registry import validator

__all__ = ["validate_email"]


@validator("email")
def validate_email(value: Any) -> str:
    """Normalise an email address (strip, lower-case the domain).

    Args:
        value: Raw email value.

    Returns:
        The normalised address.

    Raises:
        DomainError: ``E_DOM_EMAIL_001`` if missing/not a string, ``E_DOM_EMAIL_002`` if malformed.
    """
    if not isinstance(value, str) or not value.strip():
        raise DomainError(ErrorCode.E_DOM_EMAIL_001, "email is required", field_path="email")
    text = value.strip()
    if text.count("@") != 1:
        raise DomainError(ErrorCode.E_DOM_EMAIL_002, "email needs exactly one '@'", field_path="email")
    local, domain = text.split("@")
    if not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
        raise DomainError(ErrorCode.E_DOM_EMAIL_002, "email is malformed", field_path="email")
    return f"{local}@{domain.lower()}"

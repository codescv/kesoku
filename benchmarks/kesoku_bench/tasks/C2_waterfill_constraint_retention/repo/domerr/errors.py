"""The single exception type raised by domerr validators."""

from typing import Any

from domerr.codes import ErrorCode


class DomainError(Exception):
    """A validation failure with a machine-readable code.

    Attributes:
        code: The :class:`ErrorCode` member describing the failure.
        message: Human-readable explanation.
        field_path: Dotted path of the offending field, or ``None`` when not applicable.
    """

    def __init__(self, code: ErrorCode | str, message: str = "", field_path: str | None = None) -> None:
        """Create the error.

        Args:
            code: An :class:`ErrorCode` member (or its string value).
            message: Human-readable explanation.
            field_path: Dotted path of the offending field.

        Raises:
            ValueError: If ``code`` is not a declared :class:`ErrorCode`.
        """
        self.code = ErrorCode(code)
        self.message = message
        self.field_path = field_path
        super().__init__(str(self))

    def __str__(self) -> str:
        """Render as ``[CODE] path: message``."""
        where = f" {self.field_path}:" if self.field_path else ""
        return f"[{self.code.value}]{where} {self.message}".rstrip()

    def to_dict(self) -> dict[str, Any]:
        """Serialise for API responses."""
        return {"code": self.code.value, "message": self.message, "field_path": self.field_path}

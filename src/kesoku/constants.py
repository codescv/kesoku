"""System-wide constants and enums for Kesoku AI Agent framework."""

from enum import StrEnum


class MessageRole(StrEnum):
    """Message roles within the Kesoku framework."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class MessageType(StrEnum):
    """Message types within the Kesoku framework."""

    TEXT = "text"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class MessageStatus(StrEnum):
    """Message lifecycle statuses within the Kesoku framework."""

    # User message statuses
    PENDING_AGENT = "pending_agent"
    PROCESSING = "processing"
    PROCESSED = "processed"
    INTERRUPTED = "interrupted"
    ERROR = "error"

    # Assistant & Tool message statuses
    PENDING = "pending"
    DELIVERED = "delivered"
    RESPONDED = "responded"


# Backward compatibility aliases (as strings via StrEnum inheritance)
ROLE_USER = MessageRole.USER
ROLE_ASSISTANT = MessageRole.ASSISTANT
ROLE_TOOL = MessageRole.TOOL
ROLE_SYSTEM = MessageRole.SYSTEM

TYPE_TEXT = MessageType.TEXT
TYPE_THOUGHT = MessageType.THOUGHT
TYPE_TOOL_CALL = MessageType.TOOL_CALL
TYPE_TOOL_RESULT = MessageType.TOOL_RESULT

STATUS_PENDING_AGENT = MessageStatus.PENDING_AGENT
STATUS_PROCESSING = MessageStatus.PROCESSING
STATUS_PROCESSED = MessageStatus.PROCESSED
STATUS_INTERRUPTED = MessageStatus.INTERRUPTED
STATUS_ERROR = MessageStatus.ERROR
STATUS_PENDING = MessageStatus.PENDING
STATUS_DELIVERED = MessageStatus.DELIVERED
STATUS_RESPONDED = MessageStatus.RESPONDED

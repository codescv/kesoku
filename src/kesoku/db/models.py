"""Database Pydantic models for Kesoku AI Agent."""

import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from kesoku.constants import MessageRole, MessageStatus, MessageType


class Message(BaseModel):
    """Represents a conversational message within the Kesoku framework."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(..., description="Internal unique conversational session identifier")
    chatbot_id: str = Field(..., description="Unique identifier of the chatbot platform/instance (e.g., 'cli')")
    channel_id: str = Field(..., description="External platform-specific channel or room identifier")
    sender: str = Field(..., description="Sender identifier or username")
    role: MessageRole = Field(default=MessageRole.USER, description="Role of the message sender")
    type: MessageType = Field(default=MessageType.TEXT, description="Type of message content or action")
    content: str = Field(..., description="Text content of the message")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata for platform-specific attributes (e.g., attachments, guild_id)",
    )
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of when the message was created")
    status: MessageStatus = Field(default=MessageStatus.PENDING, description="Current lifecycle status of the message")
    parent_id: str | None = Field(default=None, description="Links tool results or followups to specific message/call")


class Session(BaseModel):
    """Represents a persistent conversational chat session."""

    id: str = Field(..., description="Unique alphanumeric identifier for the session")
    title: str = Field(..., description="Summary title or first message snippet")
    created_at: float = Field(default_factory=time.time, description="Creation unix timestamp")
    updated_at: float = Field(default_factory=time.time, description="Last updated unix timestamp")
    system_prompt: str = Field(default="", description="The main system prompt instructions for the session")

    @property
    def workspace_name(self) -> str:
        """Generate a unique, escaped folder name for the session workspace.

        Returns:
            An escaped folder name string.
        """
        escaped = re.sub(r"[^\w\-]", "_", self.title)
        escaped = re.sub(r"_+", "_", escaped)
        escaped = escaped.strip("_")
        if not escaped:
            escaped = "session"
        title_escaped = escaped[:20].strip("_")
        ts_str = time.strftime("%y%m%d-%H-%M", time.localtime(self.created_at))
        return f"{ts_str}_{title_escaped}_{self.id}"


class AgentMemory(BaseModel):
    """Represents a structured agent memory record in the SQLite database."""

    id: int | None = Field(default=None, description="Database autoincrement primary key")
    category: str = Field(..., description="Category: 'progress', 'learnings', 'user_preferences' etc.")
    key: str = Field(..., description="snake_case unique identifier")
    title: str = Field(..., description="Human-readable label or title for the entry")
    content: str = Field(..., description="Markdown text or structured content payload")
    updated_at: float = Field(default_factory=time.time, description="Unix timestamp of last update")
    role: str = Field(default="default", description="Optional roleplay-specific character persona binding")


class CrossSessionContext(BaseModel):
    """Represents a cross-session context/memory summary for a specific persona role."""

    role: str = Field(..., description="The unique role/persona identifier")
    content: str = Field(..., description="The summarized memory/context content string")
    updated_at: float = Field(default_factory=time.time, description="Unix timestamp of last consolidation")
    status: str = Field(default="idle", description="Lock status: 'idle' or 'updating'")

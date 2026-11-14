"""Conversion of parsed message segments and stored messages into the client wire shapes."""

from __future__ import annotations

import mimetypes
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

from kesoku.constants import MessageRole
from kesoku.db.models import Message
from kesoku.gateway.chatbot.base import parse_message_content
from kesoku.utils.ui_markup import card_media_refs

_ATTACHMENTS_SUFFIX_RE = re.compile(r"\n*\s*Attachments:\n(?:\[Attachment: [^\n]*\]\n?)+\s*$")


@dataclass
class MediaJob:
    """A media reference in a live `message.final` that must be delivered as `media.*` events."""

    placeholder_id: str
    kind: str  # "file" | "voice"
    ref: str


def new_placeholder_id() -> str:
    """Return a fresh placeholder id."""
    return f"ph_{uuid.uuid4().hex[:16]}"


def _mime_for(ref: str) -> str | None:
    mime, _ = mimetypes.guess_type(ref)
    if mime:
        return mime
    return {".webp": "image/webp", ".opus": "audio/ogg"}.get(os.path.splitext(ref)[1].lower())


def to_wire_segments(
    segments: list[dict[str, Any]],
    *,
    message_id: str,
    live: bool,
) -> tuple[list[dict[str, Any]], list[MediaJob]]:
    """Convert internal `parse_message_content(..., rich_blocks=True)` segments to wire segments.

    Args:
        segments: Internal segments.
        message_id: Source assistant message id (used for questions).
        live: When True, every media reference gets a `placeholder_id` and a `MediaJob` is returned for it.

    Returns:
        `(wire_segments, media_jobs)`.
    """
    wire: list[dict[str, Any]] = []
    jobs: list[MediaJob] = []

    def _media(kind: str, ref: str) -> dict[str, Any]:
        item: dict[str, Any] = {"ref": ref}
        if live:
            pid = new_placeholder_id()
            item["placeholder_id"] = pid
            jobs.append(MediaJob(placeholder_id=pid, kind=kind, ref=ref))
        return item

    for seg in segments:
        stype = seg.get("type")
        if stype == "text":
            text = str(seg.get("content") or "")
            if text.strip():
                wire.append({"type": "text", "markdown": text})
        elif stype == "question":
            wire.append(
                {
                    "type": "question",
                    "prompt": str(seg.get("question") or ""),
                    "choices": list(seg.get("choices") or []),
                    "multi": bool(seg.get("multi", False)),
                    "max": int(seg.get("max", 1) or 1),
                    "question_message_id": message_id,
                }
            )
        elif stype == "details":
            wire.append(
                {"type": "details", "title": str(seg.get("title") or ""), "body_markdown": str(seg.get("body") or "")}
            )
        elif stype in ("file", "voice"):
            ref = str(seg.get("path") or "").strip()
            if not ref:
                continue
            item = {"type": stype, **_media(stype, ref), "name": os.path.basename(ref)}
            mime = _mime_for(ref)
            if mime:
                item["mime"] = mime
            wire.append(item)
        elif stype == "bilingual":
            item = {k: v for k, v in seg.items() if k != "voice"}
            voice = seg.get("voice")
            if isinstance(voice, dict) and voice.get("ref"):
                v = {**voice, **_media("voice", str(voice["ref"]))}
                item["voice"] = v
            wire.append(item)
        elif stype == "card":
            data = seg.get("data") if isinstance(seg.get("data"), dict) else {}
            item = {"type": "card", "kind": seg.get("kind"), "data": data}
            # Live finals map `data` keys holding `[file: ...]` tags to placeholder ids; history keeps only the
            # tags inside `data` (clients extract the ref and call `media.hydrate`).
            refs = card_media_refs(data)
            if refs and live:
                item["media"] = {key: _media(ref["kind"], ref["ref"])["placeholder_id"] for key, ref in refs.items()}
            wire.append(item)
    return wire, jobs


def user_display_text(msg: Message) -> str:
    """Return the human-visible text of a user message (without the injected attachment listing)."""
    meta = msg.metadata or {}
    display = meta.get("display_text")
    if isinstance(display, str):
        return display
    content = msg.content or ""
    if meta.get("attachments"):
        content = _ATTACHMENTS_SUFFIX_RE.sub("", content)
    return content


def message_to_record(
    msg: Message,
    app_chatbot_id: str,
    *,
    role_id: str | None = None,
    role_display_name: str | None = None,
) -> dict[str, Any]:
    """Convert a stored user / assistant text message to a `MessageRecord` (media as refs only).

    `role_id` / `role_display_name` (the session persona) are attached to assistant senders, except
    notifications, so clients can show the persona name instead of the bot name.
    """
    meta = msg.metadata or {}
    if msg.role == MessageRole.USER:
        attachments = []
        for a in meta.get("attachments") or []:
            if isinstance(a, dict) and a.get("path"):
                attachments.append(
                    {
                        "ref": a.get("path"),
                        "mime": a.get("mime_type") or a.get("mime") or "application/octet-stream",
                        "filename": a.get("filename") or os.path.basename(str(a.get("path"))),
                        "object": a.get("object") or "",
                        "bytes": int(a.get("bytes") or 0),
                    }
                )
        record: dict[str, Any] = {
            "message_id": msg.id,
            "role": "user",
            "ts": float(msg.timestamp),
            "text": user_display_text(msg),
            "attachments": attachments,
            "sender": {
                "kind": "user",
                "id": msg.sender,
                "name": meta.get("sender_name") or msg.sender,
                "chatbot_id": msg.chatbot_id,
            },
            "chatbot_id": msg.chatbot_id,
        }
        if meta.get("client_msg_id"):
            record["client_msg_id"] = meta["client_msg_id"]
        if isinstance(meta.get("app_seq"), int):
            record["seq"] = meta["app_seq"]
        return record
    segments, _ = to_wire_segments(parse_message_content(msg.content or "", rich_blocks=True), message_id=msg.id,
                                   live=False)
    record = {
        "message_id": msg.id,
        "role": "assistant" if msg.role == MessageRole.ASSISTANT else "system",
        "ts": float(msg.timestamp),
        "segments": segments,
        "raw_markdown": msg.content or "",
        "sender": {"kind": "assistant", "name": msg.sender, "chatbot_id": msg.chatbot_id},
        "chatbot_id": msg.chatbot_id,
    }
    if msg.role == MessageRole.ASSISTANT and msg.sender != "Notification":
        if role_id:
            record["sender"]["role_id"] = role_id
        if role_display_name:
            record["sender"]["display_name"] = role_display_name
    if msg.parent_id:
        record["turn_id"] = msg.parent_id
    if isinstance(meta.get("app_seq"), int):
        record["seq"] = meta["app_seq"]
    _ = app_chatbot_id
    return record

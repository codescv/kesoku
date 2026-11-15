"""v0.3.7: user-message attachments on history records (voice / image pills instead of server paths)."""

from kesoku.constants import MessageRole
from kesoku.db import Message
from kesoku.gateway.chatbot.app.segments import message_to_record, user_attachments, user_display_text

_LISTING = (
    "\n\nAttachments:\n"
    "[Attachment: voice.m4a (audio/mp4) saved at /ws/s1/uploads/voice.m4a — image/audio attachments are already "
    "visible to you in this message; do not call analyze_media on them]\n"
    "[Attachment: menu.jpg (image/jpeg) saved at /ws/s1/uploads/menu.jpg — image/audio attachments are already "
    "visible to you in this message; do not call analyze_media on them]"
)


def _user(content: str, metadata: dict | None = None) -> Message:
    return Message(
        session_id="s1", chatbot_id="app", channel_id="app:s1", sender="dev-1", role=MessageRole.USER,
        content=content, metadata=metadata or {},
    )


def test_record_exposes_structured_attachments_and_clean_text() -> None:
    meta = {
        "display_text": "帮我听听",
        "attachments": [
            {"path": "/ws/s1/uploads/voice.m4a", "mime_type": "audio/mp4", "filename": "voice.m4a",
             "object": "media/s1/voice.m4a", "bytes": 1234},
        ],
    }
    rec = message_to_record(_user("帮我听听" + _LISTING, meta), "app")
    assert rec["text"] == "帮我听听"
    assert rec["attachments"] == [
        {"ref": "/ws/s1/uploads/voice.m4a", "mime": "audio/mp4", "filename": "voice.m4a",
         "object": "media/s1/voice.m4a", "bytes": 1234},
    ]


def test_listing_is_stripped_and_parsed_without_metadata() -> None:
    """Cross-channel / legacy rows without `attachments` metadata: strip the listing and recover the pills."""
    msg = _user("这是什么？" + _LISTING)
    assert user_display_text(msg) == "这是什么？"
    atts = user_attachments(msg)
    assert [(a["filename"], a["mime"], a["ref"]) for a in atts] == [
        ("voice.m4a", "audio/mp4", "/ws/s1/uploads/voice.m4a"),
        ("menu.jpg", "image/jpeg", "/ws/s1/uploads/menu.jpg"),
    ]
    rec = message_to_record(msg, "app")
    assert "saved at" not in rec["text"] and "Attachments:" not in rec["text"]
    assert len(rec["attachments"]) == 2


def test_attachment_only_message_has_empty_text() -> None:
    msg = _user("Attachments:\n[Attachment: rec.webm (audio/webm) saved at /ws/s1/uploads/rec.webm]")
    rec = message_to_record(msg, "app")
    assert rec["text"] == ""
    assert rec["attachments"][0]["mime"] == "audio/webm" and rec["attachments"][0]["object"] == ""


def test_metadata_mime_fallback_from_extension() -> None:
    msg = _user("hi", {"attachments": [{"path": "/ws/a/clip.wav"}]})
    assert user_attachments(msg)[0]["mime"] == "audio/x-wav" or user_attachments(msg)[0]["mime"].startswith("audio/")


def test_plain_text_mentioning_attachments_is_untouched() -> None:
    msg = _user("Attachments: none today")
    assert user_display_text(msg) == "Attachments: none today"
    assert user_attachments(msg) == []

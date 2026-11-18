"""Conversion of parsed message segments and stored messages into the client wire shapes."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from kesoku.constants import MessageRole
from kesoku.db.models import Message
from kesoku.gateway.chatbot.base import parse_message_content
from kesoku.logger import setup_logger
from kesoku.utils.ui_markup import card_media_refs

logger = setup_logger(__name__)

_ATTACHMENTS_SUFFIX_RE = re.compile(r"\n*\s*Attachments:\n(?:\[Attachment: [^\n]*\]\n?)+\s*$")
# One `[Attachment: <filename> (<mime>) saved at <path>[ — note]]` line of the injected listing.
_ATTACHMENT_LINE_RE = re.compile(
    r"^\[Attachment: (?P<filename>.*?) \((?P<mime>[^()\s]*)\) saved at (?P<path>.+?)(?: — [^\n]*)?\]$",
    re.MULTILINE,
)


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


# ---------------------------------------------------------------------------------------------------------------
# Character-picker enrichment: question options that name registered characters render as avatar cards.
# The registry is generic: `<workspace_dir>/characters/*/characters.json` lists `{name, aliases?, image|avatar?,
# draftable?}` entries; nothing here knows about any particular game or roster.

# Options asking for a fresh batch of candidates do not count towards the "mostly characters" ratio.
_REROLL_RE = re.compile(r"再帮我想|换一批|更多|刷新")
# Optional leading list index (`1. `, `2、`, `3) `) or bullet.
_LEADING_INDEX_RE = re.compile(r"^\s*(?:\d{1,3}\s*[.、)）]\s*|[-*•]\s+)")
# Optional single leading emoji (with variation selector / keycap) followed by whitespace.
_LEADING_EMOJI_RE = re.compile(
    "^[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u3297\u3299][\uFE0F\u20E3]?\\s*"
)
# Characters allowed right after a character name at the start of an option.
_NAME_DELIMITERS = frozenset("（(·｜|：:—-/")
# Hiragana, katakana and CJK ideographs: aliases must contain one of these to be used for detection.
_CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
# The part of an option before its first parenthesis must stay short to count as "name (subtitle)".
_MAX_NAME_PREFIX_CHARS = 14
_PAREN_PAIRS = {"（": "）", "(": ")"}


@dataclass(frozen=True)
class _CharacterRef:
    """A lookup key in the character registry."""

    name: str  # canonical `name` of the first registered entry for this key
    avatar: str  # workspace-relative avatar path ("" when no existing file)
    draftable: bool  # True when any entry registered under this key is draftable (eligible for auto cards)


_registry_lock = threading.Lock()
# realpath(workspace_dir) -> (signature, keys sorted longest-first, {lowercased key: ref})
_registry_cache: dict[str, tuple[tuple[Any, ...], list[str], dict[str, _CharacterRef]]] = {}


def _registry_files(workspace_dir: str) -> list[str]:
    """Return sorted `characters/*/characters.json` paths under `workspace_dir`."""
    root = os.path.join(workspace_dir, "characters")
    try:
        subs = sorted(os.listdir(root))
    except OSError:
        return []
    out: list[str] = []
    for sub in subs:
        path = os.path.join(root, sub, "characters.json")
        if os.path.isfile(path):
            out.append(path)
    return out


def _resolve_avatar(workspace_dir: str, folder: str, raw: str) -> str:
    """Return the workspace-relative path of an existing avatar file, or "" when it cannot be found.

    Paths are tried relative to the workspace first, then relative to the entry's character folder; the
    result must stay inside the workspace.
    """
    raw = raw.strip()
    if not raw or os.path.isabs(raw):
        return ""
    ws_real = os.path.realpath(workspace_dir)
    for base in (workspace_dir, folder):
        cand = os.path.realpath(os.path.join(base, raw))
        if (cand == ws_real or cand.startswith(ws_real + os.sep)) and os.path.isfile(cand):
            return os.path.relpath(cand, ws_real).replace(os.sep, "/")
    return ""


def _build_registry(workspace_dir: str, files: list[str]) -> dict[str, _CharacterRef]:
    """Build the `{lowercased key: _CharacterRef}` lookup from registry files (first folder wins duplicates)."""
    refs: dict[str, _CharacterRef] = {}
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                entries = json.load(f)
        except (OSError, ValueError) as e:
            logger.debug(f"Skipping unreadable character registry {path}: {e}")
            continue
        if not isinstance(entries, list):
            continue
        folder = os.path.dirname(path)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            avatar = _resolve_avatar(workspace_dir, folder, str(entry.get("avatar") or entry.get("image") or ""))
            draftable = entry.get("draftable") is not False
            keys = [name]
            aliases = entry.get("aliases")
            if isinstance(aliases, list):
                keys.extend(
                    a.strip() for a in aliases if isinstance(a, str) and len(a.strip()) >= 2 and _CJK_RE.search(a)
                )
            for key in keys:
                low = key.lower()
                prev = refs.get(low)
                if prev is None:
                    refs[low] = _CharacterRef(name=name, avatar=avatar, draftable=draftable)
                elif draftable and not prev.draftable:
                    # Keep the first folder's identity/avatar but remember a draftable entry shares the key.
                    refs[low] = _CharacterRef(name=prev.name, avatar=prev.avatar or avatar, draftable=True)
                elif not prev.avatar and avatar:
                    refs[low] = _CharacterRef(name=prev.name, avatar=avatar, draftable=prev.draftable)
    return refs


def load_character_registry(workspace_dir: str | None = None) -> tuple[list[str], dict[str, _CharacterRef]]:
    """Return `(keys longest-first, {lowercased key: ref})` for `workspace_dir`, cached by registry file mtimes.

    Args:
        workspace_dir: Agent working directory (defaults to the current working directory).

    Returns:
        Lookup keys (lowercased, longest first) and the key -> character reference map.
    """
    ws = os.path.realpath(workspace_dir or os.getcwd())
    files = _registry_files(ws)
    signature: list[Any] = []
    for path in files:
        try:
            st = os.stat(path)
        except OSError:
            continue
        signature.append((path, st.st_mtime_ns, st.st_size))
    sig = tuple(signature)
    with _registry_lock:
        cached = _registry_cache.get(ws)
        if cached is not None and cached[0] == sig:
            return cached[1], cached[2]
    refs = _build_registry(ws, files) if files else {}
    keys = sorted(refs, key=len, reverse=True)
    with _registry_lock:
        _registry_cache[ws] = (sig, keys, refs)
    return keys, refs


def _strip_option_prefix(option: str) -> str:
    """Strip an optional leading list index / bullet and a single leading emoji from an option."""
    text = _LEADING_INDEX_RE.sub("", option, count=1)
    text = _LEADING_EMOJI_RE.sub("", text, count=1)
    return text.strip()


def match_character_option(
    option: str,
    keys: list[str],
    refs: dict[str, _CharacterRef],
    *,
    draftable_only: bool,
) -> dict[str, str] | None:
    """Match an option that starts with a registered character name, returning `{name, subtitle?, avatar?}`.

    The name must be followed by end-of-string, whitespace or a delimiter (`（ ( · ｜ | ： : — - /`), and the
    text before the first parenthesis must be at most `_MAX_NAME_PREFIX_CHARS` characters, so that free-form
    action sentences mentioning a character are never treated as picker options.
    """
    text = _strip_option_prefix(option)
    if not text:
        return None
    paren_idx = min((i for i in (text.find("（"), text.find("(")) if i >= 0), default=len(text))
    if len(text[:paren_idx].rstrip()) > _MAX_NAME_PREFIX_CHARS:
        return None
    low = text.lower()
    for key in keys:
        if not low.startswith(key):
            continue
        rest = text[len(key) :]
        if rest and not (rest[0].isspace() or rest[0] in _NAME_DELIMITERS):
            continue
        ref = refs[key]
        if draftable_only and not ref.draftable:
            continue
        rest = rest.strip()
        closing = _PAREN_PAIRS.get(rest[:1])
        if closing and rest.endswith(closing):
            subtitle = rest[1:-1].strip()
        else:
            subtitle = rest.lstrip("".join(_NAME_DELIMITERS) + " ").strip()
        out = {"name": text[: len(key)]}
        if subtitle:
            out["subtitle"] = subtitle
        if ref.avatar:
            out["avatar"] = ref.avatar
        return out
    return None


def _meta_for_choice(
    idx: int,
    choice: str,
    option_items: list[dict[str, Any]],
    by_value: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return the `option_items` entry for `(idx, choice)`, preferring positional alignment when it matches."""
    if idx < len(option_items) and str(option_items[idx].get("value") or "") == choice:
        return option_items[idx]
    return by_value.get(choice) or {}


def _explicit_choice_items(
    choices: list[str],
    option_items: list[dict[str, Any]],
    keys: list[str],
    refs: dict[str, _CharacterRef],
) -> list[dict[str, Any]]:
    """Build `choice_items` for an explicitly card-laid-out question, filling missing avatars from the registry."""
    by_value = {str(i.get("value")): i for i in option_items if isinstance(i, dict) and i.get("value")}
    out: list[dict[str, Any]] = []
    for idx, choice in enumerate(choices):
        meta = _meta_for_choice(idx, choice, option_items, by_value)
        item: dict[str, Any] = {"value": choice}
        for field in ("name", "subtitle", "avatar", "group"):
            if meta.get(field):
                item[field] = str(meta[field])
        if "avatar" not in item:
            probe = item.get("name") or choice
            matched = match_character_option(str(probe), keys, refs, draftable_only=False)
            if matched is None and probe != choice:
                matched = match_character_option(choice, keys, refs, draftable_only=False)
            if matched:
                for field, value in matched.items():
                    item.setdefault(field, value)
        item.setdefault("name", choice)
        out.append(item)
    return out


def _metadata_choice_items(
    choices: list[str],
    option_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build `choice_items` from structured `option_items` (`value`, `name?`, `subtitle?`, `group?`)."""
    by_value = {str(i.get("value")): i for i in option_items if isinstance(i, dict) and i.get("value")}
    out: list[dict[str, Any]] = []
    for idx, choice in enumerate(choices):
        meta = _meta_for_choice(idx, choice, option_items, by_value)
        entry: dict[str, Any] = {"value": choice}
        for field in ("name", "subtitle", "group"):
            if meta.get(field):
                entry[field] = str(meta[field])
        out.append(entry)
    return out


def _auto_choice_items(
    choices: list[str], keys: list[str], refs: dict[str, _CharacterRef]
) -> list[dict[str, Any]] | None:
    """Return `choice_items` when the options are mostly registered (draftable) characters with avatars."""
    non_reroll = [c for c in choices if not _REROLL_RE.search(c)]
    if not non_reroll:
        return None
    matches: dict[str, dict[str, str]] = {}
    for choice in non_reroll:
        matched = match_character_option(choice, keys, refs, draftable_only=True)
        if matched:
            matches[choice] = matched
    with_avatar = sum(1 for m in matches.values() if m.get("avatar"))
    needed = 1 if len(non_reroll) == 1 else 2
    if with_avatar < needed or with_avatar * 2 < len(non_reroll):
        return None
    return [{"value": c, **matches[c]} if c in matches else {"value": c, "name": c} for c in choices]


def enrich_question_segment(item: dict[str, Any], seg: dict[str, Any], workspace_dir: str | None) -> None:
    """Add `layout` / `choice_items` to a wire question `item` built from internal segment `seg` (in place).

    Cards are emitted for explicit `layout: cards` / explicit option avatars, or when the plain options are
    auto-detected as a character picker (see `_auto_choice_items`). An explicit `layout: list` disables
    auto-detection. When structured `option_items` carry rich metadata (`subtitle`, `group`, or a distinct
    `name`) without `layout: cards` / avatars, `choice_items` is still populated without forcing `cards`.
    """
    choices = [str(c) for c in item.get("choices") or []]
    if not choices:
        return
    layout = seg.get("layout")
    option_items = [i for i in seg.get("option_items") or [] if isinstance(i, dict)]
    explicit_cards = layout == "cards" or any(i.get("avatar") for i in option_items)
    has_rich_meta = any(
        i.get("subtitle") or i.get("group") or (i.get("name") and i.get("name") != i.get("value"))
        for i in option_items
    )
    if layout == "list" and not explicit_cards:
        item["layout"] = "list"
        if has_rich_meta:
            item["choice_items"] = _metadata_choice_items(choices, option_items)
        return
    try:
        keys, refs = load_character_registry(workspace_dir)
    except Exception as e:  # registry problems must never break message delivery
        logger.warning(f"Character registry unavailable for question enrichment: {e}")
        keys, refs = [], {}
    if explicit_cards:
        item["layout"] = "cards"
        item["choice_items"] = _explicit_choice_items(choices, option_items, keys, refs)
        return
    if refs:
        auto = _auto_choice_items(choices, keys, refs)
        if auto is not None:
            if option_items:
                by_value = {str(i.get("value")): i for i in option_items if i.get("value")}
                for idx, entry in enumerate(auto):
                    meta = _meta_for_choice(idx, entry["value"], option_items, by_value)
                    if meta.get("group"):
                        entry["group"] = str(meta["group"])
            item["layout"] = "cards"
            item["choice_items"] = auto
            return
    if has_rich_meta:
        item["choice_items"] = _metadata_choice_items(choices, option_items)


def to_wire_segments(
    segments: list[dict[str, Any]],
    *,
    message_id: str,
    live: bool,
    workspace_dir: str | None = None,
) -> tuple[list[dict[str, Any]], list[MediaJob]]:
    """Convert internal `parse_message_content(..., rich_blocks=True)` segments to wire segments.

    Args:
        segments: Internal segments.
        message_id: Source assistant message id (used for questions).
        live: When True, every media reference gets a `placeholder_id` and a `MediaJob` is returned for it.
        workspace_dir: Agent working directory holding the `characters/` registry used to enrich question
            options with avatar cards (defaults to the current working directory).

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
            question: dict[str, Any] = {
                "type": "question",
                "prompt": str(seg.get("question") or ""),
                "choices": list(seg.get("choices") or []),
                "multi": bool(seg.get("multi", False)),
                "max": int(seg.get("max", 1) or 1),
                "question_message_id": message_id,
            }
            enrich_question_segment(question, seg, workspace_dir)
            wire.append(question)
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
    # Strip the injected `Attachments:\n[Attachment: ... saved at ...]` listing whenever it is present (also for
    # messages ingested by other channels, whose metadata may not carry `attachments`/`display_text`).
    return _ATTACHMENTS_SUFFIX_RE.sub("", msg.content or "")


def user_attachments(msg: Message) -> list[dict[str, Any]]:
    """Return wire attachments (`{ref, mime, filename, object, bytes}`) of a stored user message.

    Prefers structured `metadata["attachments"]`; falls back to parsing the injected listing in the content so
    history of older / cross-channel messages still renders voice and image pills instead of server paths.
    """
    meta = msg.metadata or {}
    out: list[dict[str, Any]] = []
    for a in meta.get("attachments") or []:
        if isinstance(a, dict) and a.get("path"):
            path = str(a.get("path"))
            mime = a.get("mime_type") or a.get("mime") or _mime_for(path) or "application/octet-stream"
            out.append(
                {
                    "ref": path,
                    "mime": mime,
                    "filename": a.get("filename") or os.path.basename(path),
                    "object": a.get("object") or "",
                    "bytes": int(a.get("bytes") or 0),
                }
            )
    if out:
        return out
    suffix = _ATTACHMENTS_SUFFIX_RE.search(msg.content or "")
    if not suffix:
        return out
    for m in _ATTACHMENT_LINE_RE.finditer(suffix.group(0)):
        path = m.group("path").strip()
        out.append(
            {
                "ref": path,
                "mime": m.group("mime") or _mime_for(path) or "application/octet-stream",
                "filename": m.group("filename").strip() or os.path.basename(path),
                "object": "",
                "bytes": 0,
            }
        )
    return out


def message_to_record(
    msg: Message,
    app_chatbot_id: str,
    *,
    role_id: str | None = None,
    role_display_name: str | None = None,
    workspace_dir: str | None = None,
) -> dict[str, Any]:
    """Convert a stored user / assistant text message to a `MessageRecord` (media as refs only).

    `role_id` / `role_display_name` (the session persona) are attached to assistant senders, except
    notifications, so clients can show the persona name instead of the bot name. `workspace_dir` locates the
    character registry used to enrich question options (see `to_wire_segments`).
    """
    meta = msg.metadata or {}
    if msg.role == MessageRole.USER:
        attachments = user_attachments(msg)
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
    segments, _ = to_wire_segments(
        parse_message_content(msg.content or "", rich_blocks=True),
        message_id=msg.id,
        live=False,
        workspace_dir=workspace_dir,
    )
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

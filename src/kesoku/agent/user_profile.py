"""User profile persistence and its managed block inside every role's pinned `user/profile.md` memory.

`$AWD/user/profile.json` stores `{nickname, bio, cameo_opt_in, character_dir, avatar_rev, avatar_ai}`.
Nickname and bio reach the system prompt through the existing PINNED memory channel: each role's
`roles/<role>/memories/user/profile.md` carries a managed block::

    <!-- kesoku:user-profile -->
    - 称呼：...
    - 简介：...
    <!-- /kesoku:user-profile -->

Text outside the block is never modified. Writes use the memory writer primitives
(`with_memory_lock` + `atomic_write_text`), then `sync_role_memories` and `default_pinned_cache.invalidate`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterable
from typing import Any

from kesoku.agent.memory_store import default_pinned_cache, sync_role_memories
from kesoku.agent.memory_writer import atomic_write_text, with_memory_lock
from kesoku.db.manager import DatabaseManager

logger = logging.getLogger(__name__)

BLOCK_START = "<!-- kesoku:user-profile -->"
BLOCK_END = "<!-- /kesoku:user-profile -->"
PROFILE_REL_PATH = "user/profile.md"
DEFAULT_CHARACTER_DIR = "user"
CHARACTER_DIR_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$")
NICKNAME_MAX = 32
BIO_MAX_BYTES = 2048
_BLOCK_RE = re.compile(re.escape(BLOCK_START) + r".*?" + re.escape(BLOCK_END) + r"\n?", re.DOTALL)
_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_NEW_FILE_FRONTMATTER = '---\ndescription: "用户核心档案（称呼、简介与长期偏好）"\nvisibility: PINNED\n---\n'

DEFAULT_PROFILE: dict[str, Any] = {
    "nickname": "",
    "bio": "",
    "cameo_opt_in": False,
    "character_dir": DEFAULT_CHARACTER_DIR,
    "avatar_rev": 0,
    "avatar_ai": False,
}


def _clean_inline(text: str) -> str:
    """Collapse whitespace/control characters and neutralize HTML comment markers."""
    cleaned = _CTRL_RE.sub(" ", text or "").replace("<!--", "<!-").replace("-->", "->")
    return " ".join(cleaned.split())


def render_profile_block(nickname: str, bio: str) -> str:
    """Render the managed block (empty string when both fields are blank)."""
    nick = _clean_inline(nickname)
    bio_lines = [_clean_inline(ln) for ln in (bio or "").splitlines()]
    bio_lines = [ln for ln in bio_lines if ln]
    if not nick and not bio_lines:
        return ""
    lines = [BLOCK_START]
    if nick:
        lines.append(f"- 称呼：{nick}")
    if bio_lines:
        lines.append(f"- 简介：{bio_lines[0]}")
        lines.extend(f"  {ln}" for ln in bio_lines[1:])
    lines.append(BLOCK_END)
    return "\n".join(lines) + "\n"


def upsert_managed_block(text: str, block: str) -> str:
    """Replace (or insert, or remove when `block` is empty) the managed block in `text`.

    A new block goes right after the YAML frontmatter, before any `##` section, so the memory writer keeps
    it in the document preamble. A file without frontmatter gets the block prepended.
    """
    original = text or ""
    if _BLOCK_RE.search(original):
        return _BLOCK_RE.sub(lambda _m: block, original, count=1)
    if not block:
        return original
    if not original.strip():
        return _NEW_FILE_FRONTMATTER + "\n" + block
    fm = _FRONTMATTER_RE.match(original)
    if fm:
        head = original[: fm.end()]
        if not head.endswith("\n"):
            head += "\n"
        rest = original[fm.end() :].lstrip("\n")
        return head + "\n" + block + ("\n" + rest if rest else "")
    return block + "\n" + original


def sync_profile_block_to_role(
    db: DatabaseManager | None,
    role: str,
    memories_dir: str,
    nickname: str,
    bio: str,
) -> bool:
    """Upsert the managed block into `<memories_dir>/user/profile.md`; returns True when the file changed."""
    block = render_profile_block(nickname, bio)
    target = os.path.join(memories_dir, PROFILE_REL_PATH)
    changed = False
    with with_memory_lock(memories_dir):
        try:
            with open(target, encoding="utf-8") as f:
                current = f.read()
        except FileNotFoundError:
            current = ""
        if not current and not block:
            return False
        new_text = upsert_managed_block(current, block)
        if new_text != current:
            atomic_write_text(target, new_text)
            changed = True
    if changed:
        if db is not None:
            try:
                sync_role_memories(db, role, memories_dir)
            except Exception as e:  # noqa: BLE001 - indexing failure must not lose the profile write
                logger.warning("sync_role_memories failed for role %s: %s", role, e)
        default_pinned_cache.invalidate(role)
    return changed


def sync_profile_to_roles(
    db: DatabaseManager | None,
    roles_dir: str,
    role_ids: Iterable[str],
    nickname: str,
    bio: str,
) -> list[str]:
    """Sync the managed block into every role's memories; returns the roles whose file changed."""
    changed: list[str] = []
    real_root = os.path.realpath(roles_dir)
    for role in role_ids:
        mem_dir = os.path.join(real_root, role, "memories")
        real_role = os.path.realpath(os.path.join(real_root, role))
        if not real_role.startswith(real_root.rstrip(os.sep) + os.sep):
            continue
        try:
            if sync_profile_block_to_role(db, role, mem_dir, nickname, bio):
                changed.append(role)
        except OSError as e:
            logger.warning("Failed syncing user profile block into role %s: %s", role, e)
    return changed


class UserProfileStore:
    """Reads and atomically writes `$AWD/user/profile.json`."""

    def __init__(self, awd: str) -> None:
        """Bind the store to the agent working directory."""
        self.user_dir = os.path.join(os.path.abspath(awd), "user")
        self.path = os.path.join(self.user_dir, "profile.json")
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        """Return the profile merged over defaults (invalid fields fall back to defaults)."""
        data: dict[str, Any] = {}
        try:
            with open(self.path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            pass
        out = dict(DEFAULT_PROFILE)
        out.update({k: v for k, v in data.items() if k not in DEFAULT_PROFILE})
        if isinstance(data.get("nickname"), str):
            out["nickname"] = data["nickname"]
        if isinstance(data.get("bio"), str):
            out["bio"] = data["bio"]
        out["cameo_opt_in"] = bool(data.get("cameo_opt_in", False))
        cdir = data.get("character_dir")
        out["character_dir"] = cdir if isinstance(cdir, str) and CHARACTER_DIR_RE.fullmatch(cdir) else (
            DEFAULT_CHARACTER_DIR
        )
        try:
            out["avatar_rev"] = max(0, int(data.get("avatar_rev") or 0))
        except (TypeError, ValueError):
            out["avatar_rev"] = 0
        out["avatar_ai"] = bool(data.get("avatar_ai", False))
        return out

    def save(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Atomically persist `profile` and return it."""
        with self._lock:
            os.makedirs(self.user_dir, exist_ok=True)
            atomic_write_text(self.path, json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
            return dict(profile)

    def update(self, **changes: Any) -> dict[str, Any]:
        """Read-modify-write the profile under the store lock."""
        with self._lock:
            profile = self.load()
            profile.update(changes)
            return self.save(profile)

    def avatar_path(self) -> str:
        """Absolute path of the user avatar."""
        return os.path.join(self.user_dir, "avatar.webp")

    def public_view(self, profile: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the `UserProfile` wire shape (adds `avatar_ref`)."""
        p = profile if profile is not None else self.load()
        has_avatar = os.path.isfile(self.avatar_path())
        return {
            "nickname": p.get("nickname", ""),
            "bio": p.get("bio", ""),
            "cameo_opt_in": bool(p.get("cameo_opt_in")),
            "avatar_rev": int(p.get("avatar_rev") or 0),
            "avatar_ref": "user/avatar.webp" if has_avatar else None,
            "avatar_ai": bool(p.get("avatar_ai")) if has_avatar else False,
            "character_dir": p.get("character_dir", DEFAULT_CHARACTER_DIR),
        }

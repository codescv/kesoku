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

Cross-channel compatibility: when `$AWD/instructions/User.md` carries a `# USER PROFILE` section (a
`user_prompts` file injected into every channel), the profile is written there instead and the role blocks are
removed, so each fact reaches the prompt at most once. Bio lines that already exist as role memory bullets stay
owned by the memory files (the memory writer keeps updating them in place) and are not copied anywhere.
`UserProfileStore.import_from_workspace` reads those existing sources back so users never re-type their info.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time
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

USER_MD_REL_PATH = os.path.join("instructions", "User.md")
USER_MD_SECTION_RE = re.compile(r"^#[ \t]+USER PROFILE[ \t]*$", re.IGNORECASE | re.MULTILINE)
_TOP_HEADING_RE = re.compile(r"^#[ \t]+\S", re.MULTILINE)
_BULLET_RE = re.compile(r"^[ \t]*(?:[-*•+]|\d+[.)])[ \t]+")
_TAG_TAIL_RE = re.compile(r"(?:[ \t]+#[\w:./-]+)+[ \t]*$")
_NICK_LINE_RE = re.compile(r"^[-*•][ \t]*(?:Nickname|Nick|称呼|昵称)[ \t]*[:：][ \t]*(.*)$", re.IGNORECASE)
_NICK_SPLIT_RE = re.compile(r"^([^\n(（]*?)[ \t]*([(（].*)?$")
_BIO_HEAD_RE = re.compile(r"^-[ \t]*简介[ \t]*[:：][ \t]*(.*)$")
AVATAR_IMPORT_SIZE = 512
AVATAR_IMPORT_MAX_BYTES = 20 * 1024 * 1024

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
        lines.append(f"- 简介：{_BULLET_RE.sub('', bio_lines[0])}")
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


def fact_key(line: str) -> str:
    """Normalized identity of one profile fact (bullet marker, trailing `#tags` and case are ignored)."""
    text = _TAG_TAIL_RE.sub("", _BULLET_RE.sub("", line or "")).strip()
    return " ".join(text.split()).casefold()


def _as_bullet(line: str) -> str:
    """Render a fact as a Markdown bullet (indented sub-bullets are kept as they are)."""
    stripped = line.rstrip()
    if _BULLET_RE.match(stripped):
        return stripped
    return f"- {stripped.strip()}"


def bio_lines(bio: str) -> list[str]:
    """Non-empty bio lines (right-stripped, original indentation kept)."""
    return [ln.rstrip() for ln in (bio or "").splitlines() if ln.strip()]


def _strip_frontmatter(text: str) -> str:
    m = _FRONTMATTER_RE.match(text or "")
    return (text or "")[m.end() :] if m else (text or "")


def parse_profile_block(text: str) -> tuple[str, str]:
    """Return `(nickname, bio)` stored in an existing managed block (empty strings when absent)."""
    m = _BLOCK_RE.search(text or "")
    if not m:
        return "", ""
    nick, bio = "", []
    in_bio = False
    for raw in m.group(0).splitlines()[1:-1]:
        line = raw.rstrip()
        if line.startswith("- 称呼："):
            nick, in_bio = line[len("- 称呼：") :].strip(), False
            continue
        head = _BIO_HEAD_RE.match(line)
        if head:
            in_bio = True
            if head.group(1).strip():
                bio.append(head.group(1).strip())
            continue
        if in_bio and line.strip() and line.startswith(" "):
            bio.append(line[2:] if line.startswith("  ") else line.strip())
    return nick, "\n".join(bio)


def memory_profile_facts(text: str) -> tuple[str, list[str]]:
    """Return `(nickname, facts)` from a role `profile.md` outside the managed block and `## 历史`.

    Facts are top-level bullets without their trailing `#direct #session:… #time:…` tags.
    """
    body = _BLOCK_RE.sub("", _strip_frontmatter(text))
    nickname = ""
    facts: list[str] = []
    skip = False
    for raw in body.splitlines():
        line = raw.rstrip()
        if line.startswith("#"):
            skip = "历史" in line
            continue
        if skip or not line.strip() or line[:1] in (" ", "\t") or not _BULLET_RE.match(line):
            continue
        content = _TAG_TAIL_RE.sub("", _BULLET_RE.sub("", line)).strip()
        if not content:
            continue
        nm = re.match(r"^(?:称呼|昵称)[ \t]*[:：][ \t]*(.+)$", content)
        if nm:
            nickname = nickname or nm.group(1).strip()
            continue
        facts.append(f"- {content}")
    return nickname, facts


def split_user_md(text: str) -> tuple[str, list[str], str] | None:
    """Split `User.md` into `(before, USER PROFILE body lines, after)`; None without that section."""
    m = USER_MD_SECTION_RE.search(text or "")
    if not m:
        return None
    body_start = m.end() + 1 if text[m.end() : m.end() + 1] == "\n" else m.end()
    nxt = _TOP_HEADING_RE.search(text, body_start)
    body_end = nxt.start() if nxt else len(text)
    return text[:body_start], text[body_start:body_end].splitlines(), text[body_end:]


def parse_user_md_section(body: list[str]) -> dict[str, Any]:
    """Parse `# USER PROFILE` body lines into nickname (+ preserved suffix/sub-lines) and other facts."""
    out: dict[str, Any] = {"nickname": "", "nickname_suffix": "", "nickname_children": [], "nickname_index": -1,
                           "facts": []}
    in_nick = False
    for raw in body:
        line = raw.rstrip()
        if not line.strip():
            in_nick = False
            continue
        indented = line[:1] in (" ", "\t")
        if in_nick and indented:
            out["nickname_children"].append(line)
            continue
        in_nick = False
        nm = _NICK_LINE_RE.match(line) if not indented else None
        if nm and out["nickname_index"] < 0:
            split = _NICK_SPLIT_RE.match(nm.group(1).strip())
            out["nickname"] = (split.group(1) if split else nm.group(1)).strip()
            out["nickname_suffix"] = ((split.group(2) if split else "") or "").strip()
            out["nickname_index"] = len(out["facts"])
            in_nick = True
            continue
        out["facts"].append(line if indented or _BULLET_RE.match(line) else f"- {line.strip()}")
    return out


def render_user_md_section(parsed: dict[str, Any], nickname: str, facts: list[str]) -> list[str]:
    """Render the `# USER PROFILE` body: the (suffix-preserving) nickname line placed among `facts`."""
    lines = [_as_bullet(f) for f in facts]
    nick = _clean_inline(nickname)
    if nick:
        suffix = parsed.get("nickname_suffix") or ""
        nick_lines = [f"- Nickname: {nick}" + (f" {suffix}" if suffix else "")]
        nick_lines += list(parsed.get("nickname_children") or [])
        idx = parsed.get("nickname_index", -1)
        idx = len(lines) if idx is None or idx < 0 else min(idx, len(lines))
        lines[idx:idx] = nick_lines
    return lines


def default_user_md_path(roles_dir: str) -> str:
    """`<parent of roles_dir>/instructions/User.md`."""
    return os.path.join(os.path.dirname(os.path.realpath(roles_dir)), USER_MD_REL_PATH)


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, NotADirectoryError):
        return ""


def sync_profile_to_user_md(path: str, nickname: str, facts: list[str]) -> bool:
    """Rewrite only the `# USER PROFILE` section of `User.md`; returns True when the file changed.

    Every other `# ` section is preserved byte-for-byte, as is the nickname line's parenthetical note and
    its indented sub-bullets.
    """
    current = _read_text(path)
    split = split_user_md(current)
    if split is None:
        return False
    before, body, after = split
    parsed = parse_user_md_section(body)
    new_body = render_user_md_section(parsed, nickname, facts)
    trailing = "\n" if after else ""
    new_text = before + "\n".join(new_body) + ("\n" if new_body else "") + trailing + after
    if new_text == current:
        return False
    atomic_write_text(path, new_text)
    return True


def sync_profile_block_to_role(
    db: DatabaseManager | None,
    role: str,
    memories_dir: str,
    nickname: str,
    bio: str,
    *,
    dedupe_existing: bool = True,
) -> bool:
    """Upsert the managed block into `<memories_dir>/user/profile.md`; returns True when the file changed.

    With `dedupe_existing`, bio lines already present as bullets outside the block are left out of the
    block, so each fact appears at most once in the file.
    """
    target = os.path.join(memories_dir, PROFILE_REL_PATH)
    changed = False
    with with_memory_lock(memories_dir):
        try:
            with open(target, encoding="utf-8") as f:
                current = f.read()
        except FileNotFoundError:
            current = ""
        lines = bio_lines(bio)
        if dedupe_existing and current:
            existing = {fact_key(f) for f in memory_profile_facts(current)[1]}
            lines = [ln for ln in lines if fact_key(ln) not in existing]
        block = render_profile_block(nickname, "\n".join(lines))
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


def _role_memory_dirs(roles_dir: str, role_ids: Iterable[str]) -> list[tuple[str, str]]:
    real_root = os.path.realpath(roles_dir)
    out: list[tuple[str, str]] = []
    for role in role_ids:
        real_role = os.path.realpath(os.path.join(real_root, role))
        if not real_role.startswith(real_root.rstrip(os.sep) + os.sep):
            continue
        out.append((role, os.path.join(real_root, role, "memories")))
    return out


def sync_profile_to_roles(
    db: DatabaseManager | None,
    roles_dir: str,
    role_ids: Iterable[str],
    nickname: str,
    bio: str,
    *,
    instructions_user_md: str | None = None,
) -> list[str]:
    """Sync the profile to every channel's prompt sources without duplicating any fact.

    `instructions_user_md` defaults to `<parent of roles_dir>/instructions/User.md` (pass "" to disable).
    When that file has a `# USER PROFILE` section, the profile is written there (bio lines that already exist
    as role memory bullets are left to the memory files; a fully blank profile leaves the section untouched)
    and the role blocks are removed. Otherwise the managed
    block is written into each role's `user/profile.md`, minus bullets already present in that file.
    Returns the roles whose memory file changed.
    """
    roles = _role_memory_dirs(roles_dir, role_ids)
    user_md = default_user_md_path(roles_dir) if instructions_user_md is None else instructions_user_md
    has_section = bool(user_md) and split_user_md(_read_text(user_md)) is not None
    changed: list[str] = []
    if has_section:
        memory_keys: set[str] = set()
        for _role, mem_dir in roles:
            memory_keys.update(
                fact_key(f) for f in memory_profile_facts(_read_text(os.path.join(mem_dir, PROFILE_REL_PATH)))[1]
            )
        facts = [ln for ln in bio_lines(bio) if fact_key(ln) not in memory_keys]
        try:
            # A fully blank profile never wipes the hand-written section (it stays re-importable).
            if _clean_inline(nickname) or facts:
                sync_profile_to_user_md(user_md, nickname, facts)
        except OSError as e:
            logger.warning("Failed syncing user profile into %s: %s", user_md, e)
    for role, mem_dir in roles:
        try:
            if has_section:
                did = sync_profile_block_to_role(db, role, mem_dir, "", "")
            else:
                did = sync_profile_block_to_role(db, role, mem_dir, nickname, bio)
            if did:
                changed.append(role)
        except OSError as e:
            logger.warning("Failed syncing user profile block into role %s: %s", role, e)
    return changed


def _fit_bio(lines: list[str], limit: int = BIO_MAX_BYTES) -> str:
    """Join deduplicated lines, dropping whole lines that would exceed `limit` UTF-8 bytes."""
    out: list[str] = []
    seen: set[str] = set()
    size = 0
    for ln in lines:
        key = fact_key(ln)
        if not key or key in seen:
            continue
        cost = len(ln.encode("utf-8")) + (1 if out else 0)
        if size + cost > limit:
            continue
        seen.add(key)
        out.append(ln)
        size += cost
    return "\n".join(out)


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

    def exists(self) -> bool:
        """True when `profile.json` has been written."""
        return os.path.isfile(self.path)

    def get(self) -> dict[str, Any]:
        """Profile plus `avatar` (absolute path or None)."""
        out = self.load()
        out["avatar"] = self.avatar_path() if os.path.isfile(self.avatar_path()) else None
        return out

    def needs_import(self) -> bool:
        """True when no profile was saved yet, or it is still blank and was never imported."""
        if not self.exists():
            return True
        p = self.load()
        return not p["nickname"] and not p["bio"].strip() and not p.get("imported_at")

    def import_from_workspace(
        self,
        awd_dir: str | None = None,
        roles_dir: str | None = None,
        *,
        characters_dir: str | None = None,
    ) -> dict[str, Any]:
        """Collect the user's existing info from workspace files (does not write `profile.json`).

        Sources (first non-empty nickname wins; bio lines are deduplicated and capped at 2 KiB):
        `<awd>/instructions/User.md` `# USER PROFILE`, then each `<roles_dir>/*/memories/user/profile.md`
        (bullets outside `## 历史`, plus a legacy managed block). When no avatar exists yet, the
        `characters/*/characters.json` entry whose name/aliases match the nickname provides it (written to
        `user/avatar.webp`). Returns `{nickname, bio, sources, avatar_imported, character_dir?}`.
        """
        awd = os.path.abspath(awd_dir) if awd_dir else os.path.dirname(self.user_dir)
        roles = os.path.abspath(roles_dir) if roles_dir else os.path.join(awd, "roles")
        nickname = ""
        md_facts: list[str] = []
        mem_facts: list[str] = []
        sources: list[str] = []
        split = split_user_md(_read_text(os.path.join(awd, USER_MD_REL_PATH)))
        if split is not None:
            parsed = parse_user_md_section(split[1])
            nickname = parsed["nickname"]
            md_facts = list(parsed["facts"])
            if nickname or md_facts:
                sources.append(USER_MD_REL_PATH.replace(os.sep, "/"))
        block_bio: list[str] = []
        try:
            role_names = sorted(os.listdir(roles))
        except OSError:
            role_names = []
        for name in role_names:
            path = os.path.join(roles, name, "memories", PROFILE_REL_PATH)
            text = _read_text(path) if os.path.isfile(path) else ""
            if not text:
                continue
            mem_nick, facts = memory_profile_facts(text)
            blk_nick, blk_bio = parse_profile_block(text)
            nickname = nickname or mem_nick or blk_nick
            mem_facts.extend(facts)
            block_bio.extend(bio_lines(blk_bio))
            if facts or mem_nick or blk_nick or blk_bio:
                sources.append(os.path.relpath(path, awd).replace(os.sep, "/"))
        result: dict[str, Any] = {
            "nickname": _clean_inline(nickname)[:NICKNAME_MAX],
            "bio": _fit_bio(md_facts + [_as_bullet(b) for b in block_bio] + mem_facts),
            "sources": sources,
            "avatar_imported": False,
        }
        cdir_root = os.path.abspath(characters_dir) if characters_dir else os.path.join(awd, "characters")
        match = self._find_character(cdir_root, result["nickname"]) if result["nickname"] else None
        if match is not None:
            dir_name, image = match
            result["character_dir"] = dir_name
            if image and not os.path.isfile(self.avatar_path()):
                result["avatar_imported"] = self._import_avatar(image)
        return result

    @staticmethod
    def _find_character(characters_root: str, nickname: str) -> tuple[str, str | None] | None:
        """Find `(dir_name, image_path|None)` of the registry entry named/aliased `nickname`."""
        wanted = nickname.casefold()
        base = os.path.dirname(os.path.realpath(characters_root))
        try:
            dirs = sorted(os.listdir(characters_root))
        except OSError:
            return None
        for dir_name in dirs:
            if not CHARACTER_DIR_RE.fullmatch(dir_name):
                continue
            try:
                with open(os.path.join(characters_root, dir_name, "characters.json"), encoding="utf-8") as f:
                    entries = json.load(f)
            except (OSError, ValueError):
                continue
            for e in entries if isinstance(entries, list) else []:
                if not isinstance(e, dict):
                    continue
                names = [e.get("name")] + list(e.get("aliases") or [])
                if wanted not in {str(n).casefold() for n in names if isinstance(n, str) and n}:
                    continue
                image = e.get("image")
                path = None
                if isinstance(image, str) and image and not os.path.isabs(image):
                    real = os.path.realpath(os.path.join(base, image))
                    if real.startswith(base.rstrip(os.sep) + os.sep) and os.path.isfile(real):
                        path = real
                return dir_name, path
        return None

    def _import_avatar(self, image_path: str) -> bool:
        """Convert an existing character image into the 512×512 `user/avatar.webp` (False on failure)."""
        try:
            from PIL import Image, ImageOps

            if os.path.getsize(image_path) > AVATAR_IMPORT_MAX_BYTES:
                return False
            with Image.open(image_path) as src:
                im = ImageOps.exif_transpose(src).convert("RGB")
            im = ImageOps.fit(im, (AVATAR_IMPORT_SIZE, AVATAR_IMPORT_SIZE), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "WEBP", quality=90)
        except Exception as e:  # noqa: BLE001 - an unreadable image just means no imported avatar
            logger.warning("Cannot import user avatar from %s: %s", image_path, e)
            return False
        os.makedirs(self.user_dir, exist_ok=True)
        tmp = self.avatar_path() + ".tmp"
        with open(tmp, "wb") as f:
            f.write(buf.getvalue())
        os.replace(tmp, self.avatar_path())
        return True

    def apply_import(self, imported: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
        """Merge an `import_from_workspace` result into the stored profile and persist it.

        Empty fields are filled; with `overwrite` the bio is merged (existing lines first, deduplicated).
        """
        with self._lock:
            profile = self.load()
            if imported.get("nickname") and (overwrite or not profile["nickname"]):
                profile["nickname"] = imported["nickname"]
            new_bio = str(imported.get("bio") or "")
            if new_bio:
                if not profile["bio"].strip():
                    profile["bio"] = new_bio
                elif overwrite:
                    profile["bio"] = _fit_bio(bio_lines(profile["bio"]) + bio_lines(new_bio))
            cdir = imported.get("character_dir")
            if cdir and profile.get("character_dir") in (None, DEFAULT_CHARACTER_DIR):
                if isinstance(cdir, str) and CHARACTER_DIR_RE.fullmatch(cdir):
                    profile["character_dir"] = cdir
                    profile["cameo_opt_in"] = True
            if imported.get("avatar_imported"):
                profile["avatar_rev"] = int(profile.get("avatar_rev") or 0) + 1
                profile["avatar_ai"] = False
            profile["imported_at"] = time.time()
            profile["imported_from"] = list(imported.get("sources") or [])
            return self.save(profile)

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

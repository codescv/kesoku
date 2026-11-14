"""Role (persona) registry backed by the `roles_dir` directory tree.

Each role lives in `<roles_dir>/<role_id>/` with `intro.md`, `preferences.md`, `memories/`, optional
`avatar.webp` and an optional `role.json` metadata file::

    {"role_id", "display_name", "tagline", "builtin", "voice_preset", "avatar_rev", "updated_at"}

Roles that pre-exist without `role.json` are synthesized from the directory name plus the first meaningful
line of `intro.md`, marked `builtin: true` (they shipped with the working directory, not created by a client),
and the synthesized metadata is written back lazily. The registry never hard-codes any role names.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import unicodedata
from typing import Any

from kesoku.agent.memory_writer import atomic_write_text

logger = logging.getLogger(__name__)

ROLE_ID_RE = re.compile(r"^[a-z0-9_]{2,32}$")
ROLE_JSON = "role.json"
INTRO_MD = "intro.md"
PREFERENCES_MD = "preferences.md"
AVATAR_FILE = "avatar.webp"
DISPLAY_NAME_MAX = 32
TAGLINE_MAX = 80
_GENERIC_NAME_HEADINGS = frozenset({"name", "名字", "名称", "姓名", "名前", "角色", "role"})
_ROLE_MARKER_FILES = (ROLE_JSON, INTRO_MD, PREFERENCES_MD)
_SUMMARY_KEYS = ("role_id", "display_name", "tagline", "builtin", "voice_preset", "avatar_rev", "updated_at")


def is_valid_role_id(role_id: Any) -> bool:
    """Return True when `role_id` matches `^[a-z0-9_]{2,32}$`."""
    return isinstance(role_id, str) and bool(ROLE_ID_RE.fullmatch(role_id))


def slugify_role_id(display_name: str, fallback: str = "role") -> str:
    """Return a `^[a-z0-9_]{2,32}$` slug for `display_name` (ASCII-folded; `fallback` when nothing remains)."""
    folded = unicodedata.normalize("NFKD", display_name or "").encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", folded.lower()).strip("_")[:32].strip("_")
    if len(slug) < 2:
        slug = fallback
    return slug


def _first_display_line(intro_text: str) -> str:
    """Extract a display name from intro.md: first meaningful line, skipping generic `# Name` headings."""
    lines = [ln.strip() for ln in intro_text.splitlines()]
    lines = [ln for ln in lines if ln and ln != "---"]
    for i, line in enumerate(lines):
        is_heading = line.startswith("#")
        text = line.lstrip("#").strip().strip("*_`").strip()
        if not text:
            continue
        if is_heading and text.rstrip(":：").strip().lower() in _GENERIC_NAME_HEADINGS:
            for nxt in lines[i + 1 :]:
                if nxt.startswith("#"):
                    break
                value = nxt.lstrip("-* ").strip().strip("*_`").strip()
                if value:
                    return value
            continue
        return text
    return ""


def _clip(text: Any, limit: int) -> str:
    """Return `text` as a single-line string clipped to `limit` characters."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())[:limit]


class RoleRegistry:
    """Lists, reads and updates role metadata under `roles_dir` (thread-safe; call from worker threads)."""

    def __init__(self, roles_dir: str) -> None:
        """Bind the registry to an absolute (or relative) roles directory."""
        self.roles_dir = os.path.abspath(roles_dir)
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    # ------------------------------------------------------------------ paths
    def role_dir(self, role_id: str) -> str | None:
        """Return the absolute role directory when `role_id` is valid and stays inside `roles_dir`."""
        if not is_valid_role_id(role_id):
            return None
        candidate = os.path.join(self.roles_dir, role_id)
        real_root = os.path.realpath(self.roles_dir)
        real_cand = os.path.realpath(candidate)
        if not real_cand.startswith(real_root.rstrip(os.sep) + os.sep):
            return None
        return real_cand

    def exists(self, role_id: str) -> bool:
        """Return True when `role_id` names an existing role directory."""
        d = self.role_dir(role_id)
        return bool(d) and os.path.isdir(d) and any(os.path.exists(os.path.join(d, f)) for f in _ROLE_MARKER_FILES)

    def list_role_ids(self) -> list[str]:
        """Return sorted role ids of every role directory (dot-dirs and invalid names skipped)."""
        try:
            names = sorted(os.listdir(self.roles_dir))
        except OSError:
            return []
        return [n for n in names if is_valid_role_id(n) and self.exists(n)]

    # ------------------------------------------------------------------ metadata
    def reload(self) -> list[dict[str, Any]]:
        """Drop cached metadata and re-read every role; returns the fresh summaries."""
        with self._lock:
            self._cache.clear()
        return self.list_roles()

    def list_roles(self) -> list[dict[str, Any]]:
        """Return `PersonaSummary` dicts for every role, sorted by builtin first then role id."""
        out = [s for s in (self.get(rid) for rid in self.list_role_ids()) if s]
        out.sort(key=lambda s: (not s.get("builtin"), s["role_id"]))
        return out

    def get(self, role_id: str) -> dict[str, Any] | None:
        """Return the `PersonaSummary` for `role_id` (synthesizing `role.json` lazily), or None."""
        meta = self.load_meta(role_id)
        if meta is None:
            return None
        return self.summary_from_meta(role_id, meta)

    def summary_from_meta(self, role_id: str, meta: dict[str, Any]) -> dict[str, Any]:
        """Project raw role metadata to a `PersonaSummary` dict (adds `avatar_ref`)."""
        d = self.role_dir(role_id) or ""
        summary: dict[str, Any] = {k: meta.get(k) for k in _SUMMARY_KEYS}
        summary["role_id"] = role_id
        summary["display_name"] = _clip(meta.get("display_name"), DISPLAY_NAME_MAX) or role_id
        summary["tagline"] = _clip(meta.get("tagline"), TAGLINE_MAX)
        summary["builtin"] = bool(meta.get("builtin"))
        summary["avatar_rev"] = max(0, int(meta.get("avatar_rev") or 0))
        vp = meta.get("voice_preset")
        summary["voice_preset"] = vp if isinstance(vp, str) and vp else None
        summary["updated_at"] = float(meta.get("updated_at") or 0.0)
        has_avatar = bool(d) and os.path.isfile(os.path.join(d, AVATAR_FILE))
        summary["avatar_ref"] = f"roles/{role_id}/{AVATAR_FILE}" if has_avatar else None
        summary["avatar_ai"] = bool(meta.get("avatar_ai")) if has_avatar else False
        return summary

    def load_meta(self, role_id: str) -> dict[str, Any] | None:
        """Return the raw `role.json` dict for `role_id`, synthesizing and persisting it when absent."""
        d = self.role_dir(role_id)
        if not d or not self.exists(role_id):
            return None
        path = os.path.join(d, ROLE_JSON)
        with self._lock:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = -1.0
            cached = self._cache.get(role_id)
            if cached and cached[0] == mtime and mtime >= 0:
                return dict(cached[1])
            meta: dict[str, Any] | None = None
            if mtime >= 0:
                try:
                    with open(path, encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        meta = loaded
                except (OSError, ValueError):
                    logger.warning("Unreadable %s for role %s; re-synthesizing", ROLE_JSON, role_id)
            if meta is None:
                meta = self._synthesize(role_id, d)
                try:
                    self._write_meta_locked(role_id, d, meta)
                    mtime = os.path.getmtime(path)
                except OSError as e:
                    logger.warning("Could not persist synthesized %s for %s: %s", ROLE_JSON, role_id, e)
                    mtime = -1.0
            meta["role_id"] = role_id
            if mtime >= 0:
                self._cache[role_id] = (mtime, dict(meta))
            return dict(meta)

    def _synthesize(self, role_id: str, d: str) -> dict[str, Any]:
        """Build default metadata for a pre-existing role directory without `role.json`."""
        display = ""
        try:
            with open(os.path.join(d, INTRO_MD), encoding="utf-8", errors="replace") as f:
                display = _first_display_line(f.read(16384))
        except OSError:
            pass
        try:
            updated = os.path.getmtime(d)
        except OSError:
            updated = time.time()
        return {
            "role_id": role_id,
            "display_name": _clip(display, DISPLAY_NAME_MAX) or role_id,
            "tagline": "",
            "builtin": True,
            "voice_preset": None,
            "avatar_rev": 0,
            "updated_at": updated,
        }

    def write_meta(self, role_id: str, meta: dict[str, Any]) -> dict[str, Any]:
        """Atomically persist `role.json` for an existing (or just created) role directory."""
        d = self.role_dir(role_id)
        if not d:
            raise ValueError(f"invalid role id: {role_id!r}")
        with self._lock:
            data = dict(meta)
            data["role_id"] = role_id
            self._write_meta_locked(role_id, d, data)
            return data

    def update_meta(self, role_id: str, **changes: Any) -> dict[str, Any]:
        """Read-modify-write `role.json` under the registry lock (sets `updated_at`)."""
        with self._lock:
            meta = self.load_meta(role_id)
            if meta is None:
                raise KeyError(role_id)
            meta.update(changes)
            meta["updated_at"] = time.time()
            return self.write_meta(role_id, meta)

    def bump_avatar_rev(self, role_id: str, *, avatar_ai: bool) -> dict[str, Any]:
        """Increment `avatar_rev` (and record whether the avatar is AI-generated)."""
        with self._lock:
            meta = self.load_meta(role_id)
            if meta is None:
                raise KeyError(role_id)
            return self.update_meta(role_id, avatar_rev=int(meta.get("avatar_rev") or 0) + 1, avatar_ai=avatar_ai)

    def _write_meta_locked(self, role_id: str, d: str, meta: dict[str, Any]) -> None:
        """Write `role.json` atomically and drop the cache entry."""
        os.makedirs(d, exist_ok=True)
        atomic_write_text(os.path.join(d, ROLE_JSON), json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        self._cache.pop(role_id, None)

    def forget(self, role_id: str) -> None:
        """Drop cached metadata for `role_id` (after delete/rename)."""
        with self._lock:
            self._cache.pop(role_id, None)

    def unique_role_id(self, display_name: str, *, reserved: set[str] | None = None) -> str:
        """Slugify `display_name` and append `_2`, `_3`, ... until no role directory uses the id."""
        base = slugify_role_id(display_name)
        taken = set(reserved or ())
        candidate = base
        n = 2
        while candidate in taken or os.path.exists(os.path.join(self.roles_dir, candidate)):
            suffix = f"_{n}"
            candidate = base[: 32 - len(suffix)].rstrip("_") + suffix
            n += 1
        return candidate

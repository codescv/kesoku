"""Persona workshop, user profile and AI avatar restyle handlers for the app channel (§3.3 / §3.7).

`PersonaService(host)` serves `persona.*`, `voice.preview`, `user_profile.*`, `avatar.generate` and
`avatar.apply`. All filesystem access is jailed with realpath + trailing-separator checks:

* roles: `<roles_dir>/<role_id>/{role.json, intro.md, preferences.md, avatar.webp, memories/}`
* user: `$AWD/user/{profile.json, avatar.webp}`
* face references: `<characters_dir>/<dir>/{images/ref_face.webp, characters.json, .kesoku_prev_image}`
  (`characters_dir` defaults to `$AWD/characters`; `image` paths are relative to its parent)
* avatar jobs: `<sessions_dir>/avatar_jobs/<job_id>/{source.png, cand_<i>.png, cand_<i>.webp}`

`avatar.generate` may omit `source_object` when the target already has an image on disk (original reference
photo, registry image or current avatar); see `PersonaService._avatar_source_candidates` for the lookup order.

Avatar candidates are produced by an external generator (argv template `AppConfig.avatar_generator_command`,
else `<skills_dir>/ai-image/scripts/generate_image.py --async`) and tracked through the shared
`<target>.pending.json` / `<target>.error.json` deferred media sidecar protocol.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import re
import shutil
import time
import warnings
from typing import Any

from PIL import Image, ImageOps

from kesoku.agent.memory_store import default_pinned_cache, sync_role_memories
from kesoku.agent.role_registry import (
    AVATAR_FILE,
    INTRO_MD,
    PREFERENCES_MD,
    RoleRegistry,
    is_valid_role_id,
)
from kesoku.agent.user_profile import (
    BIO_MAX_BYTES,
    CHARACTER_DIR_RE,
    NICKNAME_MAX,
    UserProfileStore,
    sync_profile_block_to_role,
    sync_profile_to_roles,
)
from kesoku.gateway.chatbot.app.host import (
    AppHost,
    Reply,
    RequestContext,
    atomic_write_bytes,
    realpath_within,
    safe_resolve,
)
from kesoku.gateway.chatbot.app.protocol import (
    AVATAR_REF_RE,
    AVATAR_SOURCE_OBJECT_RE,
    LIMITS,
    ProtocolError,
    new_id,
    optional_str,
    require_str,
)
from kesoku.gateway.chatbot.deferred_media import (
    DEAD_WORKER_GRACE_SEC,
    is_deferred_target_ready,
    is_pid_alive,
    read_json_dict,
    sanitize_error_reason,
    terminate_deferred_worker,
)

logger = logging.getLogger(__name__)

AVATAR_CANDIDATE_COUNT = 4
AVATAR_RATE_LIMIT = 6
AVATAR_RATE_WINDOW_SEC = 600.0
AVATAR_SOURCE_MAX_AGE_SEC = 600.0
AVATAR_MAX_PIXELS = 8192 * 8192
AVATAR_SIZE = 512
AVATAR_WEBP_QUALITY = 90
AVATAR_JOB_TTL_SEC = 86400.0
AVATAR_ETA_MS = 30000
GENERATOR_OUTPUT_MAX_BYTES = 64 * 1024 * 1024
WATCH_TIMEOUT_SEC = 240.0
POLL_MIN_SEC = 0.5
POLL_MAX_SEC = 2.0
VOICE_PREVIEW_TIMEOUT_SEC = 60.0
VOICE_PREVIEW_MAX_BYTES = 8 * 1024 * 1024
VOICE_PREVIEW_DEFAULT_TEXT = "你好，很高兴认识你。"
CLEANUP_INTERVAL_SEC = 3600.0
MAX_ROLES = 200
ROLE_SOURCE_IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")
REF_FACE_FILE = "ref_face.webp"

JOB_ID_RE = re.compile(r"^avj_[0-9A-Z]{26}$")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_TEMPLATE_FIELD_RE = re.compile(r"\{(source|output|style|prompt|voice|text)\}")

STYLE_TEXT: dict[str, str] = {
    "anime_clean": "clean Japanese anime character illustration, crisp line art, soft cel shading",
    "cyber_flat": "cyberpunk neon flat illustration, bold shapes, glowing rim light",
    "watercolor": "warm hand-painted watercolor illustration, soft paper texture",
    "pixel": "retro pixel art portrait, limited palette, crisp pixels",
}
VARIANTS: tuple[str, ...] = (
    "front-facing with a gentle smile",
    "head slightly tilted, three-quarter view",
    "soft backlight with a warm glow",
    "vivid high-saturation studio lighting",
)


def _clean_line(value: str) -> str:
    """Strip control characters and collapse whitespace into a single line."""
    return " ".join(_CTRL_RE.sub(" ", value or "").split())


def sanitize_prompt_hint(raw: str | None) -> str:
    """Return a single-line prompt hint without control characters or option-like `--` prefixes (≤ 80 chars)."""
    text = _clean_line(raw or "")
    while "--" in text:
        text = text.replace("--", "-")
    text = text.lstrip("-").strip()
    return text[: LIMITS["PROMPT_HINT_MAX"]]


def build_avatar_prompt(style_label: str, variant: str, hint: str) -> str:
    """Build the identity-preserving restyle prompt for one candidate."""
    prompt = (
        f"Restyle the person in Image 1 as a {style_label} portrait avatar; keep the same face shape, hairstyle, "
        f"hair color and signature accessories; {variant}; square head-and-shoulders composition, simple background"
    )
    if hint:
        prompt += f"; extra details: {hint}"
    return prompt


def fill_argv_template(template: list[str], values: dict[str, str]) -> list[str]:
    """Substitute `{name}` placeholders in a single pass (values are never re-expanded)."""
    return [_TEMPLATE_FIELD_RE.sub(lambda m: values.get(m.group(1), m.group(0)), str(arg)) for arg in template]


def sniff_image_kind(data: bytes) -> str | None:
    """Return `jpeg` / `png` / `webp` / `heic` from magic bytes, or None."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:8] == b"ftyp" and data[8:12] in (b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1"):
        return "heic"
    return None


_PIL_FORMATS = {"jpeg": {"JPEG", "MPO"}, "png": {"PNG"}, "webp": {"WEBP"}, "heic": {"HEIF", "HEIC"}}


def decode_image_safely(data: bytes) -> Image.Image:
    """Decode untrusted image bytes into a clean, metadata-free, EXIF-upright RGB/RGBA image.

    Raises:
        ValueError: unknown magic bytes, undecodable data, more than `AVATAR_MAX_PIXELS` or multiple frames.
    """
    kind = sniff_image_kind(data)
    if kind is None:
        raise ValueError("unsupported image type")
    if kind == "heic":
        try:
            import pillow_heif  # type: ignore[import-not-found]

            pillow_heif.register_heif_opener()
        except ImportError as e:
            raise ValueError("HEIC images are not supported on this host") from e
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data))
            if im.format not in _PIL_FORMATS[kind]:
                raise ValueError("image content does not match its type")
            width, height = im.size
            if width <= 0 or height <= 0 or width * height > AVATAR_MAX_PIXELS:
                raise ValueError("image dimensions out of range")
            if getattr(im, "n_frames", 1) > 1 or getattr(im, "is_animated", False):
                raise ValueError("multi-frame images are not allowed")
            im.load()
            im = ImageOps.exif_transpose(im)
            has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
            mode = "RGBA" if has_alpha else "RGB"
            im = im.convert(mode)
            return Image.frombytes(mode, im.size, im.tobytes())
    except ValueError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise ValueError("image too large") from e
    except Exception as e:  # noqa: BLE001 - any decoder failure means an invalid image
        raise ValueError(f"cannot decode image: {type(e).__name__}") from e


def encode_avatar_webp(im: Image.Image) -> bytes:
    """Center-crop to a square, resize to 512×512 and encode as WebP q90."""
    square = ImageOps.fit(im, (AVATAR_SIZE, AVATAR_SIZE), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    buf = io.BytesIO()
    square.save(buf, "WEBP", quality=AVATAR_WEBP_QUALITY, method=4)
    return buf.getvalue()


def encode_png(im: Image.Image) -> bytes:
    """Encode a clean image as PNG."""
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def sniff_audio_mime(data: bytes) -> str:
    """Return an audio MIME type from magic bytes."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if data[4:8] == b"ftyp":
        return "audio/mp4"
    return "application/octet-stream"


def _wav_duration_ms(data: bytes) -> int | None:
    """Return the duration of a PCM WAV payload, if parseable."""
    try:
        import wave

        with wave.open(io.BytesIO(data)) as w:
            frames, rate = w.getnframes(), w.getframerate()
        return int(frames * 1000 / rate) if rate else None
    except Exception:  # noqa: BLE001
        return None


def _within(root: str, path: str) -> bool:
    """Return True when `path` resolves strictly inside `root` (symlinks followed)."""
    return realpath_within(root, path)


def _read_bytes(path: str) -> bytes:
    """Read a whole file."""
    with open(path, "rb") as f:
        return f.read()


def _unlink_quiet(path: str) -> None:
    """Unlink ignoring errors."""
    with contextlib.suppress(OSError):
        os.unlink(path)


def _poll_deferred(target: str, pending_path: str, error_path: str) -> tuple[str, dict[str, Any] | None]:
    """Classify a deferred target: `ready`, `failed` (error dict), `missing`, or `pending` (marker)."""
    marker = read_json_dict(pending_path)
    if marker is None:
        err = read_json_dict(error_path)
        if err is not None:
            return "failed", err
    if is_deferred_target_ready(target, pending_path, marker):
        return "ready", marker
    if marker is None and not os.path.exists(target):
        return "missing", None
    return "pending", marker


class _CandidateFailed(Exception):
    """A single avatar candidate failed (reason is user-facing after sanitizing)."""


class PersonaService:
    """Persona workshop + user profile + avatar restyle feature service (see module docstring)."""

    def __init__(
        self,
        host: AppHost,
        *,
        characters_dir: str | None = None,
        watch_timeout_sec: float = WATCH_TIMEOUT_SEC,
        poll_min_sec: float = POLL_MIN_SEC,
        poll_max_sec: float = POLL_MAX_SEC,
    ) -> None:
        """Bind the service to the app host (no I/O until `start`)."""
        self.host = host
        self.registry = RoleRegistry(host.roles_dir)
        self.profiles = UserProfileStore(host.awd)
        self.characters_dir = os.path.abspath(characters_dir or os.path.join(host.awd, "characters"))
        self.watch_timeout_sec = float(watch_timeout_sec)
        self.poll_min_sec = float(poll_min_sec)
        self.poll_max_sec = float(poll_max_sec)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._procs: set[asyncio.subprocess.Process] = set()
        self._meta_lock = asyncio.Lock()
        self._rate_lock = asyncio.Lock()
        self._job_locks: dict[str, asyncio.Lock] = {}
        self._job_candidates: dict[str, list[dict[str, Any]]] = {}
        self._cleanup_task: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------------ lifecycle
    @property
    def jobs_root(self) -> str:
        """Absolute avatar job root (`<sessions_dir>/avatar_jobs`)."""
        return os.path.join(os.path.abspath(self.host.sessions_dir), "avatar_jobs")

    async def start(self) -> None:
        """Clean expired avatar jobs and start the periodic cleanup loop."""
        with contextlib.suppress(Exception):
            await self.cleanup_expired_jobs()
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Cancel background generation, kill launched generator processes and stop cleanup."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._cleanup_task
            self._cleanup_task = None
        for proc in list(self._procs):
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        tasks = list(self._tasks)
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def wait_idle(self) -> None:
        """Wait until every background avatar job finished (tests / graceful shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _cleanup_loop(self) -> None:
        """Periodically delete expired avatar jobs."""
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_SEC)
            try:
                await self.cleanup_expired_jobs()
            except Exception as e:  # noqa: BLE001
                logger.warning("Avatar job cleanup failed: %s", e)

    async def cleanup_expired_jobs(self) -> int:
        """Delete expired avatar job rows and their job directories; returns the number removed."""
        rows = await asyncio.to_thread(self.host.sync_db.list_expired_avatar_jobs)
        removed = 0
        for row in rows:
            job_id = str(row.get("job_id") or "")
            if job_id in self._job_candidates:
                continue
            await asyncio.to_thread(self._remove_job_dir, job_id)
            await asyncio.to_thread(self.host.sync_db.delete_avatar_job, job_id)
            removed += 1
        return removed

    def _remove_job_dir(self, job_id: str) -> None:
        """Delete a jailed job directory (worker thread)."""
        job_dir = self._job_dir(job_id)
        if job_dir and os.path.isdir(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        """Track a background task."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ------------------------------------------------------------------ helpers
    def _job_dir(self, job_id: str) -> str | None:
        """Return the jailed job directory for a syntactically valid `job_id`."""
        if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            return None
        path = os.path.join(self.jobs_root, job_id)
        return os.path.realpath(path) if _within(self.jobs_root, path) else None

    def _skills_dir(self) -> str:
        """Absolute skills directory from the active configuration."""
        raw = self.host.kconfig.workspace.skills_dir
        return raw if os.path.isabs(raw) else os.path.join(self.host.awd, raw)

    def _generator_argv(self, source: str, output: str, style: str, prompt: str) -> list[str]:
        """Return the generator argv for one candidate (raises BAD_REQUEST when no generator exists)."""
        values = {"source": source, "output": output, "style": style, "prompt": prompt}
        template = list(self.host.app_config.avatar_generator_command or [])
        if template:
            return fill_argv_template(template, values)
        script = os.path.join(self._skills_dir(), "ai-image", "scripts", "generate_image.py")
        if not os.path.isfile(script):
            raise ProtocolError("BAD_REQUEST", "avatar generator is not configured on this host")
        uv = shutil.which("uv") or "uv"
        return [
            uv, "run", script,
            "--image", source,
            "--no-auto-ref",
            "--aspect-ratio", "1:1",
            "--image-size", "1K",
            "--style", style,
            "--prompt", prompt,
            "--output", output,
            "--async",
        ]  # fmt: skip

    async def _fetch_avatar_source(self, ctx: RequestContext, object_name: Any) -> bytes:
        """Validate ownership/freshness of an uploaded avatar source object and download it."""
        if not isinstance(object_name, str) or not AVATAR_SOURCE_OBJECT_RE.fullmatch(object_name):
            raise ProtocolError("AVATAR_SOURCE_INVALID", "invalid avatar source object")
        try:
            info = await self.host.media.sign(object_name)
        except Exception as e:  # noqa: BLE001
            raise ProtocolError("AVATAR_SOURCE_INVALID", "avatar source object not found") from e
        if not isinstance(info, dict) or info.get("issued_device_id") != ctx.device_id:
            raise ProtocolError("AVATAR_SOURCE_INVALID", "avatar source was not uploaded by this device")
        try:
            issued_at = float(info.get("issued_at"))
        except (TypeError, ValueError) as e:
            raise ProtocolError("AVATAR_SOURCE_INVALID", "avatar source has no issue time") from e
        if issued_at > 1e12:  # milliseconds
            issued_at /= 1000.0
        age = time.time() - issued_at
        if age > AVATAR_SOURCE_MAX_AGE_SEC or age < -60.0:
            raise ProtocolError("AVATAR_SOURCE_INVALID", "avatar source upload has expired")
        url = info.get("url")
        if not isinstance(url, str) or not url:
            raise ProtocolError("AVATAR_SOURCE_INVALID", "avatar source has no download URL")
        max_bytes = LIMITS["AVATAR_SOURCE_MAX_BYTES"]
        try:
            data = await self.host.media.download(url, max_bytes)
        except Exception as e:  # noqa: BLE001
            raise ProtocolError("AVATAR_SOURCE_INVALID", "avatar source download failed") from e
        if not isinstance(data, (bytes, bytearray)) or not data or len(data) > max_bytes:
            raise ProtocolError("AVATAR_SOURCE_INVALID", "avatar source is empty or too large")
        return bytes(data)

    async def _decode_source(self, data: bytes) -> Image.Image:
        """Decode untrusted bytes off the event loop, mapping failures to AVATAR_SOURCE_INVALID."""
        try:
            return await asyncio.to_thread(decode_image_safely, data)
        except ValueError as e:
            raise ProtocolError("AVATAR_SOURCE_INVALID", str(e)) from e

    async def _avatar_webp_from_object(self, ctx: RequestContext, object_name: Any) -> bytes:
        """Fetch + sanitize a directly uploaded avatar and return 512×512 WebP bytes."""
        data = await self._fetch_avatar_source(ctx, object_name)
        im = await self._decode_source(data)
        return await asyncio.to_thread(encode_avatar_webp, im)

    def _require_role_id(self, payload: dict[str, Any], key: str = "role_id") -> str:
        """Return a valid role id from `payload[key]` or raise BAD_REQUEST."""
        role_id = payload.get(key)
        if not is_valid_role_id(role_id):
            raise ProtocolError("BAD_REQUEST", f"'{key}' must match ^[a-z0-9_]{{2,32}}$")
        return str(role_id)

    def _require_existing_role(self, role_id: str) -> str:
        """Return the jailed role directory or raise NOT_FOUND."""
        d = self.registry.role_dir(role_id)
        if not d or not self.registry.exists(role_id):
            raise ProtocolError("NOT_FOUND", f"role '{role_id}' not found")
        return d

    @staticmethod
    def _read_capped(path: str, limit: int = LIMITS["PERSONA_FILE_MAX_BYTES"]) -> str:
        """Read a UTF-8 text file capped at `limit` bytes ('' when missing)."""
        try:
            with open(path, "rb") as f:
                return f.read(limit).decode("utf-8", errors="replace")
        except OSError:
            return ""

    def resolve_avatar_ref(self, ref: str) -> str | None:
        """Resolve a `media.hydrate` avatar ref (`roles/<id>/avatar.webp` / `user/avatar.webp`) to a file."""
        if not isinstance(ref, str) or not AVATAR_REF_RE.fullmatch(ref):
            return None
        if ref.startswith("user/"):
            path = safe_resolve(self.profiles.user_dir, AVATAR_FILE)
        else:
            path = safe_resolve(self.host.roles_dir, ref[len("roles/") :])
        return path if path and os.path.isfile(path) else None

    async def _broadcast_roles(self, roles: list[dict[str, Any]], deleted: list[str] | None = None) -> None:
        """Broadcast `persona.updated` to every device."""
        payload: dict[str, Any] = {"roles": roles}
        if deleted:
            payload["deleted"] = deleted
        await self.host.broadcast("persona.updated", payload)

    # ------------------------------------------------------------------ persona.*
    async def handle_persona_list(self, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any]:
        """`persona.list` → `{roles:[PersonaSummary]}`."""
        roles = await asyncio.to_thread(lambda: [self._role_view(r) for r in self.registry.list_roles()])
        return {"roles": roles}

    async def handle_persona_get(self, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any]:
        """`persona.get{role_id}` → `PersonaDetail` (summary + `intro_md` + `preferences_md`)."""
        role_id = self._require_role_id(payload)

        def _load() -> dict[str, Any]:
            d = self._require_existing_role(role_id)
            meta = self.registry.load_meta(role_id) or {}
            detail = self._role_view(self.registry.summary_from_meta(role_id, meta))
            detail["intro_md"] = self._read_capped(os.path.join(d, INTRO_MD))
            detail["preferences_md"] = self._read_capped(os.path.join(d, PREFERENCES_MD))
            readonly = [f for f in meta.get("readonly_files") or [] if f in (INTRO_MD, PREFERENCES_MD)]
            if readonly:
                detail["readonly_notes"] = "Read-only: " + ", ".join(readonly)
            return detail

        return await asyncio.to_thread(_load)

    async def handle_persona_upsert(self, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any]:
        """`persona.upsert` → create/update a role; broadcasts `persona.updated`."""
        display_name = _clean_line(require_str(payload, "display_name", max_len=256))
        if not display_name or len(display_name) > LIMITS["DISPLAY_NAME_MAX"]:
            raise ProtocolError("BAD_REQUEST", f"'display_name' must be 1-{LIMITS['DISPLAY_NAME_MAX']} chars")
        tagline_raw = optional_str(payload, "tagline", max_len=1024)
        tagline = _clean_line(tagline_raw) if tagline_raw is not None else None
        if tagline is not None and len(tagline) > LIMITS["TAGLINE_MAX"]:
            raise ProtocolError("BAD_REQUEST", f"'tagline' must be ≤ {LIMITS['TAGLINE_MAX']} chars")
        docs: dict[str, str] = {}
        for key, fname in (("intro_md", INTRO_MD), ("preferences_md", PREFERENCES_MD)):
            value = payload.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or len(value.encode("utf-8")) > LIMITS["PERSONA_FILE_MAX_BYTES"]:
                raise ProtocolError("BAD_REQUEST", f"'{key}' must be a string ≤ 64 KiB")
            docs[fname] = value.replace("\x00", "")
        voice_set = "voice_preset" in payload
        voice = optional_str(payload, "voice_preset", max_len=64)
        voice = _clean_line(voice) or None if voice is not None else None
        role_id: str | None = None
        if payload.get("role_id") is not None:
            role_id = self._require_role_id(payload)
        if payload.get("new_role_id") not in (None, role_id):
            if role_id and await asyncio.to_thread(self._is_builtin, role_id):
                raise ProtocolError("PERSONA_BUILTIN", "built-in roles cannot be renamed")
            raise ProtocolError("BAD_REQUEST", "renaming roles is not supported; create a new role instead")
        avatar = None
        if payload.get("avatar_object"):
            avatar = await self._avatar_webp_from_object(ctx, payload["avatar_object"])

        async with self._meta_lock:
            summary = await asyncio.to_thread(
                self._upsert_role_sync, role_id, display_name, tagline, docs, voice_set, voice, avatar
            )
        await self._broadcast_roles([summary])
        return {"ok": True, "role": summary}

    def _is_builtin(self, role_id: str) -> bool:
        """Return True when `role_id` exists and is marked built-in."""
        meta = self.registry.load_meta(role_id)
        return bool(meta and meta.get("builtin"))

    def _upsert_role_sync(
        self,
        role_id: str | None,
        display_name: str,
        tagline: str | None,
        docs: dict[str, str],
        voice_set: bool,
        voice: str | None,
        avatar: bytes | None,
    ) -> dict[str, Any]:
        """Write role files + metadata (worker thread, under the meta lock)."""
        if role_id is None:
            role_id = self.registry.unique_role_id(display_name)
        creating = not self.registry.exists(role_id)
        if creating and len(self.registry.list_role_ids()) >= MAX_ROLES:
            raise ProtocolError("BAD_REQUEST", "too many roles")
        d = self.registry.role_dir(role_id)
        if not d:
            raise ProtocolError("BAD_REQUEST", "invalid role directory")
        os.makedirs(os.path.join(d, "memories"), exist_ok=True)
        if not _within(self.host.roles_dir, d):
            raise ProtocolError("BAD_REQUEST", "invalid role directory")
        meta: dict[str, Any]
        if creating:
            meta = {"builtin": False, "voice_preset": None, "avatar_rev": 0, "tagline": ""}
        else:
            meta = self.registry.load_meta(role_id) or {}
        readonly = set(meta.get("readonly_files") or [])
        for fname in (INTRO_MD, PREFERENCES_MD):
            target = os.path.join(d, fname)
            if fname in docs and fname not in readonly:
                atomic_write_bytes(target, docs[fname].encode("utf-8"))
            elif creating and not os.path.exists(target):
                atomic_write_bytes(target, b"")
        meta["display_name"] = display_name
        if tagline is not None:
            meta["tagline"] = tagline
        if voice_set:
            meta["voice_preset"] = voice
        if avatar is not None:
            atomic_write_bytes(os.path.join(d, AVATAR_FILE), avatar)
            meta["avatar_rev"] = int(meta.get("avatar_rev") or 0) + 1
            meta["avatar_ai"] = False
        meta["updated_at"] = time.time()
        self.registry.write_meta(role_id, meta)
        if creating:
            profile = self.profiles.load()
            try:
                sync_profile_block_to_role(
                    self.host.sync_db, role_id, os.path.join(d, "memories"), profile["nickname"], profile["bio"]
                )
            except OSError as e:
                logger.warning("Could not seed user profile block into new role %s: %s", role_id, e)
        return self._role_view(self.registry.get(role_id) or {})

    async def handle_persona_delete(self, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any]:
        """`persona.delete{role_id}` → move to `.trash/`, reassign references, broadcast `persona.updated`."""
        role_id = self._require_role_id(payload)
        default_role = self.host.app_config.default_role
        async with self._meta_lock:
            await asyncio.to_thread(self._delete_role_sync, role_id, default_role)
        await asyncio.to_thread(self.host.sync_db.reassign_role_references, role_id, default_role)
        gone_dir = os.path.join(self.host.roles_dir, role_id, "memories")
        with contextlib.suppress(Exception):
            await asyncio.to_thread(sync_role_memories, self.host.sync_db, role_id, gone_dir)
        default_pinned_cache.invalidate(role_id)
        await self._broadcast_roles([], deleted=[role_id])
        return {"ok": True}

    def _delete_role_sync(self, role_id: str, default_role: str) -> None:
        """Move role (and its characters dir) into trash (worker thread)."""
        d = self._require_existing_role(role_id)
        if self._is_builtin(role_id):
            raise ProtocolError("PERSONA_BUILTIN", "built-in roles cannot be deleted")
        if role_id == default_role:
            raise ProtocolError("BAD_REQUEST", "the default role cannot be deleted")
        stamp = f"{role_id}-{int(time.time() * 1000)}"
        trash = os.path.join(os.path.realpath(self.host.roles_dir), ".trash")
        os.makedirs(trash, exist_ok=True)
        shutil.move(d, os.path.join(trash, stamp))
        cdir = os.path.join(self.characters_dir, role_id)
        if os.path.isdir(cdir) and _within(self.characters_dir, cdir):
            ctrash = os.path.join(os.path.realpath(self.characters_dir), ".trash")
            os.makedirs(ctrash, exist_ok=True)
            shutil.move(os.path.realpath(cdir), os.path.join(ctrash, stamp))
        self.registry.forget(role_id)

    # ------------------------------------------------------------------ voice.preview
    async def handle_voice_preview(self, ctx: RequestContext, payload: dict[str, Any]) -> Reply:
        """`voice.preview{voice_preset, text?}` → `media.ready{kind:"voice"}` via `voice_preview_command`."""
        voice = _clean_line(require_str(payload, "voice_preset", max_len=64)).lstrip("-").strip()
        if not voice:
            raise ProtocolError("BAD_REQUEST", "'voice_preset' must be a non-empty string")
        text_raw = optional_str(payload, "text", max_len=80)
        text = sanitize_prompt_hint(text_raw) if text_raw else VOICE_PREVIEW_DEFAULT_TEXT
        text = text or VOICE_PREVIEW_DEFAULT_TEXT
        template = list(self.host.app_config.voice_preview_command or [])
        if not template:
            raise ProtocolError("BAD_REQUEST", "voice preview is not configured on this host")
        output = await asyncio.to_thread(self._voice_output_path)
        argv = fill_argv_template(template, {"voice": voice, "text": text, "output": output})
        try:
            await self._run_process(argv, VOICE_PREVIEW_TIMEOUT_SEC)
            head = await asyncio.to_thread(self._read_voice_output, output)
            size = len(head)
            mime = sniff_audio_mime(head)
            uploaded = await self.host.media.upload_file(output, purpose="asset", mime=mime)
        except _CandidateFailed as e:
            raise ProtocolError("INTERNAL", sanitize_error_reason(str(e), "voice preview failed")) from e
        finally:
            await asyncio.to_thread(_unlink_quiet, output)
        result: dict[str, Any] = {
            "kind": "voice",
            "object": uploaded.get("object"),
            "url": uploaded.get("url"),
            "mime": uploaded.get("mime") or mime,
            "bytes": int(uploaded.get("bytes") or size),
            "expires_at": uploaded.get("expires_at"),
        }
        duration = _wav_duration_ms(head) if mime == "audio/wav" else None
        if duration is not None:
            result["duration_ms"] = duration
        return Reply("media.ready", result)

    def _voice_output_path(self) -> str:
        """Allocate a voice preview output path under `<sessions_dir>/voice_previews/`."""
        out_dir = os.path.join(os.path.abspath(self.host.sessions_dir), "voice_previews")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{new_id('vp')}.wav")

    @staticmethod
    def _read_voice_output(output: str) -> bytes:
        """Validate and read a generated voice preview."""
        if not os.path.isfile(output):
            raise ProtocolError("INTERNAL", "voice preview produced no audio")
        size = os.path.getsize(output)
        if size <= 0 or size > VOICE_PREVIEW_MAX_BYTES:
            raise ProtocolError("INTERNAL", "voice preview output size out of range")
        return _read_bytes(output)

    def _proc_cwd(self) -> str | None:
        """Working directory for launched tools (`$AWD` when it exists)."""
        return self.host.awd if os.path.isdir(self.host.awd) else None

    async def _run_process(self, argv: list[str], limit_sec: float) -> tuple[int, str]:
        """Run argv (never through a shell) within `limit_sec`; raise `_CandidateFailed` on failure."""
        if not argv:
            raise _CandidateFailed("empty command")
        cwd = self._proc_cwd()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except OSError as e:
            raise _CandidateFailed(f"cannot launch generator: {e.strerror or e}") from e
        self._procs.add(proc)
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=max(1.0, limit_sec))
        except TimeoutError as e:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            raise _CandidateFailed("generation timed out") from e
        finally:
            self._procs.discard(proc)
        stderr_tail = (err or b"").decode("utf-8", errors="replace").strip().splitlines()[-3:]
        tail = " ".join(stderr_tail)
        if proc.returncode != 0:
            raise _CandidateFailed(tail or f"exit status {proc.returncode}")
        return int(proc.returncode or 0), tail

    # ------------------------------------------------------------------ user_profile.*
    async def handle_user_profile_get(self, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any]:
        """`user_profile.get` → `UserProfile`."""
        return await asyncio.to_thread(lambda: self._profile_view(self.profiles.public_view()))

    async def handle_user_profile_put(self, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any]:
        """`user_profile.put{nickname, bio, cameo_opt_in, avatar_object?}` → save + sync + broadcast."""
        nickname = payload.get("nickname")
        if nickname is not None:
            if not isinstance(nickname, str):
                raise ProtocolError("BAD_REQUEST", "'nickname' must be a string")
            nickname = _clean_line(nickname)
            if len(nickname) > NICKNAME_MAX:
                raise ProtocolError("BAD_REQUEST", f"'nickname' must be ≤ {NICKNAME_MAX} chars")
        bio = payload.get("bio")
        if bio is not None:
            if not isinstance(bio, str) or len(bio.encode("utf-8")) > BIO_MAX_BYTES:
                raise ProtocolError("BAD_REQUEST", "'bio' must be a string ≤ 2 KiB")
            bio = bio.replace("\x00", "").strip()
        cameo = payload.get("cameo_opt_in")
        if cameo is not None and not isinstance(cameo, bool):
            raise ProtocolError("BAD_REQUEST", "'cameo_opt_in' must be a boolean")
        avatar = None
        if payload.get("avatar_object"):
            avatar = await self._avatar_webp_from_object(ctx, payload["avatar_object"])
        async with self._meta_lock:
            view = await asyncio.to_thread(self._put_profile_sync, nickname, bio, cameo, avatar)
        await self.host.broadcast("user_profile.updated", view)
        return {"ok": True, "profile": view}

    def _put_profile_sync(
        self, nickname: str | None, bio: str | None, cameo: bool | None, avatar: bytes | None
    ) -> dict[str, Any]:
        """Persist the profile, reconcile the face reference and sync managed blocks (worker thread)."""
        prev = self.profiles.load()
        profile = dict(prev)
        if nickname is not None:
            profile["nickname"] = nickname
        if bio is not None:
            profile["bio"] = bio
        if cameo is not None:
            profile["cameo_opt_in"] = cameo
        if avatar is not None:
            atomic_write_bytes(self.profiles.avatar_path(), avatar)
            profile["avatar_rev"] = int(profile.get("avatar_rev") or 0) + 1
            profile["avatar_ai"] = False
        self.profiles.save(profile)
        cdir = profile.get("character_dir") or "user"
        if not profile["cameo_opt_in"]:
            self._restore_reference(cdir)
        elif not prev.get("cameo_opt_in"):
            names = [profile["nickname"]] if profile["nickname"] else []
            self._register_reference(cdir, None, names)
        sync_profile_to_roles(
            self.host.sync_db,
            self.host.roles_dir,
            self.registry.list_role_ids(),
            profile["nickname"],
            profile["bio"],
        )
        return self._profile_view(self.profiles.public_view(profile))

    # ------------------------------------------------------------------ characters.json registry
    def _character_dir(self, dir_name: str, *, create: bool) -> str | None:
        """Return the jailed `<characters_dir>/<dir_name>` path (optionally creating it)."""
        if not isinstance(dir_name, str) or not CHARACTER_DIR_RE.fullmatch(dir_name):
            return None
        path = os.path.join(self.characters_dir, dir_name)
        if create:
            os.makedirs(os.path.join(path, "images"), exist_ok=True)
        if not _within(self.characters_dir, path):
            return None
        if create and not _within(path, os.path.join(path, "images")):
            return None
        return os.path.realpath(path)

    @staticmethod
    def _load_registry(path: str) -> list[dict[str, Any]]:
        """Load a characters.json list (missing = empty; unreadable raises so it is never clobbered)."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as e:
            raise ProtocolError("INTERNAL", "characters.json is unreadable; refusing to overwrite") from e
        if not isinstance(data, list):
            raise ProtocolError("INTERNAL", "characters.json is not a list; refusing to overwrite")
        return [e for e in data if isinstance(e, dict)]

    @staticmethod
    def _find_entry(entries: list[dict[str, Any]], names: list[str]) -> dict[str, Any] | None:
        """Find the entry whose name/aliases match `names` (or the only entry of the file)."""
        wanted = {n.casefold() for n in names if n}
        for e in entries:
            pool = [str(e.get("name") or "")] + [str(a) for a in e.get("aliases") or [] if isinstance(a, str)]
            if wanted & {p.casefold() for p in pool if p}:
                return e
        return entries[0] if len(entries) == 1 else None

    def _register_reference(self, dir_name: str, data: bytes | None, names: list[str]) -> bool:
        """Write `images/ref_face.webp` (when `data`) and upsert `characters.json` in place."""
        cdir = self._character_dir(dir_name, create=data is not None)
        if not cdir:
            return False
        ref = os.path.join(cdir, "images", "ref_face.webp")
        if data is not None:
            atomic_write_bytes(ref, data)
        elif not os.path.isfile(ref):
            return False
        reg_path = os.path.join(cdir, "characters.json")
        entries = self._load_registry(reg_path)
        rel_image = os.path.relpath(ref, os.path.dirname(os.path.realpath(self.characters_dir))).replace(os.sep, "/")
        clean_names = [n for n in dict.fromkeys(_clean_line(x) for x in names) if n] or [dir_name]
        entry = self._find_entry(entries, clean_names)
        prev_path = os.path.join(cdir, ".kesoku_prev_image")
        if entry is None:
            entry = {"name": clean_names[0], "aliases": [], "image": "", "draftable": False}
            entries.append(entry)
        if not os.path.exists(prev_path) and entry.get("image") != rel_image:
            prev = {"name": entry.get("name"), "image": entry.get("image") or ""}
            atomic_write_bytes(prev_path, json.dumps(prev, ensure_ascii=False).encode("utf-8"))
        entry["image"] = rel_image
        aliases = [a for a in entry.get("aliases") or [] if isinstance(a, str)]
        for n in clean_names:
            if n != entry.get("name") and n not in aliases:
                aliases.append(n)
        entry["aliases"] = aliases
        atomic_write_bytes(reg_path, (json.dumps(entries, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        return True

    def _restore_reference(self, dir_name: str) -> bool:
        """Revert the registry `image` to its pre-registration value (entry and aliases are kept)."""
        cdir = self._character_dir(dir_name, create=False)
        if not cdir:
            return False
        prev_path = os.path.join(cdir, ".kesoku_prev_image")
        prev = read_json_dict(prev_path)
        if not prev:
            return False
        reg_path = os.path.join(cdir, "characters.json")
        entries = self._load_registry(reg_path)
        target = next((e for e in entries if e.get("name") == prev.get("name")), None)
        if target is None and len(entries) == 1:
            target = entries[0]
        if target is not None:
            target["image"] = str(prev.get("image") or "")
            atomic_write_bytes(reg_path, (json.dumps(entries, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        with contextlib.suppress(OSError):
            os.unlink(prev_path)
        return True

    # ------------------------------------------------------------------ existing avatar sources
    def _role_view(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Attach `has_source_image` to a role summary (worker thread)."""
        role_id = summary.get("role_id")
        if isinstance(role_id, str) and role_id:
            summary["has_source_image"] = self.has_existing_avatar_source("role", role_id)
        return summary

    def _profile_view(self, view: dict[str, Any]) -> dict[str, Any]:
        """Attach `has_source_image` to a `UserProfile` view (worker thread)."""
        view["has_source_image"] = self.has_existing_avatar_source("user", "user")
        return view

    def _jailed_source_file(self, path: str | None) -> str | None:
        """Return the real path of a regular, non-empty, size-capped file strictly inside the AWD."""
        if not isinstance(path, str) or not path or "\x00" in path:
            return None
        if not _within(self.host.awd, path):
            return None
        real = os.path.realpath(path)
        try:
            if not os.path.isfile(real):
                return None
            size = os.path.getsize(real)
        except OSError:
            return None
        if not 0 < size <= LIMITS["AVATAR_SOURCE_MAX_BYTES"]:
            return None
        return real

    def _registry_image_path(self, rel: Any) -> str | None:
        """Resolve a characters.json `image` value against the parent of `characters_dir` (unjailed)."""
        if not isinstance(rel, str) or not rel.strip() or "\x00" in rel:
            return None
        rel = rel.strip()
        if os.path.isabs(rel):
            return rel
        char_parent = os.path.dirname(os.path.realpath(self.characters_dir))
        return os.path.join(char_parent, rel)

    def _avatar_source_candidates(self, target_type: str, target_id: str) -> list[str]:
        """Return candidate source image paths for `avatar.generate` without an upload, best first.

        Role: pre-registration original (`.kesoku_prev_image`), registry image (non-reference), the role's
        `images/<id>.<ext>`, the role avatar, then any face reference. User: pre-registration original, the
        user avatar, then the registry image / face reference. Candidates are not yet jailed or validated.
        """
        names: list[str]
        if target_type == "role":
            if not is_valid_role_id(target_id):
                return []
            dir_name = target_id
            meta = self.registry.load_meta(target_id) or {}
            names = [str(meta.get("display_name") or ""), target_id]
        elif target_type == "user":
            profile = self.profiles.load()
            dir_name = str(profile.get("character_dir") or "user")
            names = [str(profile.get("nickname") or "")]
        else:
            return []
        prev_images: list[str] = []
        primary: list[str] = []
        references: list[str] = []
        cdir = self._character_dir(dir_name, create=False)
        if cdir:
            prev = read_json_dict(os.path.join(cdir, ".kesoku_prev_image")) or {}
            prev_path = self._registry_image_path(prev.get("image"))
            if prev_path:
                prev_images.append(prev_path)
            try:
                entries = self._load_registry(os.path.join(cdir, "characters.json"))
            except ProtocolError:
                entries = []
            matched = self._find_entry(entries, names)
            ordered = ([matched] if matched is not None else []) + [e for e in entries if e is not matched]
            for entry in ordered:
                path = self._registry_image_path(entry.get("image"))
                if not path:
                    continue
                (references if os.path.basename(path) == REF_FACE_FILE else primary).append(path)
            references.append(os.path.join(cdir, "images", REF_FACE_FILE))
        out = list(prev_images)
        if target_type == "role":
            role_dir = os.path.join(self.host.roles_dir, target_id)
            out += primary
            out += [os.path.join(role_dir, "images", f"{target_id}.{ext}") for ext in ROLE_SOURCE_IMAGE_EXTS]
            out.append(os.path.join(role_dir, AVATAR_FILE))
            out += references
        else:
            out.append(self.profiles.avatar_path())
            out += primary
            out += references
        return list(dict.fromkeys(out))

    def has_existing_avatar_source(self, target_type: str, target_id: str) -> bool:
        """Return True when an on-disk image can seed `avatar.generate` without an upload (cheap magic check)."""
        for candidate in self._avatar_source_candidates(target_type, target_id):
            real = self._jailed_source_file(candidate)
            if not real:
                continue
            try:
                with open(real, "rb") as f:
                    head = f.read(16)
            except OSError:
                continue
            if sniff_image_kind(head) is not None:
                return True
        return False

    def _load_existing_avatar_source_sync(self, target_type: str, target_id: str) -> bytes:
        """Return the bytes of the best decodable on-disk source image (worker thread).

        Raises:
            ProtocolError: `AVATAR_SOURCE_INVALID` when no jailed, size-capped, decodable image exists.
        """
        for candidate in self._avatar_source_candidates(target_type, target_id):
            real = self._jailed_source_file(candidate)
            if not real:
                continue
            try:
                data = _read_bytes(real)
            except OSError:
                continue
            if not data or len(data) > LIMITS["AVATAR_SOURCE_MAX_BYTES"]:
                continue
            try:
                decode_image_safely(data)
            except ValueError:
                continue
            return data
        raise ProtocolError("AVATAR_SOURCE_INVALID", "no existing avatar image; please upload a photo first")

    # ------------------------------------------------------------------ avatar.generate
    async def handle_avatar_generate(self, ctx: RequestContext, payload: dict[str, Any]) -> Reply:
        """`avatar.generate` → immediate `avatar.job`; candidates follow as `media.ready` / `media.failed`.

        `source_object` is optional: when omitted (or empty) the target's existing on-disk image is used.
        """
        target_type = payload.get("target_type")
        if target_type not in ("user", "role"):
            raise ProtocolError("BAD_REQUEST", "'target_type' must be 'user' or 'role'")
        if target_type == "role":
            target_id = self._require_role_id(payload, "target_id")
            await asyncio.to_thread(self._require_existing_role, target_id)
        else:
            target_id = "user"
        style_preset = payload.get("style_preset")
        if not isinstance(style_preset, str) or style_preset not in self.host.app_config.avatar_style_presets:
            raise ProtocolError("BAD_REQUEST", "unknown 'style_preset'")
        hint_raw = optional_str(payload, "prompt_hint", max_len=LIMITS["PROMPT_HINT_MAX"])
        hint = sanitize_prompt_hint(hint_raw)
        count = payload.get("count", AVATAR_CANDIDATE_COUNT)
        if count not in (None, AVATAR_CANDIDATE_COUNT) or isinstance(count, bool):
            raise ProtocolError("BAD_REQUEST", "'count' must be 4")
        style_label = STYLE_TEXT.get(style_preset, style_preset.replace("_", " "))
        # Fail fast when no generator is available (argv built again per candidate).
        self._generator_argv("source.png", "cand.png", style_label, "probe")

        async with self._rate_lock:
            since = time.time() - AVATAR_RATE_WINDOW_SEC
            recent = await asyncio.to_thread(self.host.sync_db.count_recent_avatar_jobs, ctx.device_id, since)
            if recent >= AVATAR_RATE_LIMIT:
                raise ProtocolError("RATE_LIMITED", "too many avatar jobs; try again in a few minutes")
            source_object = payload.get("source_object")
            if source_object is None or source_object == "":
                data = await asyncio.to_thread(self._load_existing_avatar_source_sync, target_type, target_id)
            else:
                data = await self._fetch_avatar_source(ctx, source_object)
            im = await self._decode_source(data)
            job_id = new_id("avj")
            job_dir = self._job_dir(job_id)
            if not job_dir:
                raise ProtocolError("INTERNAL", "cannot allocate avatar job")
            os.makedirs(job_dir, exist_ok=True)
            source = os.path.join(job_dir, "source.png")
            png = await asyncio.to_thread(encode_png, im)
            await asyncio.to_thread(atomic_write_bytes, source, png)
            placeholders = [new_id("ph") for _ in range(AVATAR_CANDIDATE_COUNT)]
            candidates = [
                {"index": i, "placeholder_id": ph, "status": "pending"} for i, ph in enumerate(placeholders)
            ]
            now = time.time()
            await asyncio.to_thread(
                lambda: self.host.sync_db.insert_avatar_job(
                    job_id=job_id,
                    device_id=ctx.device_id,
                    target_type=target_type,
                    target_id=target_id,
                    style_preset=style_preset,
                    prompt_hint=hint or None,
                    source_path=source,
                    candidates=candidates,
                    status="running",
                    created_at=now,
                    expires_at=now + AVATAR_JOB_TTL_SEC,
                )
            )
        self._job_candidates[job_id] = [dict(c) for c in candidates]
        self._job_locks[job_id] = asyncio.Lock()
        self._spawn(self._run_job(job_id, ctx.device_id, job_dir, source, style_label, hint, placeholders))
        return Reply("avatar.job", {"job_id": job_id, "placeholders": placeholders, "eta_ms": AVATAR_ETA_MS})

    async def _run_job(
        self,
        job_id: str,
        device_id: str,
        job_dir: str,
        source: str,
        style_label: str,
        hint: str,
        placeholders: list[str],
    ) -> None:
        """Run every candidate concurrently and record the final job status."""
        try:
            await asyncio.gather(
                *(
                    self._run_candidate(job_id, device_id, job_dir, source, style_label, hint, i, ph)
                    for i, ph in enumerate(placeholders)
                ),
                return_exceptions=True,
            )
            cands = self._job_candidates.get(job_id, [])
            status = "done" if any(c.get("status") == "ready" for c in cands) else "failed"
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.host.sync_db.update_avatar_job, job_id, status=status)
        finally:
            self._job_candidates.pop(job_id, None)
            self._job_locks.pop(job_id, None)

    async def _record_candidate(self, job_id: str, index: int, **fields: Any) -> None:
        """Update one candidate's state in memory and persist the whole list."""
        lock = self._job_locks.get(job_id)
        cands = self._job_candidates.get(job_id)
        if lock is None or cands is None:
            return
        async with lock:
            for c in cands:
                if c.get("index") == index:
                    c.update(fields)
            snapshot = [dict(c) for c in cands]
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.host.sync_db.update_avatar_job, job_id, candidates=snapshot)

    async def _run_candidate(
        self,
        job_id: str,
        device_id: str,
        job_dir: str,
        source: str,
        style_label: str,
        hint: str,
        index: int,
        placeholder_id: str,
    ) -> None:
        """Generate, normalize, upload and announce one candidate (never raises except cancellation)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.watch_timeout_sec
        output = os.path.join(job_dir, f"cand_{index}.png")
        webp_path = os.path.join(job_dir, f"cand_{index}.webp")
        prompt = build_avatar_prompt(style_label, VARIANTS[index % len(VARIANTS)], hint)
        try:
            argv = self._generator_argv(source, output, style_label, prompt)
            _, tail = await self._run_process(argv, self.watch_timeout_sec)
            await self._watch_output(output, deadline, tail)
            webp = await asyncio.to_thread(self._normalize_candidate, output)
            await asyncio.to_thread(atomic_write_bytes, webp_path, webp)
            uploaded = await self.host.media.upload_file(
                webp_path, purpose="avatar_candidate", mime="image/webp", index=index
            )
            await self._record_candidate(
                job_id,
                index,
                status="ready",
                file=os.path.basename(webp_path),
                object=uploaded.get("object"),
                mime="image/webp",
                bytes=len(webp),
            )
            await self.host.send_to_device(
                device_id,
                "media.ready",
                {
                    "placeholder_id": placeholder_id,
                    "kind": "avatar_candidate",
                    "job_id": job_id,
                    "index": index,
                    "object": uploaded.get("object"),
                    "url": uploaded.get("url"),
                    "mime": uploaded.get("mime") or "image/webp",
                    "bytes": int(uploaded.get("bytes") or len(webp)),
                    "expires_at": uploaded.get("expires_at"),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - each candidate fails independently
            if isinstance(e, ProtocolError):
                raw = e.message
            elif isinstance(e, (_CandidateFailed, ValueError)):
                raw = str(e)
            else:
                logger.warning("Avatar candidate %s/%d failed: %s", job_id, index, e)
                raw = "candidate generation failed"
            reason = sanitize_error_reason(raw, "candidate generation failed")
            await self._record_candidate(job_id, index, status="failed", reason=reason)
            with contextlib.suppress(Exception):
                await self.host.send_to_device(
                    device_id, "media.failed", {"placeholder_id": placeholder_id, "reason": reason}
                )

    def _normalize_candidate(self, path: str) -> bytes:
        """Decode a generated candidate and return 512×512 WebP bytes."""
        size = os.path.getsize(path)
        if size <= 0 or size > GENERATOR_OUTPUT_MAX_BYTES:
            raise ValueError("generated image size out of range")
        with open(path, "rb") as f:
            data = f.read()
        return encode_avatar_webp(decode_image_safely(data))

    async def _watch_output(self, target: str, deadline: float, stderr_tail: str) -> None:
        """Wait for a deferred target via its sidecars (0.5 s → 2 s backoff, bounded by `deadline`)."""
        loop = asyncio.get_running_loop()
        pending_path = f"{target}.pending.json"
        error_path = f"{target}.error.json"
        delay = self.poll_min_sec
        dead_since: float | None = None
        while True:
            state, marker = _poll_deferred(target, pending_path, error_path)
            if state == "ready":
                return
            if state == "failed":
                raise _CandidateFailed(str((marker or {}).get("error") or "generation failed"))
            if state == "missing":
                raise _CandidateFailed(stderr_tail or "generator produced no output")
            if marker:
                try:
                    pid = int(marker.get("pid") or 0)
                except (TypeError, ValueError):
                    pid = 0
                if not is_pid_alive(pid):
                    dead_since = dead_since if dead_since is not None else loop.time()
                    if loop.time() - dead_since >= DEAD_WORKER_GRACE_SEC:
                        raise _CandidateFailed("generator worker exited unexpectedly")
                else:
                    dead_since = None
            if loop.time() >= deadline:
                with contextlib.suppress(Exception):
                    await terminate_deferred_worker(target, marker)
                raise _CandidateFailed("generation timed out")
            await asyncio.sleep(min(delay, max(0.0, deadline - loop.time()) + 0.01))
            delay = min(self.poll_max_sec, delay * 2)

    # ------------------------------------------------------------------ avatar.apply
    async def handle_avatar_apply(self, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any]:
        """`avatar.apply{job_id, index, target_type, target_id, register_reference}` → write avatar + broadcast."""
        job_id = payload.get("job_id")
        job_dir = self._job_dir(job_id) if isinstance(job_id, str) else None
        if not job_dir:
            raise ProtocolError("BAD_REQUEST", "invalid 'job_id'")
        index = payload.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < AVATAR_CANDIDATE_COUNT:
            raise ProtocolError("BAD_REQUEST", "'index' must be 0-3")
        register = payload.get("register_reference", False)
        if not isinstance(register, bool):
            raise ProtocolError("BAD_REQUEST", "'register_reference' must be a boolean")
        job = await asyncio.to_thread(self.host.sync_db.get_avatar_job, job_id)
        if not job or job.get("device_id") != ctx.device_id or float(job.get("expires_at") or 0) <= time.time():
            raise ProtocolError("NOT_FOUND", "avatar job not found or expired")
        target_type = payload.get("target_type", job["target_type"])
        target_id = payload.get("target_id", job["target_id"])
        if target_type != job["target_type"] or (target_type == "role" and target_id != job["target_id"]):
            raise ProtocolError("BAD_REQUEST", "target does not match the avatar job")
        cand = next((c for c in job.get("candidates") or [] if c.get("index") == index), None)
        if not cand or cand.get("status") != "ready":
            raise ProtocolError("BAD_REQUEST", "candidate is not ready")
        path = safe_resolve(job_dir, f"cand_{index}.webp")
        try:
            data = await asyncio.to_thread(_read_bytes, path or "")
        except OSError as e:
            raise ProtocolError("NOT_FOUND", "candidate file is gone") from e
        async with self._meta_lock:
            if target_type == "role":
                summary = await asyncio.to_thread(self._apply_role_avatar_sync, str(job["target_id"]), data, register)
                rev = summary.get("avatar_rev", 0)
            else:
                view = await asyncio.to_thread(self._apply_user_avatar_sync, data, register)
                rev = view.get("avatar_rev", 0)
        if target_type == "role":
            await self._broadcast_roles([summary])
        else:
            await self.host.broadcast("user_profile.updated", view)
        return {"ok": True, "avatar_rev": int(rev)}

    def _apply_role_avatar_sync(self, role_id: str, data: bytes, register: bool) -> dict[str, Any]:
        """Write a role avatar (+ optional face reference) and bump `avatar_rev` (worker thread)."""
        d = self._require_existing_role(role_id)
        atomic_write_bytes(os.path.join(d, AVATAR_FILE), data)
        meta = self.registry.bump_avatar_rev(role_id, avatar_ai=True)
        if register:
            names = [str(meta.get("display_name") or ""), role_id]
            self._register_reference(role_id, data, names)
        return self._role_view(self.registry.get(role_id) or {})

    def _apply_user_avatar_sync(self, data: bytes, register: bool) -> dict[str, Any]:
        """Write the user avatar (+ optional face reference / cameo opt-in) (worker thread)."""
        atomic_write_bytes(self.profiles.avatar_path(), data)
        profile = self.profiles.load()
        profile["avatar_rev"] = int(profile.get("avatar_rev") or 0) + 1
        profile["avatar_ai"] = True
        if register:
            profile["cameo_opt_in"] = True
        self.profiles.save(profile)
        if register:
            names = [profile["nickname"]] if profile.get("nickname") else []
            self._register_reference(profile.get("character_dir") or "user", data, names)
        return self._profile_view(self.profiles.public_view(profile))

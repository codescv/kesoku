"""`AppChatbot`: the app channel adapter.

The engine host opens exactly one outbound WebSocket to a relay (`EngineLink`). Client requests arrive as V1
envelopes; replies and session events go back through the same link. All state (sessions, outbox, memory,
roles) stays on the engine host.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from importlib import metadata as importlib_metadata
from typing import Any

from kesoku.agent.memory_proposals import ProposalStore
from kesoku.agent.prompt import build_sys_prompt
from kesoku.config import AppConfig, KesokuConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.db.manager import DatabaseManager
from kesoku.gateway.attachment_manager import AttachmentManager
from kesoku.gateway.chatbot.app.commands import CROSS_CHANNEL_IGNORE_GLOBS, RequestHandlers
from kesoku.gateway.chatbot.app.host import Reply, RequestContext
from kesoku.gateway.chatbot.app.link import EngineLink
from kesoku.gateway.chatbot.app.media import MediaManager, probe_audio_duration_ms
from kesoku.gateway.chatbot.app.memory_policy import MemoryPolicyService
from kesoku.gateway.chatbot.app.outbox import Outbox
from kesoku.gateway.chatbot.app.pairing import PairingService
from kesoku.gateway.chatbot.app.persona import PersonaService
from kesoku.gateway.chatbot.app.proposals import ProposalService
from kesoku.gateway.chatbot.app.protocol import (
    LIMITS,
    SESSION_MEDIA_OBJECT_RE,
    ProtocolError,
    make_envelope,
    make_error,
    reply_type_for,
)
from kesoku.gateway.chatbot.app.segments import MediaJob, new_placeholder_id, to_wire_segments
from kesoku.gateway.chatbot.app.state_watcher import StateWatcher
from kesoku.gateway.chatbot.base import (
    Chatbot,
    DeliveryAbortedError,
    InboundMessageAttachment,
    parse_message_content,
)
from kesoku.gateway.chatbot.deferred_media import sanitize_error_reason
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

APP_CHANNEL_PREFIX = "app:"
MAINTENANCE_INTERVAL_SECONDS = 600.0
PAIRING_POLL_SECONDS = 1.0
TURN_EVENT_BODY_MAX_CHARS = 4000
FINALIZED_TURNS_MAX = 2048
CLIENT_MSG_CACHE_MAX = 4096
ENGINE_CONTROL_TYPES = frozenset(
    {"pair.redeem", "pair.burned", "device.registered", "engine.welcome", "engine.status", "pong", "error"}
)

_CURRENT_MEDIA: contextvars.ContextVar[MediaJob | None] = contextvars.ContextVar("app_current_media", default=None)


def _engine_version() -> str:
    """Return the installed kesoku version string."""
    try:
        return importlib_metadata.version("kesoku")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0"


class AppChatbot(Chatbot):
    """Channel adapter serving app clients through an outbound relay link."""

    supports_deferred_media = True
    _voice_inline_wait_sec = 0.0

    def __init__(
        self,
        chatbot_id: str,
        gateway: Gateway,
        app_config: AppConfig,
        *,
        link: Any | None = None,
        token: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            chatbot_id: Unique chatbot identifier (usually "app").
            gateway: Gateway broker.
            app_config: `[[chatbots.app]]` configuration.
            link: Optional pre-built relay link (tests); default builds an `EngineLink` from config.
            token: Optional relay engine token (default: read from `app_config.engine_token_env`).
        """
        self.stream_min_interval_sec = float(app_config.stream_min_interval_sec)
        super().__init__(chatbot_id, gateway)
        self.app_config = app_config
        ctx = gateway.context
        self.outbox = Outbox(ctx.sync_db, app_config.outbox_max_per_session, app_config.outbox_retention_days)
        self.link: Any | None = link
        self.media = MediaManager(lambda: self.link if self.link is not None and self.link.connected else None)
        self.state_watcher = StateWatcher(self._staging_dir_for, self._emit_state_sync)
        self.pairing = PairingService(ctx.sync_db, self.send_raw, lambda t, p: make_envelope(t, p))
        self.policy = MemoryPolicyService(ctx.sync_db, app_config, self._staging_dir_for_sync, self.broadcast)
        listeners = getattr(ctx, "memory_proposal_listeners", None)
        if not isinstance(listeners, list):
            listeners = []
            ctx.memory_proposal_listeners = listeners
        self.proposal_store = ProposalStore(ctx.sync_db, ctx.config, listeners=listeners)
        self.proposals = ProposalService(self.proposal_store, self.emit_session_event)
        if self.proposals.on_staged not in listeners:
            listeners.append(self.proposals.on_staged)
        self.persona = PersonaService(self)
        self.requests = RequestHandlers(self)
        self.attachment_manager = AttachmentManager(sessions_dir=self.sessions_dir)

        ctx.memory_policy_override_provider = self.policy.resolve_override
        ctx.memory_mode_provider = self.policy.current_mode
        gateway.register_stream_handler(chatbot_id, self._on_stream)

        self._active_turns: dict[str, dict[str, Any]] = {}
        self._finalized_turns: OrderedDict[str, None] = OrderedDict()
        self._tool_calls: dict[str, dict[str, Any]] = {}
        self._client_msgs: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._last_inbound_id: str | None = None
        self._stopping = False
        self._handlers = self._build_handlers()

        if self.link is None and app_config.relay_url:
            eff_token = token if token is not None else os.environ.get(app_config.engine_token_env, "")
            if not eff_token:
                logger.error(
                    f"App channel '{chatbot_id}': relay token env '{app_config.engine_token_env}' is not set; "
                    "the relay link stays offline."
                )
            else:
                self.link = EngineLink(
                    app_config.relay_url,
                    eff_token,
                    on_frame=self._on_frame,
                    on_connected=self._on_link_connected,
                    on_disconnected=self._on_link_disconnected,
                    hello_extra=self._hello_extra,
                    engine_version=_engine_version(),
                )

    # ------------------------------------------------------------------ AppHost surface
    @property
    def kconfig(self) -> KesokuConfig:
        """Active Kesoku configuration."""
        return self.gateway.context.config

    @property
    def sync_db(self) -> DatabaseManager:
        """Synchronous database manager."""
        return self.gateway.context.sync_db

    @property
    def awd(self) -> str:
        """Absolute agent working directory."""
        return os.path.abspath(self.kconfig.agent_working_dir or os.getcwd())

    def _abs_workspace_path(self, value: str) -> str:
        if not os.path.isabs(value):
            value = os.path.join(self.awd, value)
        return os.path.abspath(value)

    @property
    def roles_dir(self) -> str:
        """Absolute roles directory."""
        return self._abs_workspace_path(self.kconfig.workspace.roles_dir)

    @property
    def sessions_dir(self) -> str:
        """Absolute session staging root."""
        return self._abs_workspace_path(self.kconfig.workspace.sessions_dir)

    def channel_for(self, session_id: str) -> str:
        """Return the app channel id bound to a session (one channel per session)."""
        return f"{APP_CHANNEL_PREFIX}{session_id}"

    def _staging_dir_for_sync(self, session_id: str) -> str | None:
        if not session_id or "/" in session_id or session_id.startswith("."):
            return None
        return os.path.join(self.sessions_dir, session_id)

    async def _staging_dir_for(self, session_id: str) -> str | None:
        session = await self.gateway.db.get_session(session_id)
        if session is None:
            return None
        return os.path.join(self.sessions_dir, session.workspace_name)

    async def send_raw(self, envelope: dict[str, Any]) -> bool:
        """Send a pre-built envelope over the relay link (False when offline)."""
        if self.link is None:
            return False
        try:
            return bool(await self.link.send(envelope))
        except Exception as e:
            logger.warning(f"App channel send failed ({envelope.get('type')}): {e}")
            return False

    async def broadcast(self, event_type: str, payload: dict[str, Any]) -> None:
        """Send an unsequenced event to every connected device."""
        await self.send_raw(make_envelope(event_type, payload))

    async def send_to_device(
        self,
        device_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        reply_to: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Unicast an event to one device."""
        await self.send_raw(
            make_envelope(event_type, payload, reply_to=reply_to, device_id=device_id, session_id=session_id)
        )

    async def emit_session_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> int:
        """Persist a sequenced session event, broadcast it, and return its `seq`."""
        env = await self.outbox.append(session_id, event_type, payload)
        await self.send_raw(env)
        return int(env["seq"])

    async def _emit_state_sync(self, session_id: str, payload: dict[str, Any]) -> None:
        await self.emit_session_event(session_id, "state.sync", payload)

    def active_turn_snapshot(self, session_id: str) -> dict[str, Any] | None:
        """Return `{turn_id, round_id, preview_text}` of the session's in-flight streamed turn, if any."""
        turn = self._active_turns.get(session_id)
        if not turn:
            return None
        return {"turn_id": turn["turn_id"], "round_id": turn["round_id"], "preview_text": turn["text"]}

    def forget_session(self, session_id: str) -> None:
        """Drop in-memory state of a deleted session."""
        self._active_turns.pop(session_id, None)
        self.state_watcher.forget(session_id)
        self._session_locks.pop(session_id, None)

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    # ------------------------------------------------------------------ lifecycle
    def _hello_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        try:
            extra["revoked_device_ids"] = self.sync_db.list_revoked_app_device_ids()
        except Exception as e:
            logger.warning(f"Failed listing revoked devices for engine.hello: {e}")
            extra["revoked_device_ids"] = []
        if self._last_inbound_id:
            extra["last_inbound_id"] = self._last_inbound_id
        return extra

    async def _on_link_connected(self) -> None:
        self.pairing.reset_link_state()
        logger.info(f"App channel '{self.chatbot_id}' connected to relay")

    async def _on_link_disconnected(self) -> None:
        logger.warning(f"App channel '{self.chatbot_id}' lost its relay connection; reconnecting")

    def _spawn(self, coro: Awaitable[Any], name: str) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        task.set_name(name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def start(self) -> None:
        """Run the outbound subscriber, the relay link and maintenance loops until cancelled."""
        await asyncio.to_thread(self.policy.seed_defaults_sync)
        await self.outbox.prune()
        subscriber = self.spawn_subscriber_task()
        workers: list[asyncio.Task[Any]] = [subscriber]
        if self.link is not None:
            workers.append(self._spawn(self.link.run(), f"app-{self.chatbot_id}-link"))
        else:
            logger.warning(f"App channel '{self.chatbot_id}' has no relay link configured (relay_url / token)")
        workers.append(self._spawn(self._pairing_loop(), f"app-{self.chatbot_id}-pairing"))
        workers.append(self._spawn(self._maintenance_loop(), f"app-{self.chatbot_id}-maintenance"))
        try:
            await self.persona.start()
        except Exception as e:
            logger.warning(f"Persona service failed to start: {e}")
        try:
            await asyncio.gather(*workers)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Stop background work (idempotent)."""
        if self._stopping:
            return
        self._stopping = True
        try:
            await self.persona.stop()
        except Exception as e:
            logger.debug(f"Persona service stop failed: {e}")
        if self.link is not None:
            try:
                await self.link.stop()
            except Exception as e:
                logger.debug(f"Relay link stop failed: {e}")
        for task in list(self._tasks):
            task.cancel()
        Chatbot.stop(self)

    async def _pairing_loop(self) -> None:
        while not self._stopping:
            try:
                if self.link is not None and self.link.connected:
                    await self.pairing.poll_once()
            except Exception as e:
                logger.warning(f"Pairing poll failed: {e}")
            await asyncio.sleep(PAIRING_POLL_SECONDS)

    async def run_maintenance_once(self) -> None:
        """Prune the outbox, expire proposals and revoke long-inactive devices."""
        await self.outbox.prune()
        try:
            await self.proposals.expire_due()
        except Exception as e:
            logger.warning(f"Proposal expiry failed: {e}")
        try:
            await self.pairing.revoke_inactive()
        except Exception as e:
            logger.warning(f"Inactive device revocation failed: {e}")

    async def _maintenance_loop(self) -> None:
        while not self._stopping:
            await self.run_maintenance_once()
            await asyncio.sleep(MAINTENANCE_INTERVAL_SECONDS)

    # ------------------------------------------------------------------ inbound frames
    def _build_handlers(self) -> dict[str, Callable[[RequestContext, dict[str, Any]], Awaitable[Any]]]:
        r = self.requests
        p = self.persona
        return {
            "sync.replay": r.handle_sync_replay,
            "session.list": r.handle_session_list,
            "session.create": r.handle_session_create,
            "session.open": r.handle_session_open,
            "session.rename": r.handle_session_rename,
            "session.delete": r.handle_session_delete,
            "session.search": r.handle_session_search,
            "history.page": r.handle_history_page,
            "message.send": r.handle_message_send,
            "ui.pick": r.handle_ui_pick,
            "turn.stop": r.handle_turn_stop,
            "mode.start": r.handle_mode_start,
            "mode.stop": r.handle_mode_stop,
            "mode.prefs.set": r.handle_mode_prefs_set,
            "model.set": r.handle_model_set,
            "role.set": r.handle_role_set,
            "memory.list": r.handle_memory_list,
            "memory.get": r.handle_memory_get,
            "memory.put": r.handle_memory_put,
            "memory.pin": r.handle_memory_pin,
            "memory.proposal.list": self.proposals.handle_list,
            "memory.proposal.resolve": self.proposals.handle_resolve,
            "memory.policy.list": self.policy.handle_list,
            "memory.policy.set": self.policy.handle_set,
            "persona.list": p.handle_persona_list,
            "persona.get": p.handle_persona_get,
            "persona.upsert": p.handle_persona_upsert,
            "persona.delete": p.handle_persona_delete,
            "voice.preview": p.handle_voice_preview,
            "user_profile.get": p.handle_user_profile_get,
            "user_profile.put": p.handle_user_profile_put,
            "avatar.generate": p.handle_avatar_generate,
            "avatar.apply": p.handle_avatar_apply,
            "media.hydrate": r.handle_media_hydrate,
            "lexical.explain": r.handle_lexical_explain,
            "device.list": r.handle_device_list,
            "device.revoke": r.handle_device_revoke,
            "ping": r.handle_ping,
        }

    async def _on_frame(self, env: dict[str, Any]) -> None:
        """Entry point for every frame received from the relay (called sequentially by the link)."""
        event_type = str(env.get("type") or "")
        payload = env.get("payload") if isinstance(env.get("payload"), dict) else {}
        if event_type in ENGINE_CONTROL_TYPES:
            await self._handle_control(event_type, payload)
            return
        if event_type not in self._handlers:
            device_id = env.get("device_id")
            if isinstance(device_id, str) and device_id:
                await self.send_raw(
                    make_error("BAD_REQUEST", f"unknown request type '{event_type}'", reply_to=env.get("id"),
                               device_id=device_id)
                )
            else:
                logger.debug(f"Ignoring unexpected relay frame type '{event_type}'")
            return
        self._last_inbound_id = str(env.get("id") or "") or self._last_inbound_id
        self._spawn(self.dispatch_request(env), f"app-req-{event_type}")

    async def _handle_control(self, event_type: str, payload: dict[str, Any]) -> None:
        try:
            if event_type == "pair.redeem":
                await self.pairing.handle_redeem(payload)
            elif event_type == "pair.burned":
                await self.pairing.handle_burned(payload)
            elif event_type == "device.registered":
                await self.pairing.handle_registered(payload)
            elif event_type == "error":
                logger.warning(f"Relay reported error: {payload.get('code')}: {payload.get('message')}")
        except Exception as e:
            logger.warning(f"Failed handling relay control frame '{event_type}': {e}")

    async def dispatch_request(self, env: dict[str, Any]) -> None:
        """Authorize, run and reply to one client request envelope."""
        event_type = str(env.get("type") or "")
        req_id = str(env.get("id") or "")
        device_id = env.get("device_id")
        payload = env.get("payload") if isinstance(env.get("payload"), dict) else {}
        if not isinstance(device_id, str) or not device_id or len(device_id) > 64:
            logger.warning(f"Dropping '{event_type}' request without an authenticated device id")
            return
        try:
            device = await self.pairing.ensure_device(device_id)
        except Exception as e:
            logger.warning(f"Device lookup failed for {device_id}: {e}")
            device = None
        if device is None:
            await self.send_raw(make_error("UNAUTHORIZED", "device revoked", reply_to=req_id, device_id=device_id))
            await self.send_raw(make_envelope("device.revoked", {"device_ids": [device_id]}))
            return
        ctx = RequestContext(
            device_id=device_id,
            request_id=req_id,
            request_type=event_type,
            session_id=env.get("session_id") if isinstance(env.get("session_id"), str) else None,
            envelope=env,
        )
        handler = self._handlers[event_type]
        try:
            result = await handler(ctx, payload)
        except ProtocolError as e:
            await self.send_raw(make_error(e.code, e.message, reply_to=req_id, device_id=device_id))
            return
        except Exception as e:
            logger.error(f"App request '{event_type}' failed: {e}", exc_info=True)
            await self.send_raw(make_error("INTERNAL", "internal error", reply_to=req_id, device_id=device_id))
            return
        if result is None:
            return
        if isinstance(result, Reply):
            reply_type, reply_payload, sid = result.type, result.payload, result.session_id
        else:
            reply_type, reply_payload = reply_type_for(event_type), result
            sid = reply_payload.get("session_id") if isinstance(reply_payload.get("session_id"), str) else None
        await self.send_to_device(device_id, reply_type, reply_payload, reply_to=req_id, session_id=sid)

    # ------------------------------------------------------------------ inbound messages
    async def ingest_app_message(
        self,
        *,
        session_id: str,
        device_id: str,
        client_msg_id: str,
        text: str,
        attachments: list[Any],
        llm_override: dict[str, Any] | None,
        extra_metadata: dict[str, Any] | None = None,
        mode_hint: str | None = None,
    ) -> dict[str, Any]:
        """Persist a user message from a device, echo it as `message.user`, and hand it to the agent.

        Cross-channel sessions (originating on another chatbot) get an additional `app` binding without
        touching the original binding, and the message carries the generic plain-chat fallback metadata.

        Returns:
            The `message.ack` payload `{client_msg_id, message_id, seq}`.
        """
        key = (session_id, client_msg_id)
        cached = self._client_msgs.get(key)
        if cached is not None:
            return {**cached, "duplicate": True}
        async with self._session_lock(session_id):
            cached = self._client_msgs.get(key)
            if cached is not None:
                return {**cached, "duplicate": True}
            session = await self.gateway.db.get_session(session_id)
            if session is None:
                raise ProtocolError("NOT_FOUND", "session not found")
            channel_id = self.channel_for(session_id)
            bindings = await asyncio.to_thread(self.sync_db.list_channel_bindings, session_id)
            other_channels = [b[0] for b in bindings if b[0] != self.chatbot_id]
            prefs = await asyncio.to_thread(self.sync_db.get_app_session_prefs, session_id) or {}
            origin = prefs.get("origin_channel") or (other_channels[0] if other_channels else self.chatbot_id)
            cross_channel = origin != self.chatbot_id
            if cross_channel and await self._session_turn_in_progress(session_id):
                recent = await asyncio.to_thread(self.sync_db.get_messages_page, session_id, None, 10, True)
                last_user = next((m for m in reversed(recent) if m.role == MessageRole.USER), None)
                if last_user is None or last_user.chatbot_id != self.chatbot_id:
                    raise ProtocolError("TURN_BUSY", "a turn is in progress on the original channel")
            if not any(b == (self.chatbot_id, channel_id) for b in bindings):
                await asyncio.to_thread(self.sync_db.bind_channel_session_if_absent, self.chatbot_id, channel_id,
                                        session_id)
            if not prefs:
                await asyncio.to_thread(
                    self.sync_db.upsert_app_session_prefs,
                    session_id,
                    origin_channel=origin,
                    created_by_device=device_id if not cross_channel else None,
                )

            saved = await self._download_attachments(session, attachments)
            device = await asyncio.to_thread(self.sync_db.get_app_device, device_id)
            sender_name = (device or {}).get("name") or device_id
            metadata: dict[str, Any] = {
                **(extra_metadata or {}),
                "client_msg_id": client_msg_id,
                "device_id": device_id,
                "sender_name": sender_name,
                "display_text": text,
                "attachments": [a.model_dump() | {"object": o, "bytes": n} for a, o, n in saved],
            }
            eff_override = llm_override if llm_override is not None else prefs.get("llm_override")
            if eff_override:
                metadata["llm_override"] = eff_override
            if prefs.get("mode_prefs"):
                metadata["mode_prefs"] = prefs["mode_prefs"]
            if mode_hint:
                metadata["mode_hint"] = mode_hint
            if cross_channel:
                metadata["origin_channel"] = origin
                metadata.setdefault("mode_hint", "chat")
                metadata["ignore_context_globs"] = list(CROSS_CHANNEL_IGNORE_GLOBS)

            content = self._format_inbound_content_parts(text, [a for a, _, _ in saved])
            msg = Message(
                session_id=session_id,
                chatbot_id=self.chatbot_id,
                channel_id=channel_id,
                sender=device_id,
                role=MessageRole.USER,
                type=MessageType.TEXT,
                content=content,
                timestamp=time.time(),
                status=MessageStatus.PENDING_AGENT,
                metadata=metadata,
            )
            seq = await self.emit_session_event(
                session_id,
                "message.user",
                {
                    "message_id": msg.id,
                    "client_msg_id": client_msg_id,
                    "text": text,
                    "attachments": [
                        {"ref": a.path, "mime": a.mime_type, "filename": a.filename, "object": o, "bytes": n}
                        for a, o, n in saved
                    ],
                    "ts": msg.timestamp,
                    "sender": {"kind": "user", "id": device_id, "name": sender_name, "device_id": device_id},
                },
            )
            msg.metadata["app_seq"] = seq
            if not cross_channel:
                await self.gateway.db.update_session_updated_at(session_id, time.time())
                role = session.role_name or self.app_config.default_role
                await self.gateway.db.update_session_system_prompt(
                    session_id, build_sys_prompt(session=session, role=role)
                )
            await self.gateway.post(msg)
            ack = {"client_msg_id": client_msg_id, "message_id": msg.id, "seq": seq, "session_id": session_id}
            self._client_msgs[key] = ack
            while len(self._client_msgs) > CLIENT_MSG_CACHE_MAX:
                self._client_msgs.popitem(last=False)
            return ack

    def _format_inbound_content_parts(self, text: str, attachments: list[InboundMessageAttachment]) -> str:
        from kesoku.gateway.chatbot.base import InboundMessageDTO

        return self._format_inbound_content(
            InboundMessageDTO(sender_id="app", channel_id="app", text=text, attachments=attachments)
        )

    async def _download_attachments(
        self, session: Any, attachments: list[Any]
    ) -> list[tuple[InboundMessageAttachment, str, int]]:
        """Fetch client-uploaded objects into the session staging dir (validated object names and sizes)."""
        saved: list[tuple[InboundMessageAttachment, str, int]] = []
        for raw in attachments:
            if not isinstance(raw, dict):
                raise ProtocolError("BAD_REQUEST", "attachment must be an object")
            obj = raw.get("object")
            if not isinstance(obj, str) or not SESSION_MEDIA_OBJECT_RE.match(obj):
                raise ProtocolError("BAD_REQUEST", "invalid attachment object")
            if not obj.startswith(f"media/{session.id}/"):
                raise ProtocolError("BAD_REQUEST", "attachment does not belong to this session")
            declared = raw.get("bytes")
            if isinstance(declared, (int, float)) and declared > LIMITS["MESSAGE_MEDIA_MAX_BYTES"]:
                raise ProtocolError("BAD_REQUEST", "attachment too large")
            mime = raw.get("mime") if isinstance(raw.get("mime"), str) else "application/octet-stream"
            filename = raw.get("filename") if isinstance(raw.get("filename"), str) else os.path.basename(obj)
            try:
                data, _ = await self.media.fetch_object(obj, LIMITS["MESSAGE_MEDIA_MAX_BYTES"])
            except ProtocolError:
                raise
            except Exception as e:
                logger.warning(f"Failed fetching attachment {obj}: {e}")
                raise ProtocolError("BAD_REQUEST", "attachment could not be fetched") from e
            info = await self.attachment_manager.save_attachment(
                filename=self.sanitize_filename(filename[:120]) or "attachment",
                workspace_name=session.workspace_name,
                data=data,
                collision_id=obj.rsplit("/", 1)[-1][:8],
            )
            saved.append((InboundMessageAttachment(path=info["path"], mime_type=mime[:100], filename=filename[:200]),
                          obj, len(data)))
        return saved

    # ------------------------------------------------------------------ modes / roles
    async def start_mode(self, session_id: str, mode_id: str, device_id: str) -> dict[str, Any] | None:
        """Inject the configured mode launch prompt as a user message (the LLM drives the actual start)."""
        mode = self.app_config.get_mode(mode_id)
        if mode is None:
            raise ProtocolError("BAD_REQUEST", "unknown mode")
        if not mode.launch_prompt:
            current = await asyncio.to_thread(self.policy.current_mode, session_id)
            if current != mode_id and current != "chat":
                return await self.stop_mode(session_id, device_id, None)
            return None
        override = mode.llm_override.model_dump(exclude_none=True) if mode.llm_override else None
        if override:
            await asyncio.to_thread(self.sync_db.upsert_app_session_prefs, session_id, llm_override=override)
        return await self.ingest_app_message(
            session_id=session_id,
            device_id=device_id,
            client_msg_id=f"mode-{mode_id}-{int(time.time() * 1000)}",
            text=mode.launch_prompt,
            attachments=[],
            llm_override=override,
            mode_hint=mode_id,
        )

    async def stop_mode(self, session_id: str, device_id: str, instance_id: str | None) -> dict[str, Any] | None:
        """Inject the stop prompt of the session's current mode (or the configured default)."""
        current = await asyncio.to_thread(self.policy.current_mode, session_id)
        mode = self.app_config.get_mode(current)
        text = (mode.stop_prompt if mode and mode.stop_prompt else None) or self.app_config.mode_stop_prompt
        if instance_id:
            text = f"{text} ({instance_id})"
        if mode is not None and mode.llm_override is not None:
            prefs = await asyncio.to_thread(self.sync_db.get_app_session_prefs, session_id) or {}
            if prefs.get("llm_override") == mode.llm_override.model_dump(exclude_none=True):
                await asyncio.to_thread(self.sync_db.upsert_app_session_prefs, session_id, llm_override=None)
        return await self.ingest_app_message(
            session_id=session_id,
            device_id=device_id,
            client_msg_id=f"mode-stop-{int(time.time() * 1000)}",
            text=text,
            attachments=[],
            llm_override=None,
            mode_hint="chat",
        )

    async def set_session_role(self, session_id: str, role: str) -> None:
        """Rebind a session's persona in place (one session per app channel)."""
        agent = self.gateway.agent
        if agent is not None:
            await agent.stop_session_worker(session_id, immediate=True)
        session = await self.gateway.db.get_session(session_id)
        old_role = session.role_name if session else None
        consolidator = getattr(self.gateway.context, "memory_consolidator", None)
        if consolidator is not None and hasattr(consolidator, "notify_close") and old_role and old_role != role:
            try:
                await consolidator.notify_close(session_id, role=old_role, reason="role_switch")
            except Exception as e:
                logger.warning(f"Failed to notify memory consolidator on role switch: {e}")
        await self.gateway.db.set_channel_role(self.chatbot_id, self.channel_for(session_id), role)
        await asyncio.to_thread(self.sync_db.update_session_role, session_id, role)
        session = await self.gateway.db.get_session(session_id)
        if session is not None:
            await self.gateway.db.update_session_system_prompt(session_id, build_sys_prompt(session=session, role=role))

    # ------------------------------------------------------------------ streaming
    def _turn_finalized(self, turn_id: str) -> bool:
        return turn_id in self._finalized_turns

    def _mark_finalized(self, session_id: str, turn_id: str) -> None:
        self._finalized_turns[turn_id] = None
        while len(self._finalized_turns) > FINALIZED_TURNS_MAX:
            self._finalized_turns.popitem(last=False)
        active = self._active_turns.get(session_id)
        if active is not None and active["turn_id"] == turn_id:
            self._active_turns.pop(session_id, None)

    async def _on_stream(self, channel_id: str, session_id: str, turn_id: str, text: str | None) -> None:
        """Convert cumulative preview updates into `stream.delta` / `stream.clear` (unsequenced)."""
        if self._turn_finalized(turn_id):
            return
        active = self._active_turns.get(session_id)
        if active is None or active["turn_id"] != turn_id:
            active = {"turn_id": turn_id, "round_id": 0, "text": ""}
            self._active_turns[session_id] = active
        if text is None:
            await self.send_raw(
                make_envelope(
                    "stream.clear", {"turn_id": turn_id, "round_id": active["round_id"]}, session_id=session_id
                )
            )
            active["round_id"] += 1
            active["text"] = ""
            return
        prev = active["text"]
        if text.startswith(prev):
            offset, delta = len(prev), text[len(prev):]
        else:
            offset, delta = 0, text
        active["text"] = text
        if not delta and offset:
            return
        await self.send_raw(
            make_envelope(
                "stream.delta",
                {"turn_id": turn_id, "round_id": active["round_id"], "offset": offset, "delta": delta,
                 "phase": "text"},
                session_id=session_id,
            )
        )

    # ------------------------------------------------------------------ outbound messages
    def supports_intermediate_messages(self) -> bool:
        """Thoughts / tool calls are rendered as `turn.event`."""
        return True

    def get_max_text_length(self) -> int:
        """App finals are sent whole (no chunking)."""
        return 10**6

    async def handle_message(self, message: Message) -> None:
        """Process one outgoing gateway message for this channel."""
        await self.render_outgoing_message(message)

    async def render_outgoing_message(self, message: Message) -> None:
        """Map gateway messages to `turn.event` / `message.final` / `media.*` events."""
        try:
            if message.type == MessageType.SESSION_RENAME:
                await self._emit_session_rename(message)
                await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
                return
            if self.is_intermediate_message(message):
                await self._emit_turn_event(message)
                await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
                return
            if message.role == MessageRole.TOOL and message.type != MessageType.TOOL_CALL:
                await self._emit_tool_result(message)
                await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
                await self.state_watcher.scan(message.session_id)
                return
            await self._deliver_final(message)
            await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
            await self.on_message_delivered(message)
        except Exception as e:
            logger.error(f"App delivery failed for message {message.id}: {e}", exc_info=True)
            await self.on_message_delivery_failed(message, e)

    async def on_message_delivered(self, message: Message) -> None:
        """Scan staging state after each delivered final message."""
        await self.state_watcher.scan(message.session_id)

    async def _emit_session_rename(self, message: Message) -> None:
        """Announce an automatic session title as a sequenced `session.updated` event."""
        session = await self.gateway.db.get_session(message.session_id)
        payload: dict[str, Any] = {
            "session_id": message.session_id,
            "title": (message.content or "").strip() or (session.title if session else ""),
            "updated_at": float(message.timestamp),
        }
        if session is not None and session.role_name:
            payload["role"] = session.role_name
        await self.emit_session_event(message.session_id, "session.updated", payload)

    def _turn_id_of(self, message: Message) -> str:
        if message.role == MessageRole.TOOL and message.type != MessageType.TOOL_CALL and message.parent_id:
            info = self._tool_calls.get(message.parent_id)
            if info is not None:
                return str(info["turn_id"])
        return str(message.parent_id or message.id)

    async def _emit_turn_event(self, message: Message) -> None:
        meta = message.metadata or {}
        body = message.content or ""
        if message.type == MessageType.THOUGHT:
            kind, title, status = "thought", (body.strip().splitlines() or [""])[0][:80], None
        elif message.type == MessageType.TOOL_CALL:
            kind, title, status = "tool_call", str(meta.get("tool_name") or message.sender or "tool"), "running"
            self._tool_calls[message.id] = {
                "turn_id": message.parent_id or message.id,
                "title": title,
                "started": time.monotonic(),
            }
            while len(self._tool_calls) > 1024:
                self._tool_calls.pop(next(iter(self._tool_calls)))
        else:
            kind, title, status = "system", (body.strip().splitlines() or [""])[0][:80], None
        payload: dict[str, Any] = {
            "turn_id": self._turn_id_of(message),
            "kind": kind,
            "message_id": message.id,
            "title": title,
            "body": body[:TURN_EVENT_BODY_MAX_CHARS],
        }
        if status:
            payload["status"] = status
        await self.emit_session_event(message.session_id, "turn.event", payload)

    async def _emit_tool_result(self, message: Message) -> None:
        meta = message.metadata or {}
        info = self._tool_calls.get(message.parent_id or "") or {}
        payload: dict[str, Any] = {
            "turn_id": self._turn_id_of(message),
            "kind": "tool_result",
            "message_id": message.id,
            "tool_call_message_id": message.parent_id,
            "title": str(meta.get("tool_name") or info.get("title") or message.sender or "tool"),
            "body": (message.content or "")[:TURN_EVENT_BODY_MAX_CHARS],
            "status": "error" if meta.get("tool_error") else "ok",
        }
        if info.get("started") is not None:
            payload["elapsed_ms"] = int((time.monotonic() - float(info["started"])) * 1000)
        await self.emit_session_event(message.session_id, "turn.event", payload)

    async def _deliver_final(self, message: Message) -> None:
        segments = parse_message_content(message.content or "", rich_blocks=True)
        wire, jobs = to_wire_segments(segments, message_id=message.id, live=True)
        turn_id = str(message.parent_id or message.id)
        session = await self.gateway.db.get_session(message.session_id)
        role_id = (session.role_name if session else None) or self.app_config.default_role
        sender: dict[str, Any] = {"kind": "assistant", "name": message.sender, "role_id": role_id}
        if message.sender != "Notification":
            display_name = await self._role_display_name(role_id)
            if display_name:
                sender["display_name"] = display_name
        payload = {
            "message_id": message.id,
            "turn_id": turn_id,
            "segments": wire,
            "raw_markdown": message.content or "",
            "ts": float(message.timestamp),
            "sender": sender,
        }
        await self.emit_session_event(message.session_id, "message.final", payload)
        if message.sender != "Notification":
            self._mark_finalized(message.session_id, turn_id)
        for job in jobs:
            await self._deliver_media_job(message, job)

    async def _role_display_name(self, role_id: str | None) -> str | None:
        """Return the persona display name for `role_id` (None when unknown or unreadable)."""
        if not role_id:
            return None
        try:
            summary = await asyncio.to_thread(self.persona.registry.get, role_id)
        except Exception as e:  # noqa: BLE001 - a broken role.json must never block delivery
            logger.debug(f"Persona lookup failed for role {role_id}: {e}")
            return None
        name = (summary or {}).get("display_name")
        return name if isinstance(name, str) and name.strip() else None

    async def _deliver_media_job(self, message: Message, job: MediaJob) -> None:
        token = _CURRENT_MEDIA.set(job)
        try:
            await self._attempt_send_attachment({"type": job.kind, "path": job.ref}, message)
        except DeliveryAbortedError as e:
            await self._emit_media_failed(message, job.placeholder_id, str(e))
        except Exception as e:
            logger.warning(f"Media delivery failed for {job.ref}: {e}")
            await self._emit_media_failed(message, job.placeholder_id, str(e))
        finally:
            _CURRENT_MEDIA.reset(token)

    async def _emit_media_failed(self, message: Message, placeholder_id: str, reason: str) -> None:
        await self.emit_session_event(
            message.session_id,
            "media.failed",
            {
                "placeholder_id": placeholder_id,
                "message_id": message.id,
                "reason": sanitize_error_reason(reason, default="附件发送失败"),
            },
        )

    async def _upload_and_announce(
        self, message: Message, file_path: str, kind: str, placeholder_id: str, ref: str | None
    ) -> None:
        uploaded = await self.media.upload_file(file_path, purpose="message", session_id=message.session_id)
        payload: dict[str, Any] = {
            "placeholder_id": placeholder_id,
            "message_id": message.id,
            "kind": kind,
            "object": uploaded["object"],
            "url": uploaded["url"],
            "mime": uploaded.get("mime"),
            "bytes": uploaded.get("bytes"),
            "expires_at": uploaded.get("expires_at"),
            "name": os.path.basename(file_path),
        }
        if ref:
            payload["ref"] = ref
        if kind == "voice":
            duration_ms = await asyncio.to_thread(probe_audio_duration_ms, file_path)
            if duration_ms is not None:
                payload["duration_ms"] = duration_ms
        await self.emit_session_event(message.session_id, "media.ready", payload)

    async def send_file_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Upload a ready file and emit `media.ready`."""
        job = _CURRENT_MEDIA.get()
        if not await asyncio.to_thread(os.path.isfile, file_path):
            raise FileNotFoundError(file_path)
        await self._upload_and_announce(
            message, file_path, "file", job.placeholder_id if job else new_placeholder_id(), job.ref if job else None
        )

    async def send_voice_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Upload a ready voice clip and emit `media.ready{kind:"voice"}` (clients play only on tap)."""
        job = _CURRENT_MEDIA.get()
        if not await asyncio.to_thread(os.path.isfile, file_path):
            raise FileNotFoundError(file_path)
        await self._upload_and_announce(
            message, file_path, "voice", job.placeholder_id if job else new_placeholder_id(), job.ref if job else None
        )

    async def send_deferred_placeholder(
        self, channel_id: str, segment: dict[str, Any], message: Message, marker: dict[str, Any]
    ) -> Any:
        """Announce a pending deferred attachment as `media.pending`; the handle carries the placeholder id."""
        job = _CURRENT_MEDIA.get()
        placeholder_id = job.placeholder_id if job else new_placeholder_id()
        kind = "voice" if segment.get("type") == "voice" else "file"
        await self.emit_session_event(
            message.session_id,
            "media.pending",
            {
                "placeholder_id": placeholder_id,
                "message_id": message.id,
                "kind": kind,
                "label": os.path.basename(str(segment.get("path") or "")),
            },
        )
        return {"placeholder_id": placeholder_id, "kind": kind, "ref": str(segment.get("path") or "")}

    def serialize_deferred_handle(self, handle: Any) -> dict[str, Any] | None:
        """Persist the placeholder handle for restart recovery."""
        return dict(handle) if isinstance(handle, dict) else None

    async def restore_deferred_handle(self, raw_handle: dict[str, Any] | None) -> Any:
        """Restore a persisted placeholder handle."""
        return dict(raw_handle) if isinstance(raw_handle, dict) else None

    async def deliver_deferred_attachment(
        self, channel_id: str, file_path: str, message: Message, handle: Any, kind: str
    ) -> None:
        """Upload a completed deferred attachment and emit `media.ready` for its placeholder."""
        placeholder_id = (handle or {}).get("placeholder_id") if isinstance(handle, dict) else None
        await self._upload_and_announce(
            message,
            file_path,
            "voice" if kind == "voice" else "file",
            placeholder_id or new_placeholder_id(),
            (handle or {}).get("ref") if isinstance(handle, dict) else None,
        )

    async def fail_deferred_attachment(
        self, handle: Any, reason: str, message: Message, segment: dict[str, Any]
    ) -> None:
        """Emit `media.failed` for a deferred attachment."""
        placeholder_id = (handle or {}).get("placeholder_id") if isinstance(handle, dict) else None
        if placeholder_id is None:
            job = _CURRENT_MEDIA.get()
            placeholder_id = job.placeholder_id if job else new_placeholder_id()
        await self._emit_media_failed(message, placeholder_id, reason)

    async def send_text_chunks(self, channel_id: str, chunks: list[str], message: Message) -> None:
        """Out-of-band notices from the shared delivery pipeline become `turn.event{kind:"system"}`."""
        text = "\n".join(c for c in chunks if c and c.strip())
        if not text:
            return
        await self.emit_session_event(
            message.session_id,
            "turn.event",
            {
                "turn_id": str(message.parent_id or message.id),
                "kind": "system",
                "message_id": message.id,
                "title": text.splitlines()[0][:80],
                "body": text[:TURN_EVENT_BODY_MAX_CHARS],
            },
        )

    async def send_question_segment(self, *args: Any, **kwargs: Any) -> None:
        """Questions are part of `message.final` segments; nothing extra to send."""
        return None

    async def is_dm_channel(self, channel_id: str) -> bool:
        """App channels are private."""
        return True

"""End-to-end tests of the app channel adapter's wire protocol (fake relay link, real SQLite + gateway)."""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

import pytest

import kesoku.config
import kesoku.gateway.chatbot.app.commands as commands_mod
import kesoku.gateway.chatbot.app.media as media_mod
from kesoku.agent.turn_executor import TurnExecutor
from kesoku.config import AppConfig, AppModeConfig, KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message
from kesoku.gateway.chatbot.app.adapter import TRANSLATOR_PURE_AUDIO_PROMPT, AppChatbot
from kesoku.gateway.chatbot.app.protocol import (
    LIMITS,
    ProtocolError,
    decode_binary_frame,
    encode_binary_frame,
    make_envelope,
    new_id,
    parse_envelope,
    reply_type_for,
)
from kesoku.gateway.chatbot.app.segments import MediaJob, message_to_record, user_attachments, user_display_text
from kesoku.gateway.chatbot.app.state_watcher import MAX_STATE_FILES, NOTES_TEXT_MAX_BYTES
from kesoku.gateway.chatbot.base import DeliveryAbortedError
from kesoku.gateway.gateway import Gateway


class FakeLink:
    """In-memory stand-in for `EngineLink` + relay media endpoints."""

    def __init__(self) -> None:
        self.connected = True
        self.sent: list[dict[str, Any]] = []
        self.objects: dict[str, dict[str, Any]] = {}

    async def send(self, envelope: dict[str, Any]) -> bool:
        self.sent.append(envelope)
        return True

    async def request(self, envelope: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        payload = envelope["payload"]
        if envelope["type"] == "media.sign":
            obj = payload["object"]
            info = self.objects.get(obj)
            if info is None:
                raise ProtocolError("NOT_FOUND", "no such object")
            return {
                "type": "media.signed",
                "payload": {
                    "object": obj,
                    "url": f"https://store.invalid/{obj}?sig=1",
                    "expires_at": time.time() + 900,
                    "issued_device_id": info.get("device"),
                    "issued_at": info.get("issued_at"),
                },
            }
        if envelope["type"] == "media.upload.ticket":
            obj = f"media/{payload.get('session_id') or 'asset'}/{uuid.uuid4()}.bin"
            return {
                "type": "media.ticket.ok",
                "payload": {
                    "object": obj,
                    "put_url": f"https://store.invalid/put/{obj}",
                    "expires_at": time.time() + 600,
                    "max_bytes": payload["bytes"],
                    "get_url": f"https://store.invalid/{obj}",
                    "get_expires_at": time.time() + 900,
                },
            }
        raise AssertionError(f"unexpected request {envelope['type']}")

    async def request_binary(self, frame: bytes, request_id: str, timeout: float | None = None) -> dict[str, Any]:
        header, data = decode_binary_frame(frame)
        ext = os.path.splitext(header.get("filename") or "x.bin")[1].lstrip(".") or "bin"
        obj = f"media/{header.get('session_id') or 'asset'}/{uuid.uuid4()}.{ext}"
        self.objects[obj] = {"data": data}
        return {
            "type": "media.put.ok",
            "reply_to": request_id,
            "payload": {
                "object": obj,
                "url": f"https://store.invalid/{obj}",
                "expires_at": time.time() + 900,
                "mime": header["mime"],
                "bytes": len(data),
            },
        }

    async def run(self) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.connected = False

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.sent if e.get("type") == event_type]


@pytest.fixture
async def env(tmp_path, monkeypatch):
    """Build a gateway + AppChatbot wired to a FakeLink on temp directories."""
    db_path = str(tmp_path / "kesoku.db")
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(
            db_path=db_path,
            roles_dir=str(tmp_path / "roles"),
            sessions_dir=str(tmp_path / "sessions"),
            skills_dir=str(tmp_path / "skills"),
        )
    )
    cfg.agent_working_dir = str(tmp_path)
    app_cfg = AppConfig(
        enabled=True,
        default_role="helper",
        allowed_models=["model-a", "model-b"],
        modes=[
            AppModeConfig(id="chat", title="Chat"),
            AppModeConfig(
                id="quiz",
                title="Quiz",
                launch_prompt="start the quiz",
                llm_override={"llm": "gemini", "model_name": "model-b", "thinking_level": "low", "stream": True},
                prefs_schema={"level": ["easy", "hard"], "focus": "string"},
            ),
            AppModeConfig(id="story", title="Story", launch_prompt="start the story"),
        ],
    )
    cfg.chatbots.app = [app_cfg]
    monkeypatch.setattr(kesoku.config, "_global_config", cfg)
    DatabaseManager(db_path).init_tables()
    for role in ("helper", "default", "other_role"):
        (tmp_path / "roles" / role).mkdir(parents=True)
        (tmp_path / "roles" / role / "intro.md").write_text(f"# {role}\nA test role.\n", encoding="utf-8")
    ctx = KesokuContext(config=cfg)
    gw = Gateway(context=ctx)
    link = FakeLink()
    bot = AppChatbot("app", gw, app_cfg, link=link)
    await asyncio.to_thread(bot.policy.seed_defaults_sync)
    yield {"bot": bot, "link": link, "gw": gw, "cfg": cfg, "ctx": ctx, "tmp": tmp_path}


async def rpc(e: dict[str, Any], event_type: str, payload: dict[str, Any], device: str = "dev-1") -> dict[str, Any]:
    """Dispatch a client request and return its reply envelope."""
    req = make_envelope(event_type, payload, device_id=device, event_id=new_id("req"))
    await e["bot"].dispatch_request(req)
    replies = [s for s in e["link"].sent if s.get("reply_to") == req["id"]]
    assert replies, f"no reply for {event_type}"
    return replies[-1]


async def create_session(e: dict[str, Any], **payload: Any) -> str:
    reply = await rpc(e, "session.create", payload)
    assert reply["type"] == "session.snapshot", reply
    return reply["payload"]["session_id"]


# ---------------------------------------------------------------------------------------------- protocol


def test_protocol_envelope_and_binary_frames() -> None:
    env = make_envelope("ping", {}, device_id="d1", event_id="req_1")
    assert parse_envelope(json.dumps(env))["device_id"] == "d1"
    with pytest.raises(ProtocolError):
        parse_envelope('{"v": 2, "type": "x", "id": "1", "payload": {}}')
    with pytest.raises(ProtocolError):
        parse_envelope('{"v": 1, "type": "x", "id": "1", "payload": []}')
    with pytest.raises(ProtocolError):
        parse_envelope("not json")
    header, data = decode_binary_frame(encode_binary_frame({"type": "media.put", "id": "r"}, b"\x00\x01"))
    assert header["type"] == "media.put" and data == b"\x00\x01"
    with pytest.raises(ProtocolError):
        encode_binary_frame({"type": "media.put"}, b"x" * (1024 * 1024))
    assert reply_type_for("session.open") == "session.snapshot"
    assert reply_type_for("message.send") == "message.ack"
    assert reply_type_for("memory.policy.list") == "memory.policy.list.result"
    assert reply_type_for("mode.defaults.get") == "mode.defaults.get.result"
    assert reply_type_for("mode.defaults.set") == "mode.defaults.set.result"
    assert reply_type_for("character.list") == "character.list.result"
    assert reply_type_for("roster.update") == "roster.update.result"
    assert LIMITS["MODE_PREFS_MAX_BYTES"] == 4096


def test_stream_interval_registration(env) -> None:
    gw = env["gw"]
    assert gw.get_stream_min_interval("app") == pytest.approx(0.05)
    assert gw.get_stream_min_interval("discord") == pytest.approx(1.5)


# ---------------------------------------------------------------------------------------------- sessions


async def test_session_create_open_list_rename_and_reply_routing(env) -> None:
    sid = await create_session(env, title="First", role="other_role")
    reply = await rpc(env, "session.open", {"session_id": sid})
    snap = reply["payload"]
    assert reply["device_id"] == "dev-1" and reply["session_id"] == sid
    assert snap["session"]["role"] == "other_role"
    assert snap["session"]["mode_id"] == "chat"
    assert snap["session"]["app_bound"] is True
    assert snap["origin_channel"] == "app"
    assert snap["head_seq"] >= 1  # session.updated from create

    upd = [s for s in env["link"].of_type("session.updated") if s["session_id"] == sid]
    assert upd and upd[0]["seq"] == 1

    for i in range(3):
        await create_session(env, title=f"S{i}")
    page1 = (await rpc(env, "session.list", {"limit": 2}))["payload"]
    assert len(page1["sessions"]) == 2 and page1["next_cursor"]
    page2 = (await rpc(env, "session.list", {"limit": 10, "cursor": page1["next_cursor"]}))["payload"]
    ids = {s["session_id"] for s in page1["sessions"]} | {s["session_id"] for s in page2["sessions"]}
    assert len(ids) == 4

    renamed = await rpc(env, "session.rename", {"session_id": sid, "title": "Renamed"})
    assert renamed["type"] == "session.rename.result"
    assert (await env["gw"].db.get_session(sid)).title == "Renamed"

    missing = await rpc(env, "session.open", {"session_id": "deadbeef"})
    assert missing["type"] == "error" and missing["payload"]["code"] == "NOT_FOUND"
    bad_role = await rpc(env, "session.create", {"role": "../etc"})
    assert bad_role["payload"]["code"] == "BAD_REQUEST"


async def test_unknown_device_auto_registered_and_revoked_device_rejected(env) -> None:
    await rpc(env, "device.list", {}, device="dev-new")
    db = env["bot"].sync_db
    assert db.get_app_device("dev-new") is not None
    listed = (await rpc(env, "device.list", {}, device="dev-new"))["payload"]["devices"]
    assert any(d["device_id"] == "dev-new" and d["current"] for d in listed)

    revoke = await rpc(env, "device.revoke", {"device_id": "dev-new"}, device="dev-1")
    assert revoke["payload"]["revoked"] is True
    assert env["link"].of_type("device.revoked")[-1]["payload"]["device_ids"] == ["dev-new"]
    listed_active = (await rpc(env, "device.list", {}, device="dev-1"))["payload"]["devices"]
    assert all(d["device_id"] != "dev-new" for d in listed_active)
    listed_with_revoked = (
        await rpc(env, "device.list", {"include_revoked": True}, device="dev-1")
    )["payload"]["devices"]
    assert any(d["device_id"] == "dev-new" and d["revoked_at"] is not None for d in listed_with_revoked)
    denied = await rpc(env, "device.list", {}, device="dev-new")
    assert denied["type"] == "error" and denied["payload"]["code"] == "UNAUTHORIZED"


async def test_unknown_request_type_and_handler_errors(env) -> None:
    req = make_envelope("nope.nothing", {}, device_id="dev-1", event_id=new_id("req"))
    await env["bot"]._on_frame(req)
    err = [s for s in env["link"].sent if s.get("reply_to") == req["id"]]
    assert err and err[0]["payload"]["code"] == "BAD_REQUEST"


# ---------------------------------------------------------------------------------------------- messages


async def test_message_send_ack_echo_idempotency_and_llm_override(env) -> None:
    sid = await create_session(env)
    posted: list[Message] = []
    orig_post = env["gw"].post

    async def capture(msg: Message) -> Message:
        posted.append(msg)
        return await orig_post(msg)

    env["gw"].post = capture
    ack = await rpc(
        env,
        "message.send",
        {"session_id": sid, "client_msg_id": "c1", "text": "hello", "llm_override": {"model_name": "model-a"}},
    )
    assert ack["type"] == "message.ack"
    echo = [s for s in env["link"].of_type("message.user") if s["session_id"] == sid][-1]
    assert echo["seq"] == ack["payload"]["seq"] == 2
    assert echo["payload"]["message_id"] == ack["payload"]["message_id"]
    assert posted[-1].metadata["llm_override"] == {"model_name": "model-a"}
    assert posted[-1].metadata["app_seq"] == 2
    assert posted[-1].channel_id == f"app:{sid}" and posted[-1].status == MessageStatus.PENDING_AGENT

    again = await rpc(env, "message.send", {"session_id": sid, "client_msg_id": "c1", "text": "hello"})
    assert again["payload"]["message_id"] == ack["payload"]["message_id"]
    assert len(posted) == 1

    bad = await rpc(
        env,
        "message.send",
        {"session_id": sid, "client_msg_id": "c2", "text": "x", "llm_override": {"model_name": "z"}},
    )
    assert bad["payload"]["code"] == "BAD_REQUEST"
    empty = await rpc(env, "message.send", {"session_id": sid, "client_msg_id": "c3", "text": "  "})
    assert empty["payload"]["code"] == "BAD_REQUEST"


async def test_message_send_attachment_download_and_validation(env) -> None:
    sid = await create_session(env)
    bot, link = env["bot"], env["link"]
    obj = f"media/{sid}/{uuid.uuid4()}.png"
    link.objects[obj] = {"data": b"\x89PNG fake", "device": "dev-1", "issued_at": time.time()}

    async def fake_get(url: str, max_bytes: int) -> bytes:
        key = url.split("store.invalid/", 1)[1].split("?", 1)[0]
        return link.objects[key]["data"]

    bot.media._http_get = fake_get
    ack = await rpc(
        env,
        "message.send",
        {
            "session_id": sid,
            "client_msg_id": "att-1",
            "text": "look",
            "attachments": [{"object": obj, "mime": "image/png", "filename": "../../evil name.png", "bytes": 9}],
        },
    )
    assert ack["type"] == "message.ack", ack
    msg = await env["gw"].db.get_message(ack["payload"]["message_id"])
    saved = msg.metadata["attachments"][0]["path"]
    assert os.path.realpath(saved).startswith(os.path.realpath(str(env["tmp"] / "sessions" / sid)) + os.sep)
    assert open(saved, "rb").read() == b"\x89PNG fake"

    foreign = await rpc(
        env,
        "message.send",
        {"session_id": sid, "client_msg_id": "att-2", "text": "x",
         "attachments": [{"object": f"media/abcdef12/{uuid.uuid4()}.png", "mime": "image/png", "filename": "a.png",
                          "bytes": 1}]},
    )
    assert foreign["payload"]["code"] == "BAD_REQUEST"
    traversal = await rpc(
        env,
        "message.send",
        {"session_id": sid, "client_msg_id": "att-3", "text": "x",
         "attachments": [{"object": "media/../../etc/passwd", "mime": "text/plain", "filename": "p", "bytes": 1}]},
    )
    assert traversal["payload"]["code"] == "BAD_REQUEST"


async def test_cross_channel_plain_chat_fallback(env) -> None:
    gw = env["gw"]
    sess = await gw.create_session(
        title="From another channel", chatbot_id="discord", channel_id="chan-1", role="helper"
    )
    posted: list[Message] = []
    orig_post = gw.post

    async def capture(msg: Message) -> Message:
        posted.append(msg)
        return await orig_post(msg)

    gw.post = capture
    listed_app_only = (await rpc(env, "session.list", {}))["payload"]["sessions"]
    assert all(s["session_id"] != sess.id for s in listed_app_only)
    listed_all = (await rpc(env, "session.list", {"all_channels": True}))["payload"]["sessions"]
    entry = next(s for s in listed_all if s["session_id"] == sess.id)
    assert entry["origin_channel"] == "discord" and entry["app_bound"] is False

    ack = await rpc(env, "message.send", {"session_id": sess.id, "client_msg_id": "x1", "text": "just chatting"})
    assert ack["type"] == "message.ack", ack
    meta = posted[-1].metadata
    assert meta["ignore_context_globs"] == ["game-*"]
    assert meta["mode_hint"] == "chat" and meta["origin_channel"] == "discord"
    bindings = env["bot"].sync_db.list_channel_bindings(sess.id)
    assert ("discord", "chan-1") in bindings and ("app", f"app:{sess.id}") in bindings
    assert (await gw.db.get_session_by_channel("discord", "chan-1")).id == sess.id

    deleted = await rpc(env, "session.delete", {"session_id": sess.id})
    assert deleted["payload"]["code"] == "BAD_REQUEST"


async def test_cross_channel_turn_busy(env) -> None:
    gw = env["gw"]
    sess = await gw.create_session(title="busy", chatbot_id="discord", channel_id="chan-2", role="helper")
    await gw.db.save_message(
        Message(session_id=sess.id, chatbot_id="discord", channel_id="chan-2", sender="u", role=MessageRole.USER,
                type=MessageType.TEXT, content="playing", status=MessageStatus.PENDING_AGENT)
    )
    busy = await rpc(env, "message.send", {"session_id": sess.id, "client_msg_id": "b1", "text": "hi"})
    assert busy["payload"]["code"] == "TURN_BUSY"


# ---------------------------------------------------------------------------------------------- outbox replay


async def test_sync_replay_done_and_reset(env) -> None:
    bot, link = env["bot"], env["link"]
    sid = await create_session(env)
    for i in range(3):
        await bot.emit_session_event(sid, "turn.event", {"turn_id": "t", "kind": "system", "message_id": f"m{i}",
                                                         "title": str(i)})
    head = await bot.outbox.head(sid)
    assert head == 4
    req = make_envelope("sync.replay", {"session_id": sid, "after_seq": 1, "limit": 2}, device_id="dev-1",
                        event_id=new_id("req"))
    originals = {e["seq"]: e["id"] for e in link.sent if e.get("session_id") == sid and "seq" in e}
    link.sent.clear()
    await bot.dispatch_request(req)
    replayed = [e for e in link.sent if e.get("type") == "turn.event"]
    assert [e["seq"] for e in replayed] == [2, 3]
    assert all(e["device_id"] == "dev-1" for e in replayed)
    assert [e["id"] for e in replayed] == [originals[2], originals[3]]
    done = link.of_type("sync.done")[-1]
    assert done["reply_to"] == req["id"]
    assert done["payload"] == {"session_id": sid, "replayed_through_seq": 3, "head_seq": 4, "has_more": True}

    bot.sync_db.prune_app_outbox(1, time.time() + 10)
    assert bot.sync_db.get_app_outbox_bounds(sid) == (4, 4)
    req2 = make_envelope("sync.replay", {"session_id": sid, "after_seq": 1}, device_id="dev-1", event_id=new_id("req"))
    await bot.dispatch_request(req2)
    reset = link.of_type("sync.reset")[-1]
    assert reset["payload"] == {"session_id": sid, "reason": "OUTBOX_PRUNED", "head_seq": 4}
    nxt = await bot.emit_session_event(sid, "turn.event", {"turn_id": "t", "kind": "system", "message_id": "m",
                                                           "title": "x"})
    assert nxt == 5


# ---------------------------------------------------------------------------------------------- streaming & finals


async def test_stream_delta_clear_offsets_and_late_frames_dropped(env) -> None:
    bot, link = env["bot"], env["link"]
    sid = "abcd1234"
    await bot._on_stream("app:x", sid, "turn-1", "Hel")
    await bot._on_stream("app:x", sid, "turn-1", "Hello 世界")
    deltas = link.of_type("stream.delta")
    assert [d["payload"]["offset"] for d in deltas] == [0, 3]
    assert deltas[1]["payload"]["delta"] == "lo 世界"
    assert "seq" not in deltas[0]
    assert bot.active_turn_snapshot(sid) == {"turn_id": "turn-1", "round_id": 0, "preview_text": "Hello 世界"}
    await bot._on_stream("app:x", sid, "turn-1", None)
    clear = link.of_type("stream.clear")[-1]["payload"]
    assert clear == {"turn_id": "turn-1", "round_id": 0}
    await bot._on_stream("app:x", sid, "turn-1", "Round two")
    assert link.of_type("stream.delta")[-1]["payload"]["round_id"] == 1
    assert link.of_type("stream.delta")[-1]["payload"]["offset"] == 0
    bot._mark_finalized(sid, "turn-1")
    n = len(link.sent)
    await bot._on_stream("app:x", sid, "turn-1", "Round two more")
    assert len(link.sent) == n
    assert bot.active_turn_snapshot(sid) is None


async def test_final_message_segments_media_and_turn_events(env) -> None:
    bot, link, gw = env["bot"], env["link"], env["gw"]
    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "scene.png").write_bytes(b"\x89PNG data")
    # A failed deferred job (error sidecar) passes the pre-flight check and surfaces as `media.failed`.
    (staging / "missing_file.png.error.json").write_text(json.dumps({"error": "render failed"}), encoding="utf-8")
    user = Message(session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", sender="dev-1", role=MessageRole.USER,
                   content="hi", status=MessageStatus.PROCESSED)
    await gw.db.save_message(user)

    tc = Message(session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", sender="Kesoku", role=MessageRole.TOOL,
                 type=MessageType.TOOL_CALL, content="call", parent_id=user.id,
                 metadata={"tool_name": "run_shell", "tool_arguments": {}}, status=MessageStatus.RESPONDED)
    await gw.db.save_message(tc)
    await bot.render_outgoing_message(tc)
    tr = Message(session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", sender="run_shell", role=MessageRole.TOOL,
                 type=MessageType.TOOL_RESULT, content="ok output", parent_id=tc.id, status=MessageStatus.PENDING)
    await gw.db.save_message(tr)
    await bot.render_outgoing_message(tr)
    events = [e for e in link.of_type("turn.event") if e["session_id"] == sid]
    assert [e["payload"]["kind"] for e in events] == ["tool_call", "tool_result"]
    assert all(e["payload"]["turn_id"] == user.id for e in events)
    assert events[1]["payload"]["status"] == "ok"

    content = (
        "Intro text\n"
        "```bilingual\nspeaker: Hana\nja: 今日(きょう)は晴(は)れ\nzh: 今天晴天\n```\n"
        "```card\nkind: info_card\ntitle: Weather\nimage: \"[file: scene.png]\"\n```\n"
        f"[file: {staging / 'scene.png'}]\n"
        "[file: missing_file.png]\n"
        "```ui\ntype: select\nprompt: Next?\noptions:\n  - A\n  - B\n```"
    )
    final = Message(session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", sender="Kesoku",
                    role=MessageRole.ASSISTANT, content=content, parent_id=user.id, status=MessageStatus.PENDING)
    await gw.db.save_message(final)
    await bot.render_outgoing_message(final)

    fin = [e for e in link.of_type("message.final") if e["session_id"] == sid][-1]
    segs = fin["payload"]["segments"]
    assert fin["payload"]["turn_id"] == user.id
    assert [s["type"] for s in segs] == ["text", "bilingual", "card", "file", "file", "question"]
    assert segs[1]["ruby_parts"][0] == {"base": "今日", "rt": "きょう"}
    assert segs[2]["kind"] == "info_card" and "image" in segs[2]["media"]
    assert segs[5]["question_message_id"] == final.id
    ready = [e for e in link.of_type("media.ready") if e["session_id"] == sid]
    ready_ids = {e["payload"]["placeholder_id"] for e in ready}
    assert segs[3]["placeholder_id"] in ready_ids
    assert segs[2]["media"]["image"] in ready_ids
    failed = [e for e in link.of_type("media.failed") if e["session_id"] == sid]
    assert segs[4]["placeholder_id"] in {e["payload"]["placeholder_id"] for e in failed}
    seqs = [e["seq"] for e in link.sent if e.get("session_id") == sid and "seq" in e]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))
    assert (await gw.db.get_message(final.id)).status == MessageStatus.DELIVERED

    hist = (await rpc(env, "history.page", {"session_id": sid}))["payload"]["messages"]
    assistant = [m for m in hist if m["role"] == "assistant"][-1]
    assert assistant["segments"][3]["ref"].endswith("scene.png") and "placeholder_id" not in assistant["segments"][3]


async def test_media_hydrate_jail(env) -> None:
    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "voice.wav").write_bytes(b"RIFF....WAVE")
    (env["tmp"] / "secret.txt").write_text("secret", encoding="utf-8")
    ok = await rpc(env, "media.hydrate", {"session_id": sid, "message_id": "m1", "ref": "voice.wav"})
    assert ok["type"] == "media.ready" and ok["payload"]["object"].startswith(f"media/{sid}/")
    for ref in ("../../secret.txt", str(env["tmp"] / "secret.txt"), "[file: ../../secret.txt]"):
        bad = await rpc(env, "media.hydrate", {"session_id": sid, "ref": ref})
        assert bad["type"] == "error" and bad["payload"]["code"] == "NOT_FOUND", ref
    os.symlink(env["tmp"] / "secret.txt", staging / "link.txt")
    sym = await rpc(env, "media.hydrate", {"session_id": sid, "ref": "link.txt"})
    assert sym["payload"]["code"] == "NOT_FOUND"
    no_session = await rpc(env, "media.hydrate", {"ref": "voice.wav"})
    assert no_session["payload"]["code"] == "BAD_REQUEST"


async def test_state_sync_watcher(env) -> None:
    bot, link = env["bot"], env["link"]
    sid = await create_session(env)
    game = env["tmp"] / "sessions" / sid / "game"
    ctxd = env["tmp"] / "sessions" / sid / "context"
    game.mkdir(parents=True, exist_ok=True)
    ctxd.mkdir(parents=True, exist_ok=True)
    (game / "party.json").write_text(json.dumps({"hp": 10}), encoding="utf-8")
    (ctxd / "20-scene.md").write_text("x" * 9000, encoding="utf-8")
    payload = await bot.state_watcher.scan(sid)
    assert payload["files"]["game/party.json"] == {"hp": 10}
    assert len(payload["files"]["context/20-scene.md"]) == 8192
    assert await bot.state_watcher.scan(sid) is None
    (game / "broken.json").write_text("{not json", encoding="utf-8")
    assert await bot.state_watcher.scan(sid) is None
    os.remove(game / "party.json")
    removal = await bot.state_watcher.scan(sid)
    assert removal["removed"] == ["game/party.json"] and removal["staging_rev"] == 2
    syncs = [e for e in link.of_type("state.sync") if e["session_id"] == sid]
    assert len(syncs) == 2 and all("seq" in e for e in syncs)
    snap = (await rpc(env, "session.open", {"session_id": sid}))["payload"]
    assert "context/20-scene.md" in snap["state_files"]


# ---------------------------------------------------------------------------------------------- modes & prefs


async def test_mode_start_prefs_model_and_role(env) -> None:
    bot, gw = env["bot"], env["gw"]
    sid = await create_session(env)
    posted: list[Message] = []
    orig_post = gw.post

    async def capture(msg: Message) -> Message:
        posted.append(msg)
        return await orig_post(msg)

    gw.post = capture
    started = await rpc(env, "mode.start", {"session_id": sid, "mode_id": "quiz"})
    assert started["type"] == "mode.start.result"
    assert posted[-1].content == "start the quiz"
    assert posted[-1].metadata["mode_hint"] == "quiz"
    assert posted[-1].metadata["llm_override"]["model_name"] == "model-b"
    unknown = await rpc(env, "mode.start", {"session_id": sid, "mode_id": "nope"})
    assert unknown["payload"]["code"] == "BAD_REQUEST"

    prefs = await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": {"level": "hard", "focus": "verbs"}})
    assert prefs["payload"]["prefs"] == {"level": "hard", "focus": "verbs"}
    bad = await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": {"level": "insane"}})
    assert bad["payload"]["code"] == "BAD_REQUEST"
    bad_key = await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": {"unknown": "x"}})
    assert bad_key["payload"]["code"] == "BAD_REQUEST"
    await rpc(env, "message.send", {"session_id": sid, "client_msg_id": "p1", "text": "q"})
    assert posted[-1].metadata["mode_prefs"] == {"level": "hard", "focus": "verbs"}

    model = await rpc(env, "model.set", {"session_id": sid, "llm_override": {"model_name": "model-a"}})
    assert model["payload"]["llm_override"] == {"model_name": "model-a"}
    rejected = await rpc(env, "model.set", {"session_id": sid, "llm_override": {"model_name": "nope"}})
    assert rejected["payload"]["code"] == "BAD_REQUEST"

    role = await rpc(env, "role.set", {"session_id": sid, "role": "other_role"})
    assert role["payload"]["role"] == "other_role"
    assert (await gw.db.get_session(sid)).role_name == "other_role"
    assert (await gw.db.get_session_by_channel("app", f"app:{sid}")).id == sid


# ---------------------------------------------------------------------------------------------- memory policy


def _write_games(tmp, sid: str, instances: Any) -> None:
    game = tmp / "sessions" / sid / "game"
    game.mkdir(parents=True, exist_ok=True)
    (game / "games.json").write_text(json.dumps(instances), encoding="utf-8")


async def test_memory_policy_list_set_and_override_provider(env) -> None:
    bot, link = env["bot"], env["link"]
    listed = (await rpc(env, "memory.policy.list", {}))["payload"]
    assert listed["policies"] == {"chat": "full", "quiz": "confirm_only", "story": "confirm_only"}
    assert listed["defaults"] == listed["policies"]
    res = await rpc(env, "memory.policy.set", {"mode_id": "quiz", "memory_policy": "isolated"})
    assert res["payload"]["memory_policy"] == "isolated"
    upd = link.of_type("memory.policy.updated")[-1]["payload"]
    assert upd["changed"] == ["quiz"] and upd["policies"]["quiz"] == "isolated"
    assert "seq" not in link.of_type("memory.policy.updated")[-1]
    assert (await rpc(env, "memory.policy.set", {"mode_id": "nope", "memory_policy": "full"}))["payload"][
        "code"
    ] == "BAD_REQUEST"
    assert (await rpc(env, "memory.policy.set", {"mode_id": "chat", "memory_policy": "real_facts_only"}))["payload"][
        "code"
    ] == "BAD_REQUEST"
    # Re-seeding never overrides the user's change.
    await asyncio.to_thread(bot.policy.seed_defaults_sync)
    assert bot.sync_db.get_app_mode_memory_policy("quiz") == "isolated"

    sid = await create_session(env)
    assert bot.policy.resolve_override(sid) == "full"
    _write_games(env["tmp"], sid, [{"id": "quiz#1", "game": "quiz", "active": True}])
    assert bot.policy.current_mode(sid) == "quiz"
    assert bot.policy.resolve_override(sid) == "isolated"
    _write_games(env["tmp"], sid, {"instances": [
        {"id": "quiz#1", "game": "quiz", "active": True},
        {"id": "story#1", "game": "story", "active": True, "parent": "quiz#1"},
    ]})
    assert bot.policy.current_mode(sid) == "quiz"
    _write_games(env["tmp"], sid, [{"id": "q#1", "game": "quiz", "active": False}])
    assert bot.policy.current_mode(sid) == "chat"
    _write_games(env["tmp"], sid, [{"id": "x#1", "game": "unconfigured_mode", "active": True}])
    assert bot.policy.resolve_override(sid) is None
    (env["tmp"] / "sessions" / sid / "game" / "games.json").write_text("{broken", encoding="utf-8")
    assert bot.policy.current_mode(sid) == "chat"


async def _run_context_injection(env, sid: str, frontmatter_policy: str, metadata: dict[str, Any]) -> tuple[str, str]:
    """Run the real per-turn context injection and return (effective policy, injected user content)."""
    staging = env["tmp"] / "sessions" / sid
    (staging / "context").mkdir(parents=True, exist_ok=True)
    (staging / "context" / "10-game-quiz#1.md").write_text(
        f"---\nmemory_policy: {frontmatter_policy}\n---\nQUIZ BLOCK BODY\n", encoding="utf-8"
    )
    (staging / "context" / "20-scene.md").write_text("SCENE BLOCK BODY\n", encoding="utf-8")
    _write_games(env["tmp"], sid, [{"id": "quiz#1", "game": "quiz", "active": True}])
    msg = Message(session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", sender="dev-1", role=MessageRole.USER,
                  content="hello", metadata=metadata, status=MessageStatus.PROCESSING)
    await env["gw"].db.save_message(msg)
    tool_runner = type("TR", (), {"tool_context": type("TC", (), {})()})()
    te = TurnExecutor(sid, env["gw"], tool_runner, context=env["ctx"])
    history = [msg]
    await te._inject_context_and_trigger_consolidation(
        history, msg, env["ctx"].get_llm, session_staging_dir=str(staging)
    )
    return te._turn_context_policy, history[0].content


async def test_user_mode_policy_overrides_static_frontmatter(env) -> None:
    """frontmatter `isolated` + user pref `full` => `full`; frontmatter `full` + user `isolated` => `isolated`."""
    bot = env["bot"]
    sid = await create_session(env)
    bot.sync_db.set_app_mode_memory_policy("quiz", "full")
    policy, _ = await _run_context_injection(env, sid, "isolated", {})
    assert policy == "full"
    bot.sync_db.set_app_mode_memory_policy("quiz", "isolated")
    policy, _ = await _run_context_injection(env, sid, "full", {})
    assert policy == "isolated"
    bot.sync_db.set_app_mode_memory_policy("quiz", "confirm_only")
    policy, _ = await _run_context_injection(env, sid, "real_facts_only", {})
    assert policy == "confirm_only"
    # Without a provider the static rank-merge applies.
    env["ctx"].memory_policy_override_provider = None
    policy, _ = await _run_context_injection(env, sid, "isolated", {})
    assert policy == "isolated"


async def test_ignore_context_globs_and_mode_prefs_injection(env) -> None:
    sid = await create_session(env)
    env["bot"].sync_db.set_app_mode_memory_policy("quiz", "confirm_only")
    policy, content = await _run_context_injection(
        env, sid, "isolated", {"ignore_context_globs": ["game-*"], "mode_prefs": {"level": "hard"}}
    )
    assert "QUIZ BLOCK BODY" not in content and "SCENE BLOCK BODY" in content
    assert '<context name="app-prefs">' in content and '"level":"hard"' in content
    # Policy still derived from the current mode, not degraded by the hidden block.
    assert policy == "confirm_only"
    _, content2 = await _run_context_injection(env, sid, "isolated", {})
    assert "QUIZ BLOCK BODY" in content2 and "app-prefs" not in content2


# ---------------------------------------------------------------------------------------------- proposals


async def test_memory_proposal_events_and_resolution(env) -> None:
    bot, link = env["bot"], env["link"]
    sid = await create_session(env)
    proposal = await asyncio.to_thread(
        bot.proposal_store.stage_sync,
        session_id=sid,
        role="helper",
        ops=[{"op": "upsert_bullet", "file": "user/profile.md", "section": "Diet", "text": "Allergic to peanuts",
              "tags": ["#direct"]}],
        message_id="m1",
        source_mode="quiz",
    )
    await bot.proposal_store.notify(proposal)
    ev = [e for e in link.of_type("memory.proposal") if e["session_id"] == sid][-1]
    assert ev["payload"]["proposal_id"] == proposal["proposal_id"] and "seq" in ev
    assert ev["payload"]["ops"] == [{"file": "user/profile.md", "section": "Diet", "text": "Allergic to peanuts",
                                     "tags": ["#direct"]}]
    listed = (await rpc(env, "memory.proposal.list", {"status": "pending"}))["payload"]["proposals"]
    assert [p["proposal_id"] for p in listed] == [proposal["proposal_id"]]

    profile = env["tmp"] / "roles" / "helper" / "memories" / "user" / "profile.md"
    assert not profile.exists()
    res = await rpc(env, "memory.proposal.resolve", {"proposal_id": proposal["proposal_id"], "action": "confirm"})
    assert res["payload"]["status"] == "applied" and res["payload"]["ok"] is True
    assert "Allergic to peanuts" in profile.read_text(encoding="utf-8")
    resolved = [e for e in link.of_type("memory.proposal.resolved") if e["session_id"] == sid][-1]
    assert resolved["payload"]["status"] == "applied" and "seq" in resolved and "device_id" not in resolved
    again = await rpc(env, "memory.proposal.resolve", {"proposal_id": proposal["proposal_id"], "action": "dismiss"})
    assert again["payload"]["code"] == "PROPOSAL_NOT_PENDING"
    missing = await rpc(env, "memory.proposal.resolve", {"proposal_id": "mp_nope", "action": "confirm"})
    assert missing["payload"]["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------------------------- memory files


async def test_memory_files_put_get_list_pin_and_traversal(env) -> None:
    put = await rpc(env, "memory.put", {"role": "helper", "rel_path": "topics/plan.md", "content": "# Plan\n- a\n"})
    assert put["type"] == "memory.put.result"
    got = await rpc(env, "memory.get", {"role": "helper", "rel_path": "topics/plan.md"})
    assert got["payload"]["content"].startswith("# Plan")
    listed = (await rpc(env, "memory.list", {"role": "helper"}))["payload"]["files"]
    assert [f["rel_path"] for f in listed] == ["topics/plan.md"]
    for rel in ("../../escape.md", "/etc/passwd", "topics/../../x.md", ".hidden.md"):
        bad = await rpc(env, "memory.put", {"role": "helper", "rel_path": rel, "content": "x"})
        assert bad["type"] == "error" and bad["payload"]["code"] == "BAD_REQUEST", rel
    bad_role = await rpc(env, "memory.list", {"role": "../roles"})
    assert bad_role["payload"]["code"] == "BAD_REQUEST"
    pinned_bad = await rpc(env, "memory.pin", {"role": "helper", "rel_path": "topics/plan.md", "pinned": True})
    assert pinned_bad["payload"]["code"] == "BAD_REQUEST"
    await rpc(env, "memory.put", {"role": "helper", "rel_path": "user/profile.md", "content": "- likes tea\n"})
    pin = await rpc(env, "memory.pin", {"role": "helper", "rel_path": "user/profile.md", "pinned": True})
    assert pin["payload"]["pinned"] is True
    text = (env["tmp"] / "roles" / "helper" / "memories" / "user" / "profile.md").read_text(encoding="utf-8")
    assert text.startswith("---\nvisibility: PINNED\n---\n")


# ---------------------------------------------------------------------------------------------- pairing


async def test_pairing_offer_redeem_burn_and_hash(env) -> None:
    from kesoku.gateway.chatbot.app.pairing import create_pairing_offer, hash_pairing_code

    bot, link = env["bot"], env["link"]
    offer = create_pairing_offer(bot.sync_db, "Phone")
    assert len(offer["code"]) == 6 and offer["code"].isdigit()
    assert offer["code_hash"] == hash_pairing_code(offer["code"], offer["salt"])
    await bot.pairing.poll_once()
    sent = link.of_type("pair.offer")[-1]["payload"]
    assert sent["code_hash"] == offer["code_hash"] and offer["code"] not in json.dumps(sent)
    await bot.pairing.poll_once()
    assert len(link.of_type("pair.offer")) == 1
    await bot._on_frame(make_envelope("pair.redeem", {"device_id": "dev-paired", "name": "Phone", "platform": "ios"}))
    assert bot.sync_db.get_app_device("dev-paired")["kind"] == "pair"
    assert bot.sync_db.get_app_pairing(offer["code_hash"])["redeemed_device_id"] == "dev-paired"

    offer2 = create_pairing_offer(bot.sync_db, "Tablet")
    await bot._on_frame(make_envelope("pair.burned", {"code_hash": offer2["code_hash"], "attempts": 5}))
    assert bot.sync_db.get_app_pairing(offer2["code_hash"]) is None

    await bot._on_frame(make_envelope("device.registered", {"device_id": "dev-web", "kind": "google",
                                                            "email": "owner@example.com", "platform": "web"}))
    assert bot.sync_db.get_app_device("dev-web")["name"] == "owner@example.com"
    bot.sync_db.revoke_app_device("dev-web")
    await bot.pairing.poll_once()
    assert "dev-web" in link.of_type("device.revoked")[-1]["payload"]["device_ids"]
    assert "dev-web" in bot._hello_extra()["revoked_device_ids"]


async def test_pairing_redeem_prefers_custom_offer_name_over_generic_client_name(env) -> None:
    from kesoku.gateway.chatbot.app.pairing import create_pairing_offer

    bot = env["bot"]
    # 1. Custom offer name overrides generic client fallbacks ("Android", "iPhone", "", "device").
    offer_android = create_pairing_offer(bot.sync_db, "Chii Pixel 9")
    await bot._on_frame(
        make_envelope("pair.redeem", {"device_id": "dev-pixel", "name": "Android", "platform": "android"})
    )
    assert bot.sync_db.get_app_device("dev-pixel")["name"] == "Chii Pixel 9"
    assert bot.sync_db.get_app_pairing(offer_android["code_hash"])["redeemed_device_id"] == "dev-pixel"

    offer_iphone = create_pairing_offer(bot.sync_db, "Chii iPhone 17")
    await bot._on_frame(
        make_envelope("pair.redeem", {"device_id": "dev-iphone", "name": "iPhone", "platform": "ios"})
    )
    assert bot.sync_db.get_app_device("dev-iphone")["name"] == "Chii iPhone 17"
    assert bot.sync_db.get_app_pairing(offer_iphone["code_hash"])["redeemed_device_id"] == "dev-iphone"

    # 2. Custom non-generic client name is preserved even when offer has a custom name.
    create_pairing_offer(bot.sync_db, "Offer Tablet")
    await bot._on_frame(
        make_envelope("pair.redeem", {"device_id": "dev-custom", "name": "Studio iPad Pro", "platform": "ios"})
    )
    assert bot.sync_db.get_app_device("dev-custom")["name"] == "Studio iPad Pro"

    # 3. Default offer name ("device") falls back to the client-provided platform name.
    create_pairing_offer(bot.sync_db, "device")
    await bot._on_frame(
        make_envelope("pair.redeem", {"device_id": "dev-default", "name": "Android", "platform": "android"})
    )
    assert bot.sync_db.get_app_device("dev-default")["name"] == "Android"


async def test_auto_session_rename_emits_session_updated_not_turn_event(env) -> None:
    bot, link, gw = env["bot"], env["link"], env["gw"]
    sid = await create_session(env, role="other_role")
    rename = Message(session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", sender="Kesoku",
                     role=MessageRole.SYSTEM, type=MessageType.SESSION_RENAME, content="Weekend plans",
                     status=MessageStatus.PENDING)
    await gw.db.save_message(rename)
    before_turn_events = len(link.of_type("turn.event"))
    await bot.render_outgoing_message(rename)
    upd = [e for e in link.of_type("session.updated") if e["session_id"] == sid][-1]
    assert upd["payload"]["title"] == "Weekend plans" and upd["payload"]["role"] == "other_role"
    assert upd["payload"]["session_id"] == sid and "seq" in upd
    assert len(link.of_type("turn.event")) == before_turn_events
    assert (await gw.db.get_message(rename.id)).status == MessageStatus.DELIVERED


async def test_memory_proposal_event_carries_turn_id(env) -> None:
    bot, link = env["bot"], env["link"]
    sid = await create_session(env)
    proposal = await asyncio.to_thread(
        bot.proposal_store.stage_sync, session_id=sid, role="helper",
        ops=[{"op": "upsert_bullet", "file": "user/profile.md", "text": "Owns a bicycle"}], message_id="user-msg-1",
    )
    await bot.proposals.on_staged(proposal)
    ev = [e for e in link.of_type("memory.proposal") if e["session_id"] == sid][-1]["payload"]
    assert ev["message_id"] == "user-msg-1" and ev["turn_id"] == "user-msg-1"


def _write_test_wav(path, seconds: float = 0.75, rate: int = 8000) -> None:
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


async def test_final_message_and_history_carry_persona_display_name(env) -> None:
    bot, link, gw = env["bot"], env["link"], env["gw"]
    registry = bot.persona.registry
    meta = registry.load_meta("helper") or {}
    meta["display_name"] = "Hana Example"
    registry.write_meta("helper", meta)
    sid = await create_session(env)
    final = Message(session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", sender="Kesoku",
                    role=MessageRole.ASSISTANT, content="こんにちは", status=MessageStatus.PENDING)
    await gw.db.save_message(final)
    await bot.render_outgoing_message(final)
    fin = [e for e in link.of_type("message.final") if e["session_id"] == sid][-1]
    assert fin["payload"]["sender"] == {
        "kind": "assistant", "name": "Kesoku", "role_id": "helper", "display_name": "Hana Example",
    }

    hist = (await rpc(env, "history.page", {"session_id": sid}))["payload"]["messages"]
    assistant = [m for m in hist if m["role"] == "assistant"][-1]
    assert assistant["sender"]["role_id"] == "helper" and assistant["sender"]["display_name"] == "Hana Example"
    snap = (await rpc(env, "session.open", {"session_id": sid}))["payload"]
    last = [m for m in snap["last_messages"] if m["role"] == "assistant"][-1]
    assert last["sender"]["display_name"] == "Hana Example" and last["sender"]["name"] == "Kesoku"


async def test_notification_final_has_no_persona_display_name(env) -> None:
    bot, link, gw = env["bot"], env["link"], env["gw"]
    sid = await create_session(env)
    note = Message(session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", sender="Notification",
                   role=MessageRole.ASSISTANT, content="reminder", status=MessageStatus.PENDING)
    await gw.db.save_message(note)
    await bot.render_outgoing_message(note)
    fin = [e for e in link.of_type("message.final") if e["session_id"] == sid][-1]
    assert "display_name" not in fin["payload"]["sender"]


async def test_voice_media_ready_and_hydrate_carry_duration(env) -> None:
    bot, link = env["bot"], env["link"]
    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid
    staging.mkdir(parents=True, exist_ok=True)
    wav = staging / "line.wav"
    _write_test_wav(wav, 0.75)
    msg = Message(session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", sender="Kesoku",
                  role=MessageRole.ASSISTANT, content=f"[voice: {wav}]")
    await bot._upload_and_announce(msg, str(wav), "voice", "ph_voice", None)
    ready = [e for e in link.of_type("media.ready") if e["payload"].get("placeholder_id") == "ph_voice"][-1]
    assert ready["payload"]["kind"] == "voice" and ready["payload"]["duration_ms"] == 750

    (staging / "note.txt").write_text("plain", encoding="utf-8")
    await bot._upload_and_announce(msg, str(staging / "note.txt"), "file", "ph_file", None)
    plain = [e for e in link.of_type("media.ready") if e["payload"].get("placeholder_id") == "ph_file"][-1]
    assert "duration_ms" not in plain["payload"]

    hyd = await rpc(env, "media.hydrate", {"session_id": sid, "message_id": msg.id, "ref": "line.wav"})
    assert hyd["type"] == "media.ready" and hyd["payload"]["duration_ms"] == 750
    (staging / "broken.wav").write_bytes(b"RIFF....WAVE")
    broken = await rpc(env, "media.hydrate", {"session_id": sid, "ref": "broken.wav"})
    assert broken["type"] == "media.ready" and "duration_ms" not in broken["payload"]


async def test_mistakes_synthesis_character_hydrate_prefs_sync_and_lexical_token(env) -> None:
    from kesoku.gateway.chatbot.app.commands import _parse_lexical_token

    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid
    (staging / "game").mkdir(parents=True, exist_ok=True)
    (staging / "game" / "lesson.json").write_text(json.dumps({"book": "初级", "lesson": 10}), encoding="utf-8")
    (staging / "game" / "pokemon.json").write_text(json.dumps({"game_version": "sv"}), encoding="utf-8")

    # 1. Global mistakes.json synthesis via StateWatcher
    g_dir = env["tmp"] / "games" / "jp_classroom" / "data"
    g_dir.mkdir(parents=True, exist_ok=True)
    (g_dir / "mistakes.json").write_text(
        json.dumps(
            {
                "next_id": 2,
                "active": [
                    {"id": 1, "created": "2026-01-01", "point": "助词に", "next_due": "2026-01-01"},
                ],
                "archived": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    snap = (await rpc(env, "session.open", {"session_id": sid}))["payload"]
    m_state = snap["state_files"].get("game/mistakes.json")
    assert isinstance(m_state, dict) and m_state["due_count"] == 1
    assert m_state["due"][0]["point"] == "助词に"

    # 2. mode.prefs.set syncs to context/05-app-prefs.md and game/pokemon.json
    env["bot"].app_config.modes[0].prefs_schema = {"game_version": ["sv", "za"], "pinned_context": "string"}
    await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": {"game_version": "za", "pinned_context": "快龙"}})
    assert "za" in (staging / "context" / "05-app-prefs.md").read_text(encoding="utf-8")
    pkmn_updated = json.loads((staging / "game" / "pokemon.json").read_text(encoding="utf-8"))
    assert pkmn_updated["game_version"] == "za" and pkmn_updated["pinned_context"] == "快龙"

    # 3. Character registry media.hydrate by character name
    char_dir = env["tmp"] / "characters" / "cast"
    char_dir.mkdir(parents=True, exist_ok=True)
    (char_dir / "heroine.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (char_dir / "characters.json").write_text(
        json.dumps([{"name": "Heroine Alpha", "trigger": "主角A", "image": "heroine.png"}]),
        encoding="utf-8",
    )
    ch_hyd = await rpc(env, "media.hydrate", {"session_id": sid, "ref": "Heroine Alpha"})
    assert ch_hyd["type"] == "media.ready" and ch_hyd["payload"]["kind"] == "file"

    # 4. _parse_lexical_token structured YAML parsing
    tok = _parse_lexical_token(
        "行こう",
        "```yaml\nreading: いこう\npos: 五段动词\ngloss_zh: 走吧、去吧\nlemma: 行く\nconj: 意向形\n```",
    )
    assert tok["surface"] == "行こう"
    assert tok["reading"] == "いこう"
    assert tok["lemma"] == "行く"
    assert tok["gloss_zh"] == "走吧、去吧"


async def test_workspace_list_and_read_jailed(env) -> None:
    tmp = env["tmp"]
    (tmp / "README.md").write_text("# Agent Workspace\nHello world\n", encoding="utf-8")
    (tmp / ".env").write_text("SECRET=123", encoding="utf-8")
    (tmp / "roles" / "helper" / "avatar.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    root_list = (await rpc(env, "workspace.list", {}))["payload"]
    assert root_list["rel_path"] == ""
    names = [e["name"] for e in root_list["entries"]]
    assert "roles" in names and "README.md" in names
    assert ".env" not in names and "kesoku.db" not in names

    sub_list = (await rpc(env, "workspace.list", {"rel_path": "roles/helper"}))["payload"]
    assert sub_list["rel_path"] == "roles/helper"
    assert any(e["name"] == "intro.md" for e in sub_list["entries"])

    read_md = (await rpc(env, "workspace.read", {"rel_path": "README.md"}))["payload"]
    assert read_md["kind"] == "text" and "# Agent Workspace" in read_md["content"]

    read_img = (await rpc(env, "workspace.read", {"rel_path": "roles/helper/avatar.png"}))["payload"]
    assert read_img["kind"] == "media" and read_img["url"].startswith("https://store.invalid/")

    for forbidden in ("../outside.txt", ".env", "kesoku.db", "/etc/passwd"):
        err = await rpc(env, "workspace.read", {"rel_path": forbidden})
        assert err["type"] == "error" and err["payload"]["code"] == "BAD_REQUEST"


async def test_workspace_write_mkdir_delete_download_crud(env) -> None:
    import base64 as _b64

    tmp = env["tmp"]
    (tmp / "README.md").write_text("# Agent Workspace\n", encoding="utf-8")

    # read exposes `writable` for complete text files
    read_md = (await rpc(env, "workspace.read", {"rel_path": "README.md"}))["payload"]
    assert read_md["writable"] is True

    # text write (creates parent dirs) + overwrite in place
    w = await rpc(env, "workspace.write", {"path": "notes/todo.md", "content": "- 买牛奶\n"})
    assert w["type"] == "workspace.write.result", w
    assert w["payload"]["path"] == "notes/todo.md" and w["payload"]["name"] == "todo.md"
    assert w["payload"]["kind"] == "text" and w["payload"]["size"] == len("- 买牛奶\n".encode())
    assert (tmp / "notes" / "todo.md").read_text(encoding="utf-8") == "- 买牛奶\n"
    w2 = (await rpc(env, "workspace.write", {"path": "notes/todo.md", "content": "done"}))["payload"]
    assert w2["size"] == 4 and (tmp / "notes" / "todo.md").read_text(encoding="utf-8") == "done"
    no_over = await rpc(env, "workspace.write", {"path": "notes/todo.md", "content": "x", "overwrite": False})
    assert no_over["type"] == "error" and no_over["payload"]["code"] == "BAD_REQUEST"

    # base64 binary upload + chunked append
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    up = (await rpc(env, "workspace.write", {"path": "uploads/pic.png", "base64": _b64.b64encode(png[:100]).decode()}))
    assert up["payload"]["kind"] == "media"
    chunk2 = _b64.b64encode(png[100:]).decode()
    await rpc(env, "workspace.write", {"path": "uploads/pic.png", "base64": chunk2, "append": True})
    assert (tmp / "uploads" / "pic.png").read_bytes() == png
    bad_b64 = await rpc(env, "workspace.write", {"path": "uploads/x.bin", "base64": "@@not-base64@@"})
    assert bad_b64["type"] == "error"

    # mkdir (idempotent) and writing onto a directory is rejected
    mk = (await rpc(env, "workspace.mkdir", {"path": "projects/new_dir"}))
    assert mk["type"] == "workspace.mkdir.result"
    assert mk["payload"] == {"path": "projects/new_dir", "name": "new_dir", "kind": "dir"}
    assert (tmp / "projects" / "new_dir").is_dir()
    assert (await rpc(env, "workspace.mkdir", {"path": "projects/new_dir"}))["type"] == "workspace.mkdir.result"
    on_dir = await rpc(env, "workspace.write", {"path": "projects/new_dir", "content": "x"})
    assert on_dir["type"] == "error"

    # download inlines base64
    dl = await rpc(env, "workspace.download", {"path": "uploads/pic.png"})
    assert dl["type"] == "workspace.download.result"
    assert dl["payload"]["mime_type"] == "image/png" and dl["payload"]["size"] == len(png)
    assert _b64.b64decode(dl["payload"]["base64"]) == png
    assert (await rpc(env, "workspace.download", {"path": "projects"}))["type"] == "error"
    missing = await rpc(env, "workspace.download", {"path": "nope.txt"})
    assert missing["payload"]["code"] == "NOT_FOUND"

    # delete file + recursive user folder; root / protected top-level dirs are refused
    d1 = await rpc(env, "workspace.delete", {"path": "notes/todo.md"})
    assert d1["type"] == "workspace.delete.result" and d1["payload"] == {"path": "notes/todo.md", "deleted": True}
    assert not (tmp / "notes" / "todo.md").exists()
    (tmp / "projects" / "new_dir" / "a.txt").write_text("a", encoding="utf-8")
    assert (await rpc(env, "workspace.delete", {"path": "projects"}))["payload"]["deleted"] is True
    assert not (tmp / "projects").exists()
    for protected in ("roles", "skills", "instructions", "", "/"):
        err = await rpc(env, "workspace.delete", {"path": protected})
        assert err["type"] == "error", protected
    assert (tmp / "roles" / "helper" / "intro.md").exists()
    # files inside protected top-level dirs remain editable / deletable
    assert (await rpc(env, "workspace.delete", {"path": "roles/other_role/intro.md"}))["payload"]["deleted"] is True
    # a folder hiding a database is not recursively deleted
    (tmp / "data").mkdir()
    (tmp / "data" / "cache.sqlite3").write_bytes(b"x")
    assert (await rpc(env, "workspace.delete", {"path": "data"}))["type"] == "error"
    assert (tmp / "data" / "cache.sqlite3").exists()


async def test_workspace_mutations_reject_traversal_and_hidden(env) -> None:
    tmp = env["tmp"]
    (tmp / ".env").write_text("SECRET=1", encoding="utf-8")
    cases = ("../outside.txt", "notes/../../outside.txt", "/etc/passwd", ".env", ".env.local", "kesoku.db",
             "kesoku.db-wal", "backup/kesoku.db.bak", ".git/config", "__pycache__/x.pyc", "id.pem")
    for bad in cases:
        for rpc_type, extra in (
            ("workspace.write", {"content": "x"}),
            ("workspace.mkdir", {}),
            ("workspace.delete", {}),
            ("workspace.download", {}),
        ):
            err = await rpc(env, rpc_type, {"path": bad, **extra})
            assert err["type"] == "error" and err["payload"]["code"] == "BAD_REQUEST", (rpc_type, bad, err)
    assert (tmp / ".env").read_text(encoding="utf-8") == "SECRET=1"
    assert (tmp / "kesoku.db").exists()
    assert not (tmp.parent / "outside.txt").exists()
    # symlink escaping the workspace is refused
    outside = tmp.parent / f"{tmp.name}_outside"
    outside.mkdir()
    (tmp / "escape").symlink_to(outside, target_is_directory=True)
    err = await rpc(env, "workspace.write", {"path": "escape/pwn.txt", "content": "x"})
    assert err["type"] == "error"
    assert not (outside / "pwn.txt").exists()
    # oversize payload
    big = await rpc(env, "workspace.write", {"path": "big.txt", "content": "x" * (10 * 1024 * 1024 + 1)})
    assert big["type"] == "error"


async def test_workspace_id_and_read_raw_sha256(env) -> None:
    import hashlib
    import re

    tmp = env["tmp"]
    # Root list exposes persistent workspace_id and workspace_cas flag; subdirs do not.
    root_list = (await rpc(env, "workspace.list", {}))["payload"]
    ws_id = root_list["workspace_id"]
    assert re.fullmatch(r"ws_[0-9a-f]{32}", ws_id)
    assert root_list["workspace_cas"] is True

    sub_list = (await rpc(env, "workspace.list", {"rel_path": "roles/helper"}))["payload"]
    assert "workspace_id" not in sub_list
    assert "workspace_cas" not in sub_list

    # Simulating reconnect / restart with a fresh RequestHandlers instance preserves workspace_id.
    env["bot"].requests = commands_mod.RequestHandlers(env["bot"])
    env["bot"]._handlers = env["bot"]._build_handlers()
    root_list_2 = (await rpc(env, "workspace.list", {}))["payload"]
    assert root_list_2["workspace_id"] == ws_id

    # Raw-byte sha256 is computed BEFORE lossy UTF-8 decoding.
    raw_invalid_utf8 = b"header\xff\xfetail\r\n"
    (tmp / "raw.txt").write_bytes(raw_invalid_utf8)
    read_raw = (await rpc(env, "workspace.read", {"rel_path": "raw.txt"}))["payload"]
    expected_raw_sha = hashlib.sha256(raw_invalid_utf8).hexdigest()
    decoded_sha = hashlib.sha256(read_raw["content"].encode("utf-8")).hexdigest()
    assert expected_raw_sha != decoded_sha
    assert read_raw["truncated"] is False
    assert read_raw["sha256"] == expected_raw_sha

    # Truncated text files omit sha256 (or return None).
    (tmp / "huge.txt").write_bytes(b"a" * (commands_mod.WORKSPACE_TEXT_MAX_BYTES + 10))
    read_huge = (await rpc(env, "workspace.read", {"rel_path": "huge.txt"}))["payload"]
    assert read_huge["truncated"] is True
    assert read_huge.get("sha256") is None


async def test_workspace_write_cas_semantics_and_validation(env) -> None:
    import hashlib

    tmp = env["tmp"]

    # 1. expected_sha256=None: create-only succeeds when absent.
    w1 = (
        await rpc(env, "workspace.write", {"path": "cas/log.json", "content": '{"v":1}', "expected_sha256": None})
    )["payload"]
    sha1 = hashlib.sha256(b'{"v":1}').hexdigest()
    assert w1["cas"] is True
    assert w1["sha256"] == sha1
    assert "conflict" not in w1

    # 2. expected_sha256=None: conflicts when file already exists without overwriting.
    c1 = (
        await rpc(env, "workspace.write", {"path": "cas/log.json", "content": '{"v":99}', "expected_sha256": None})
    )["payload"]
    assert c1["conflict"] is True
    assert c1["cas"] is True
    assert c1["path"] == "cas/log.json"
    assert c1["sha256"] == sha1
    assert isinstance(c1["mtime"], float)
    assert (tmp / "cas" / "log.json").read_text(encoding="utf-8") == '{"v":1}'

    # 3. expected_sha256=<matching>: succeeds and updates sha256.
    w2 = (
        await rpc(env, "workspace.write", {"path": "cas/log.json", "content": '{"v":2}', "expected_sha256": sha1})
    )["payload"]
    sha2 = hashlib.sha256(b'{"v":2}').hexdigest()
    assert w2["cas"] is True
    assert w2["sha256"] == sha2
    assert (tmp / "cas" / "log.json").read_text(encoding="utf-8") == '{"v":2}'

    # 4. expected_sha256=<stale>: conflicts without overwriting.
    c2 = (
        await rpc(env, "workspace.write", {"path": "cas/log.json", "content": '{"v":3}', "expected_sha256": sha1})
    )["payload"]
    assert c2["conflict"] is True
    assert c2["cas"] is True
    assert c2["sha256"] == sha2
    assert (tmp / "cas" / "log.json").read_text(encoding="utf-8") == '{"v":2}'

    # 5. expected_sha256=<hex> on missing file: conflicts with sha256=None, mtime=None.
    c3 = (
        await rpc(env, "workspace.write", {"path": "cas/missing.json", "content": "{}", "expected_sha256": sha1})
    )["payload"]
    assert c3 == {"conflict": True, "cas": True, "path": "cas/missing.json", "sha256": None, "mtime": None}
    assert not (tmp / "cas" / "missing.json").exists()

    # 6. Raw-byte CAS works even when disk file has invalid UTF-8 bytes.
    raw_bytes = b"bad\xffbytes"
    (tmp / "cas" / "raw.txt").write_bytes(raw_bytes)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    bad_utf8_sha = hashlib.sha256(raw_bytes.decode("utf-8", errors="replace").encode("utf-8")).hexdigest()
    c_raw = (
        await rpc(env, "workspace.write", {"path": "cas/raw.txt", "content": "ok", "expected_sha256": bad_utf8_sha})
    )["payload"]
    assert c_raw["conflict"] is True and c_raw["sha256"] == raw_sha
    w_raw = (
        await rpc(env, "workspace.write", {"path": "cas/raw.txt", "content": "ok", "expected_sha256": raw_sha})
    )["payload"]
    assert w_raw["cas"] is True and w_raw["sha256"] == hashlib.sha256(b"ok").hexdigest()

    # 7. Validation: expected_sha256 + append is forbidden; malformed hashes are rejected.
    for bad_payload in (
        {"path": "cas/log.json", "content": "x", "append": True, "expected_sha256": None},
        {"path": "cas/log.json", "content": "x", "append": True, "expected_sha256": sha2},
        {"path": "cas/log.json", "content": "x", "expected_sha256": ""},
        {"path": "cas/log.json", "content": "x", "expected_sha256": "A" * 64},
        {"path": "cas/log.json", "content": "x", "expected_sha256": "a" * 63},
        {"path": "cas/log.json", "content": "x", "expected_sha256": "g" * 64},
        {"path": "cas/log.json", "content": "x", "expected_sha256": 12345},
        {"path": "cas/log.json", "content": "x", "expected_sha256": True},
    ):
        err = await rpc(env, "workspace.write", bad_payload)
        assert err["type"] == "error" and err["payload"]["code"] == "BAD_REQUEST", (bad_payload, err)

    # 8. append=True returns sha256 of the complete post-append bytes and cas=True.
    a1 = (await rpc(env, "workspace.write", {"path": "cas/app.txt", "content": "hello "}))["payload"]
    assert a1["cas"] is True and a1["sha256"] == hashlib.sha256(b"hello ").hexdigest()
    a2 = (await rpc(env, "workspace.write", {"path": "cas/app.txt", "content": "world", "append": True}))["payload"]
    assert a2["cas"] is True and a2["sha256"] == hashlib.sha256(b"hello world").hexdigest()


async def test_workspace_concurrent_cas_and_path_lock_serialization(env, monkeypatch) -> None:
    import hashlib
    import threading

    tmp = env["tmp"]
    handlers = env["bot"].requests

    # Case A: Two concurrent create-only CAS writes (expected_sha256=None) -> strictly 1 succeeds, 1 conflicts.
    r1, r2 = await asyncio.gather(
        rpc(env, "workspace.write", {"path": "race/create.json", "content": '{"who":"A"}', "expected_sha256": None}),
        rpc(env, "workspace.write", {"path": "race/create.json", "content": '{"who":"B"}', "expected_sha256": None}),
    )
    payloads = [r1["payload"], r2["payload"]]
    winners = [p for p in payloads if not p.get("conflict")]
    losers = [p for p in payloads if p.get("conflict")]
    assert len(winners) == 1 and len(losers) == 1
    assert losers[0]["cas"] is True
    assert losers[0]["sha256"] == winners[0]["sha256"]
    assert hashlib.sha256((tmp / "race" / "create.json").read_bytes()).hexdigest() == winners[0]["sha256"]

    # Case B: Two concurrent CAS writes with the same expected_sha256 -> strictly 1 succeeds, 1 conflicts.
    base_sha = winners[0]["sha256"]
    r3, r4 = await asyncio.gather(
        rpc(
            env,
            "workspace.write",
            {"path": "race/create.json", "content": '{"who":"C"}', "expected_sha256": base_sha},
        ),
        rpc(
            env,
            "workspace.write",
            {"path": "race/create.json", "content": '{"who":"D"}', "expected_sha256": base_sha},
        ),
    )
    payloads2 = [r3["payload"], r4["payload"]]
    winners2 = [p for p in payloads2 if not p.get("conflict")]
    losers2 = [p for p in payloads2 if p.get("conflict")]
    assert len(winners2) == 1 and len(losers2) == 1
    assert losers2[0]["sha256"] == winners2[0]["sha256"]
    assert hashlib.sha256((tmp / "race" / "create.json").read_bytes()).hexdigest() == winners2[0]["sha256"]

    # Case C: Non-CAS write, CAS write, and delete share the exact same per-canonical-path lock (zero overlap).
    (tmp / "race" / "shared.txt").write_text("init", encoding="utf-8")
    init_sha = hashlib.sha256(b"init").hexdigest()
    active_lock = threading.Lock()
    active_count = 0
    max_concurrent = 0
    real_replace = os.replace
    real_unlink = os.unlink

    def tracked_replace(src: str, dst: str) -> None:
        nonlocal active_count, max_concurrent
        if dst.endswith("shared.txt"):
            with active_lock:
                active_count += 1
                max_concurrent = max(max_concurrent, active_count)
            try:
                time.sleep(0.05)
                real_replace(src, dst)
            finally:
                with active_lock:
                    active_count -= 1
        else:
            real_replace(src, dst)

    def tracked_unlink(p: str) -> None:
        nonlocal active_count, max_concurrent
        if str(p).endswith("shared.txt"):
            with active_lock:
                active_count += 1
                max_concurrent = max(max_concurrent, active_count)
            try:
                time.sleep(0.05)
                real_unlink(p)
            finally:
                with active_lock:
                    active_count -= 1
        else:
            real_unlink(p)

    monkeypatch.setattr(os, "replace", tracked_replace)
    monkeypatch.setattr(os, "unlink", tracked_unlink)

    await asyncio.gather(
        rpc(env, "workspace.write", {"path": "race/shared.txt", "content": "non-cas"}),
        rpc(env, "workspace.write", {"path": "race/shared.txt", "content": "cas", "expected_sha256": init_sha}),
        rpc(env, "workspace.delete", {"path": "race/shared.txt"}),
    )
    assert max_concurrent == 1
    # Ref-counted path lock map must not leak entries after operations complete.
    assert handlers._workspace_locks == {}


async def test_workspace_list_survives_workspace_id_db_error(env, monkeypatch) -> None:
    tmp = env["tmp"]
    (tmp / "README.md").write_text("hi", encoding="utf-8")
    handlers = env["bot"].requests
    handlers._workspace_id = None

    def boom() -> str:
        raise RuntimeError("sqlite locked")

    monkeypatch.setattr(env["bot"].sync_db, "get_or_create_workspace_id", boom)
    res = await rpc(env, "workspace.list", {})
    assert res["type"] == "workspace.list.result"
    payload = res["payload"]
    assert any(e["name"] == "README.md" for e in payload["entries"])
    assert "workspace_id" not in payload
    assert "workspace_cas" not in payload


async def test_workspace_path_lock_held_until_cancelled_thread_finishes(env, monkeypatch) -> None:
    import threading

    tmp = env["tmp"]
    handlers = env["bot"].requests
    (tmp / "cancel.txt").write_text("v0", encoding="utf-8")

    entered = threading.Event()
    release = threading.Event()
    overlap = False
    in_flight = False
    state_lock = threading.Lock()
    real_replace = os.replace

    def slow_replace(src: str, dst: str) -> None:
        nonlocal overlap, in_flight
        if dst.endswith("cancel.txt"):
            with state_lock:
                if in_flight:
                    overlap = True
                in_flight = True
            entered.set()
            try:
                release.wait(timeout=2.0)
                real_replace(src, dst)
            finally:
                with state_lock:
                    in_flight = False
        else:
            real_replace(src, dst)

    monkeypatch.setattr(os, "replace", slow_replace)

    ctx = commands_mod.RequestContext(device_id="dev1", request_id="req1", request_type="workspace.write")
    t1 = asyncio.create_task(handlers.handle_workspace_write(ctx, {"path": "cancel.txt", "content": "v1"}))
    await asyncio.to_thread(entered.wait, 2.0)
    assert entered.is_set()

    # Cancel t1 while its worker thread is still inside os.replace; start a second write on the same path.
    t1.cancel()
    t2 = asyncio.create_task(handlers.handle_workspace_write(ctx, {"path": "cancel.txt", "content": "v2"}))
    await asyncio.sleep(0.05)
    # Lock must still be held (t2 cannot finish while t1's thread is blocked).
    assert not t2.done()
    assert not overlap

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await t1
    r2 = await t2
    assert r2["cas"] is True
    assert not overlap
    assert (tmp / "cancel.txt").read_text(encoding="utf-8") == "v2"
    assert handlers._workspace_locks == {}


async def test_mode_stop_triggers_mode_exit_extraction(env, monkeypatch) -> None:
    bot = env["bot"]
    calls: list[tuple[str, str]] = []

    class FakeConsolidator:
        async def notify_close(self, session_id: str, role: str | None = None, reason: str = "close") -> bool:
            calls.append((session_id, reason))
            return True

    env["gw"].context.memory_consolidator = FakeConsolidator()
    sent: list[str] = []

    async def fake_ingest(**kw: Any) -> dict[str, Any]:
        sent.append(kw["text"])
        return {"ok": True}

    monkeypatch.setattr(bot, "ingest_app_message", fake_ingest)
    monkeypatch.setattr(bot.policy, "current_mode", lambda sid: "quiz")
    await bot.stop_mode("s1", "dev-1", None)
    assert calls == [("s1", "mode_exit")] and sent

    calls.clear()
    monkeypatch.setattr(bot.policy, "current_mode", lambda sid: "chat")
    await bot.stop_mode("s1", "dev-1", None)
    assert calls == []

    # Switching from one interactive mode straight into another also closes out the previous one.
    sid = await create_session(env)
    monkeypatch.setattr(bot.policy, "current_mode", lambda sid: "story")
    await bot.start_mode(sid, "quiz", "dev-1")
    assert calls == [(sid, "mode_exit")]


# ---------------------------------------------------------------------------------------------- v0.3.6


class _SentenceLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **_: Any) -> Any:
        self.prompts.append(prompt)
        await asyncio.sleep(0.05)
        return type("Res", (), {"content": self.content})()


_SENTENCE_YAML = """```yaml
- surface: こんにちは
  reading: こんにちは
  pos: 寒暄语
  gloss_zh: 你好
- surface: 今日
  reading: きょう
  pos: 名词
  gloss_zh: 今天
- surface: よろしくね
  reading: よろしくね
  pos: 惯用表达
  gloss_zh: 请多关照哦
  lemma: よろしく
```"""


async def test_lexical_explain_analyzes_whole_sentence_once(env, monkeypatch) -> None:
    sid = await create_session(env)
    llm = _SentenceLLM(_SENTENCE_YAML)
    monkeypatch.setattr(env["ctx"], "get_llm", lambda *a, **k: llm)
    text = "こんにちは！今日もよろしくね。"

    first = await rpc(env, "lexical.explain", {"session_id": sid, "text": text, "token": "今日"})
    assert first["type"] != "error", first
    body = first["payload"]
    assert body["token"]["surface"] == "今日" and body["token"]["reading"] == "きょう"
    assert [t["surface"] for t in body["tokens"]] == ["こんにちは", "今日", "よろしくね"]
    assert "<sentence>こんにちは！今日もよろしくね。</sentence>" in llm.prompts[0]
    assert "所有核心词汇与意群" in llm.prompts[0]
    assert "**今日**" in body["explanation_markdown"]

    # Another word of the same sentence: served from the sentence cache, zero extra LLM calls.
    second = await rpc(env, "lexical.explain", {"session_id": sid, "text": text, "token": "よろしくね"})
    assert second["payload"]["token"]["gloss_zh"] == "请多关照哦"
    # A sub-span of a cached chunk is answered from its chunk (surface is the tapped token).
    sub = await rpc(env, "lexical.explain", {"session_id": sid, "text": text, "token": "よろしく"})
    assert sub["payload"]["token"]["surface"] == "よろしく"
    assert "よろしくね" in sub["payload"]["token"]["note"]
    assert len(llm.prompts) == 1

    # Concurrent taps on a new sentence share one in-flight analysis.
    text2 = "今日はよろしくね"
    replies = await asyncio.gather(
        rpc(env, "lexical.explain", {"session_id": sid, "text": text2, "token": "今日"}),
        rpc(env, "lexical.explain", {"session_id": sid, "text": text2, "token": "よろしくね"}),
    )
    assert all(r["type"] != "error" for r in replies)
    assert len(llm.prompts) == 2


def test_parse_lexical_tokens_guarantees_target_entry() -> None:
    from kesoku.gateway.chatbot.app.commands import _parse_lexical_tokens

    toks = _parse_lexical_tokens("行こう", "- surface: 行こうよ\n  reading: いこうよ\n  gloss_zh: 走吧\n")
    assert toks[-1]["surface"] == "行こう" and toks[-1]["gloss_zh"] == "走吧"
    wrapped = _parse_lexical_tokens("猫", "tokens:\n  - surface: 猫\n    reading: ねこ\n")
    assert wrapped == [{"surface": "猫", "reading": "ねこ"}]
    single = _parse_lexical_tokens("犬", "reading: いぬ\ngloss_zh: 狗\n")
    assert single[-1]["surface"] == "犬" and single[-1]["reading"] == "いぬ"


async def test_media_hydrate_pkmn_art_and_staging_refs(env) -> None:
    sid = await create_session(env)
    tmp = env["tmp"]
    art = tmp / "games" / "pokemon_assistant" / "data" / "art"
    art.mkdir(parents=True)
    (art / "0987.webp").write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
    ok = await rpc(env, "media.hydrate", {"session_id": sid, "ref": "[file:pkmn/0987.webp]"})
    assert ok["type"] == "media.ready", ok
    ok2 = await rpc(env, "media.hydrate", {"session_id": sid, "ref": "games/pokemon_assistant/data/art/0987.webp"})
    assert ok2["type"] == "media.ready", ok2

    staging = tmp / "sessions" / sid
    (staging / "out").mkdir(parents=True, exist_ok=True)
    (staging / "out" / "card.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    st = await rpc(env, "media.hydrate", {"session_id": sid, "ref": "out/card.png"})
    assert st["type"] == "media.ready", st

    (tmp / "secret.txt").write_text("s", encoding="utf-8")
    for ref in ("pkmn/../../../../secret.txt", "pkmn/missing.webp", "games/../secret.txt", "pkmn/\x00.webp"):
        bad = await rpc(env, "media.hydrate", {"session_id": sid, "ref": ref})
        assert bad["type"] == "error" and bad["payload"]["code"] in ("NOT_FOUND", "BAD_REQUEST"), ref


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


# origin: tests/gateway/chatbot/app/test_v037_user_attachments.py
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


# origin: tests/gateway/chatbot/app/test_v037_user_attachments.py
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


# origin: tests/gateway/chatbot/app/test_v037_user_attachments.py
def test_attachment_only_message_has_empty_text() -> None:
    msg = _user("Attachments:\n[Attachment: rec.webm (audio/webm) saved at /ws/s1/uploads/rec.webm]")
    rec = message_to_record(msg, "app")
    assert rec["text"] == ""
    assert rec["attachments"][0]["mime"] == "audio/webm" and rec["attachments"][0]["object"] == ""


# origin: tests/gateway/chatbot/app/test_v037_user_attachments.py
def test_metadata_mime_fallback_from_extension() -> None:
    msg = _user("hi", {"attachments": [{"path": "/ws/a/clip.wav"}]})
    assert user_attachments(msg)[0]["mime"] == "audio/x-wav" or user_attachments(msg)[0]["mime"].startswith("audio/")


# origin: tests/gateway/chatbot/app/test_v037_user_attachments.py
def test_plain_text_mentioning_attachments_is_untouched() -> None:
    msg = _user("Attachments: none today")
    assert user_display_text(msg) == "Attachments: none today"
    assert user_attachments(msg) == []


TRANSLATOR_DEFAULT_PROMPT = "进入通用极速翻译模式：日语⇄简中，零寒暄。"


@pytest.fixture
async def translator_env(tmp_path, monkeypatch):
    """Gateway + AppChatbot (FakeLink) with a translator mode using the v0.3.7 prefs schema."""
    db_path = str(tmp_path / "kesoku.db")
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(
            db_path=db_path,
            roles_dir=str(tmp_path / "roles"),
            sessions_dir=str(tmp_path / "sessions"),
            skills_dir=str(tmp_path / "skills"),
        )
    )
    cfg.agent_working_dir = str(tmp_path)
    app_cfg = AppConfig(
        enabled=True,
        default_role="helper",
        allowed_models=["model-a"],
        modes=[
            AppModeConfig(id="chat", title="Chat"),
            AppModeConfig(
                id="translator",
                title="通用翻译",
                launch_prompt=TRANSLATOR_DEFAULT_PROMPT,
                prefs_schema={
                    "native_lang": "string",
                    "foreign_langs": "list",
                    "src_lang": "string",
                    "dst_lang": "string",
                    "formality": ["polite", "casual"],
                },
            ),
        ],
    )
    cfg.chatbots.app = [app_cfg]
    monkeypatch.setattr(kesoku.config, "_global_config", cfg)
    DatabaseManager(db_path).init_tables()
    (tmp_path / "roles" / "helper").mkdir(parents=True)
    (tmp_path / "roles" / "helper" / "intro.md").write_text("# helper\nA test role.\n", encoding="utf-8")
    ctx = KesokuContext(config=cfg)
    gw = Gateway(context=ctx)
    link = FakeLink()
    bot = AppChatbot("app", gw, app_cfg, link=link)
    await asyncio.to_thread(bot.policy.seed_defaults_sync)
    yield {"bot": bot, "link": link, "gw": gw, "cfg": cfg, "ctx": ctx, "tmp": tmp_path}


def _msg(sid: str, role: MessageRole, content: str, **kw: Any) -> Message:
    return Message(session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", content=content, role=role,
                   type=MessageType.TEXT, **kw)


def _finals(link: FakeLink, sid: str) -> list[dict[str, Any]]:
    return [e for e in link.of_type("message.final") if e["session_id"] == sid]


async def _start_translator(translator_env, monkeypatch, sid: str, prefs: dict[str, Any] | None = None) -> str:
    sent: list[str] = []

    async def fake_ingest(**kw: Any) -> dict[str, Any]:
        sent.append(kw["text"])
        return {"ok": True}

    monkeypatch.setattr(translator_env["bot"], "ingest_app_message", fake_ingest)
    payload: dict[str, Any] = {"session_id": sid, "mode_id": "translator"}
    if prefs is not None:
        payload["prefs"] = prefs
    reply = await rpc(translator_env, "mode.start", payload)
    assert reply["type"] == "mode.start.result", reply
    return sent[-1]


async def _send(translator_env: dict[str, Any], sid: str, cid: str, text: str = "hi") -> str:
    ack = await rpc(translator_env, "message.send", {"session_id": sid, "client_msg_id": cid, "text": text})
    assert ack["type"] == "message.ack", ack
    return str(ack["payload"]["message_id"])


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_missing_voice_first_attempt_self_heals_without_duplicate_final(translator_env) -> None:
    bot, link, gw = translator_env["bot"], translator_env["link"], translator_env["gw"]
    sid = await create_session(translator_env)
    staging = translator_env["tmp"] / "sessions" / sid
    staging.mkdir(parents=True, exist_ok=True)
    posted: list[Message] = []
    orig_post = gw.post

    async def capture(msg: Message) -> Message:
        posted.append(msg)
        return await orig_post(msg)

    gw.post = capture
    user = _msg(sid, MessageRole.USER, "穿毛衣", sender="dev-1", status=MessageStatus.PROCESSED)
    await gw.db.save_message(user)

    broken = _msg(sid, MessageRole.ASSISTANT, "哼，才不是为你穿的！\n[voice: sweater_tsun.wav]", sender="Kesoku",
                  parent_id=user.id, status=MessageStatus.PENDING)
    await gw.db.save_message(broken)
    await bot.render_outgoing_message(broken)

    # The broken first attempt is never shown: no message.final, no media.failed, status ERROR.
    assert _finals(link, sid) == []
    assert not [e for e in link.of_type("media.failed") if e["session_id"] == sid]
    assert (await gw.db.get_message(broken.id)).status == MessageStatus.ERROR
    assert link.of_type("stream.clear")[-1]["payload"]["turn_id"] == user.id
    healing = [m for m in posted if (m.metadata or {}).get("is_self_healing_retry")]
    assert len(healing) == 1 and healing[0].role == MessageRole.SYSTEM
    assert healing[0].metadata["failed_path"] == "sweater_tsun.wav"
    assert "[voice: sweater_tsun.wav]" in healing[0].content
    # Not reported as a generic delivery failure (no user-facing error text was sent).
    assert not [m for m in posted if m.role == MessageRole.ASSISTANT]

    # Healed retry: the voice clip now exists -> exactly one message.final + media.ready.
    (staging / "sweater_tsun.wav").write_bytes(b"RIFF....WAVEfmt ")
    healed = _msg(sid, MessageRole.ASSISTANT, "哼，才不是为你穿的！\n[voice: sweater_tsun.wav]", sender="Kesoku",
                  parent_id=healing[0].id, status=MessageStatus.PENDING)
    await gw.db.save_message(healed)
    await bot.render_outgoing_message(healed)
    finals = _finals(link, sid)
    assert [f["payload"]["message_id"] for f in finals] == [healed.id]
    assert [e for e in link.of_type("media.ready") if e["payload"]["message_id"] == healed.id]
    assert (await gw.db.get_message(healed.id)).status == MessageStatus.DELIVERED

    hist = (await rpc(translator_env, "history.page", {"session_id": sid}))["payload"]["messages"]
    ids = [m["message_id"] for m in hist]
    assert broken.id not in ids and healed.id in ids and user.id in ids


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_self_healing_retry_with_still_missing_media_is_delivered_once(translator_env) -> None:
    bot, link, gw = translator_env["bot"], translator_env["link"], translator_env["gw"]
    sid = await create_session(translator_env)
    healing = _msg(sid, MessageRole.SYSTEM, "File transmission failed", sender="System",
                   status=MessageStatus.PROCESSED, metadata={"is_self_healing_retry": True, "failed_path": "x.wav"})
    await gw.db.save_message(healing)
    retry = _msg(sid, MessageRole.ASSISTANT, "again\n[voice: still_missing.wav]", sender="Kesoku",
                 parent_id=healing.id, status=MessageStatus.PENDING)
    await gw.db.save_message(retry)
    await bot.render_outgoing_message(retry)
    # Retry turns skip the pre-flight (no infinite healing loop): the final is shown once.
    assert [f["payload"]["message_id"] for f in _finals(link, sid)] == [retry.id]
    assert (await gw.db.get_message(retry.id)).status == MessageStatus.DELIVERED


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_preflight_accepts_existing_files_and_deferred_sidecars(translator_env) -> None:
    bot, gw = translator_env["bot"], translator_env["gw"]
    sid = await create_session(translator_env)
    staging = translator_env["tmp"] / "sessions" / sid
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "ok.png").write_bytes(b"\x89PNG")
    (staging / "failed.wav.error.json").write_text(json.dumps({"error": "tts down"}), encoding="utf-8")
    user = _msg(sid, MessageRole.USER, "hi", sender="dev-1", status=MessageStatus.PROCESSED)
    await gw.db.save_message(user)
    msg = _msg(sid, MessageRole.ASSISTANT, "x", sender="Kesoku", parent_id=user.id, status=MessageStatus.PENDING)
    await gw.db.save_message(msg)
    await bot._preflight_self_healing(msg, [])
    await bot._preflight_self_healing(
        msg,
        [MediaJob(placeholder_id="ph_1", kind="file", ref="ok.png"),
         MediaJob(placeholder_id="ph_2", kind="voice", ref="failed.wav")],
    )
    with pytest.raises(DeliveryAbortedError):
        await bot._preflight_self_healing(msg, [MediaJob(placeholder_id="ph_3", kind="file", ref="nope.png")])


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_card_ref_resolvable_only_via_hydration_roots_never_aborts(translator_env) -> None:
    bot, link, gw = translator_env["bot"], translator_env["link"], translator_env["gw"]
    sid = await create_session(translator_env)
    art = translator_env["tmp"] / "games" / "pokemon_assistant" / "data" / "art"
    art.mkdir(parents=True, exist_ok=True)
    (art / "0987.webp").write_bytes(b"RIFF....WEBP")
    posted: list[Message] = []
    orig_post = gw.post

    async def capture(msg: Message) -> Message:
        posted.append(msg)
        return await orig_post(msg)

    gw.post = capture
    user = _msg(sid, MessageRole.USER, "show", sender="dev-1", status=MessageStatus.PROCESSED)
    await gw.db.save_message(user)
    content = "```card\nkind: info_card\ntitle: Flutter Mane\nimage: \"[file: pkmn/0987.webp]\"\n```"
    final = _msg(sid, MessageRole.ASSISTANT, content, sender="Kesoku", parent_id=user.id,
                 status=MessageStatus.PENDING)
    await gw.db.save_message(final)
    await bot.render_outgoing_message(final)

    finals = _finals(link, sid)
    assert [f["payload"]["message_id"] for f in finals] == [final.id]
    pid = finals[0]["payload"]["segments"][0]["media"]["image"]
    ready = [e for e in link.of_type("media.ready") if e["payload"]["placeholder_id"] == pid]
    assert ready and ready[0]["payload"]["ref"] == "pkmn/0987.webp"
    assert not [e for e in link.of_type("media.failed") if e["session_id"] == sid]
    assert not [m for m in posted if (m.metadata or {}).get("is_self_healing_retry")]
    assert (await gw.db.get_message(final.id)).status == MessageStatus.DELIVERED

    # An artwork ref that the hydrator cannot find either still triggers the pre-flight self-healing.
    missing = _msg(sid, MessageRole.ASSISTANT, content.replace("0987", "9999"), sender="Kesoku", parent_id=user.id,
                   status=MessageStatus.PENDING)
    await gw.db.save_message(missing)
    await bot.render_outgoing_message(missing)
    assert len(_finals(link, sid)) == 1
    assert (await gw.db.get_message(missing.id)).status == MessageStatus.ERROR


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_history_page_hides_error_assistant_but_keeps_error_user_messages(translator_env) -> None:
    gw = translator_env["gw"]
    sid = await create_session(translator_env)
    u = _msg(sid, MessageRole.USER, "user text", sender="dev-1", status=MessageStatus.ERROR)
    a_err = _msg(sid, MessageRole.ASSISTANT, "aborted", sender="Kesoku", status=MessageStatus.ERROR)
    a_ok = _msg(sid, MessageRole.ASSISTANT, "fine", sender="Kesoku", status=MessageStatus.DELIVERED)
    for m in (u, a_err, a_ok):
        await gw.db.save_message(m)
    visible = [m.id for m in translator_env["bot"].sync_db.get_messages_page(sid, visible_only=True)]
    assert u.id in visible and a_ok.id in visible and a_err.id not in visible
    everything = [m.id for m in translator_env["bot"].sync_db.get_messages_page(sid, visible_only=False)]
    assert a_err.id in everything


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_prefs_schema_list_rule(translator_env) -> None:
    sid = await create_session(translator_env)
    ok = await rpc(translator_env, "mode.prefs.set", {"session_id": sid,
                                           "prefs": {"native_lang": "zh-CN", "foreign_langs": ["ja", "en", "ko"]}})
    assert ok["payload"]["prefs"] == {"native_lang": "zh-CN", "foreign_langs": ["ja", "en", "ko"]}
    for bad in ("ja,en", ["x" * 41], [f"l{i}" for i in range(17)], ["ja", 3]):
        err = await rpc(translator_env, "mode.prefs.set", {"session_id": sid, "prefs": {"foreign_langs": bad}})
        assert err["type"] == "error" and err["payload"]["code"] == "BAD_REQUEST", bad
    edge = await rpc(translator_env, "mode.prefs.set", {"session_id": sid, "prefs": {"foreign_langs": ["y" * 40] * 16}})
    assert edge["type"] == "mode.prefs.set.result", edge
    bad_formality = await rpc(translator_env, "mode.prefs.set", {"session_id": sid, "prefs": {"formality": "rude"}})
    assert bad_formality["payload"]["code"] == "BAD_REQUEST"


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_prefs_sync_updates_translator_json(translator_env) -> None:
    sid = await create_session(translator_env)
    game = translator_env["tmp"] / "sessions" / sid / "game"
    game.mkdir(parents=True, exist_ok=True)
    (game / "translator.json").write_text(json.dumps({"src_lang": "ja", "dst_lang": "zh-CN", "turns": 3}),
                                          encoding="utf-8")
    prefs = {"native_lang": "en", "foreign_langs": ["ja", "fr"], "src_lang": "fr", "dst_lang": "en",
             "formality": "casual"}
    await rpc(translator_env, "mode.prefs.set", {"session_id": sid, "prefs": prefs})
    data = json.loads((game / "translator.json").read_text(encoding="utf-8"))
    assert data == {**prefs, "turns": 3}
    md = (translator_env["tmp"] / "sessions" / sid / "context" / "05-app-prefs.md").read_text(encoding="utf-8")
    assert "`native_lang`: en" in md


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_prefs_sync_does_not_create_translator_json(translator_env) -> None:
    sid = await create_session(translator_env)
    await rpc(translator_env, "mode.prefs.set", {"session_id": sid, "prefs": {"native_lang": "ja"}})
    assert not (translator_env["tmp"] / "sessions" / sid / "game" / "translator.json").exists()


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_translator_start_uses_default_prompt_without_language_prefs(translator_env, monkeypatch) -> None:
    sid = await create_session(translator_env)
    assert await _start_translator(translator_env, monkeypatch, sid) == TRANSLATOR_DEFAULT_PROMPT


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_translator_start_builds_prompt_from_stored_prefs_and_syncs_staging(translator_env, monkeypatch) -> None:
    sid = await create_session(translator_env)
    game = translator_env["tmp"] / "sessions" / sid / "game"
    game.mkdir(parents=True, exist_ok=True)
    (game / "translator.json").write_text(json.dumps({"src_lang": "ja", "dst_lang": "zh-CN"}), encoding="utf-8")
    await asyncio.to_thread(
        translator_env["bot"].sync_db.upsert_app_session_prefs, sid,
        mode_prefs={"native_lang": "zh-CN", "foreign_langs": ["en", "ja"], "src_lang": "zh-CN", "dst_lang": "ko"},
    )
    prompt = await _start_translator(translator_env, monkeypatch, sid)
    assert prompt == "进入通用极速翻译模式：母语 zh-CN ⇄ 常用外语 ko（常用外语：en、ja），零寒暄。"
    data = json.loads((game / "translator.json").read_text(encoding="utf-8"))
    assert data["dst_lang"] == "ko" and data["native_lang"] == "zh-CN" and data["foreign_langs"] == ["en", "ja"]


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_translator_start_payload_prefs_override_and_persist(translator_env, monkeypatch) -> None:
    sid = await create_session(translator_env)
    prompt = await _start_translator(translator_env, monkeypatch, sid, {"native_lang": "ja", "foreign_langs": ["en"]})
    assert prompt == "进入通用极速翻译模式：母语 ja ⇄ 常用外语 en，零寒暄。"
    stored = translator_env["bot"].sync_db.get_app_session_prefs(sid)["mode_prefs"]
    assert stored == {"native_lang": "ja", "foreign_langs": ["en"]}
    md = (translator_env["tmp"] / "sessions" / sid / "context" / "05-app-prefs.md").read_text(encoding="utf-8")
    assert "`native_lang`: ja" in md

    # Explicit foreign<->foreign pair (neither side is the native language).
    prompt = await _start_translator(translator_env, monkeypatch, sid, {"src_lang": "en", "dst_lang": "fr"})
    assert prompt == "进入通用极速翻译模式：母语 ja，当前外语对 en ⇄ fr，零寒暄。"

    for bad_prefs in ({"formality": "rude"}, {"unknown_key": "x"}, {"foreign_langs": "ja"}):
        bad = await rpc(translator_env, "mode.start", {"session_id": sid, "mode_id": "translator", "prefs": bad_prefs})
        assert bad["type"] == "error" and bad["payload"]["code"] == "BAD_REQUEST", bad_prefs
    not_obj = await rpc(translator_env, "mode.start", {"session_id": sid, "mode_id": "translator", "prefs": ["x"]})
    assert not_obj["payload"]["code"] == "BAD_REQUEST"
    # Rejected prefs never overwrite the stored ones.
    assert translator_env["bot"].sync_db.get_app_session_prefs(sid)["mode_prefs"]["native_lang"] == "ja"


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_translator_start_src_dst_only_defaults_native(translator_env, monkeypatch) -> None:
    sid = await create_session(translator_env)
    prompt = await _start_translator(translator_env, monkeypatch, sid, {"src_lang": "ja", "dst_lang": "zh-CN"})
    assert prompt == "进入通用极速翻译模式：母语 zh-CN ⇄ 常用外语 ja，零寒暄。"


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_translator_native_only_prompt_avoids_native_as_foreign(translator_env, monkeypatch) -> None:
    sid = await create_session(translator_env)
    prompt = await _start_translator(translator_env, monkeypatch, sid, {"native_lang": "ja"})
    assert prompt == "进入通用极速翻译模式：母语 ja ⇄ 常用外语 en，零寒暄。"


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_session_create_with_translator_mode_and_prefs(translator_env, monkeypatch) -> None:
    bot, gw = translator_env["bot"], translator_env["gw"]
    posted: list[Message] = []
    orig_post = gw.post

    async def capture(msg: Message) -> Message:
        posted.append(msg)
        return await orig_post(msg)

    monkeypatch.setattr(gw, "post", capture)
    prefs = {"native_lang": "zh-CN", "foreign_langs": ["ja", "en", "ko"], "src_lang": "ja", "dst_lang": "zh-CN"}
    # Default session.create defers the mode launch prompt until the user's first message so no turn is wasted.
    sid = await create_session(translator_env, mode="translator", prefs=prefs)
    stored = bot.sync_db.get_app_session_prefs(sid)
    assert stored["mode_prefs"] == prefs
    assert stored["pending_mode"] == "translator"
    assert bot.policy.current_mode(sid) == "translator"
    assert posted == []

    # When the user sends their first message, the mode launch prompt is automatically prepended to the LLM content
    # while display_text preserves only what the user typed.
    await rpc(translator_env, "message.send", {"session_id": sid, "client_msg_id": "m1", "text": "すみません"})
    assert len(posted) == 1
    assert "进入通用极速翻译模式：母语 zh-CN ⇄ 常用外语 ja（常用外语：ja、en、ko），零寒暄。" in posted[0].content
    assert "すみません" in posted[0].content
    assert posted[0].metadata["display_text"] == "すみません"
    assert posted[0].metadata["mode_hint"] == "translator"
    assert bot.sync_db.get_app_session_prefs(sid)["pending_mode"] is None

    # Switching mode before any message is sent (defer=True) updates pending_mode without posting a message.
    sid_defer = await create_session(translator_env)
    d_res = await rpc(
        translator_env, "mode.start", {"session_id": sid_defer, "mode_id": "translator", "prefs": prefs, "defer": True}
    )
    assert d_res["payload"]["deferred"] is True
    assert bot.policy.current_mode(sid_defer) == "translator"
    assert len(posted) == 1
    s_res = await rpc(translator_env, "mode.stop", {"session_id": sid_defer, "defer": True})
    assert s_res["payload"]["deferred"] is True
    assert bot.policy.current_mode(sid_defer) == "chat"

    # chat-mode session.create just persists prefs.
    sid2 = await create_session(translator_env, prefs={"native_lang": "en"})
    assert bot.sync_db.get_app_session_prefs(sid2)["mode_prefs"] == {"native_lang": "en"}
    assert len(posted) == 1

    # Invalid prefs are rejected before any session is created.
    before = len((await rpc(translator_env, "session.list", {"limit": 50}))["payload"]["sessions"])
    bad = await rpc(translator_env, "session.create", {"mode": "translator", "prefs": {"formality": "rude"}})
    assert bad["type"] == "error" and bad["payload"]["code"] == "BAD_REQUEST"
    assert len((await rpc(translator_env, "session.list", {"limit": 50}))["payload"]["sessions"]) == before


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_v038_turn_active_immediately_after_ingest_and_cleared_on_final(translator_env) -> None:
    bot, link, gw = translator_env["bot"], translator_env["link"], translator_env["gw"]
    sid = await create_session(translator_env)
    mid = await _send(translator_env, sid, "t1")
    # Active before any token: snapshot / resume can show the typing indicator, and devices got a stream.clear.
    assert bot.active_turn_snapshot(sid) == {"turn_id": mid, "round_id": 0, "preview_text": ""}
    clear = [e for e in link.of_type("stream.clear") if e["session_id"] == sid][-1]
    assert clear["payload"] == {"turn_id": mid, "round_id": 0}
    assert "seq" not in clear
    user_seq = [e for e in link.of_type("message.user") if e["session_id"] == sid][-1]["seq"]
    assert link.sent.index(clear) > link.sent.index(
        next(e for e in link.sent if e.get("type") == "message.user" and e.get("seq") == user_seq)
    )
    snap = await rpc(translator_env, "session.open", {"session_id": sid})
    assert snap["type"] == "session.snapshot", snap
    assert snap["payload"]["active_turn"]["turn_id"] == mid
    done = await rpc(translator_env, "sync.replay", {"session_id": sid, "after_seq": 0})
    assert done["type"] == "sync.done" and done["payload"]["active_turn"]["turn_id"] == mid

    reply = _msg(sid, MessageRole.ASSISTANT, "hello", sender="Kesoku", parent_id=mid, status=MessageStatus.PENDING)
    await gw.db.save_message(reply)
    await bot.render_outgoing_message(reply)
    assert bot.active_turn_snapshot(sid) is None
    done = await rpc(translator_env, "sync.replay", {"session_id": sid, "after_seq": 0})
    # Explicit null on the last page so devices drop a stale typing indicator.
    assert "active_turn" in done["payload"] and done["payload"]["active_turn"] is None


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_v038_turn_events_reactivate_and_self_healing_final_clears(translator_env) -> None:
    bot, link, gw = translator_env["bot"], translator_env["link"], translator_env["gw"]
    sid = await create_session(translator_env)
    mid = await _send(translator_env, sid, "t2", "穿毛衣")
    bot._active_turns.pop(sid)  # e.g. engine restarted mid-turn
    thought = _msg(sid, MessageRole.ASSISTANT, "thinking…", sender="Kesoku", parent_id=mid,
                   status=MessageStatus.PENDING)
    thought.type = MessageType.THOUGHT
    await bot._emit_turn_event(thought)
    assert bot.active_turn_snapshot(sid)["turn_id"] == mid

    broken = _msg(sid, MessageRole.ASSISTANT, "哼！\n[voice: sweater.wav]", sender="Kesoku", parent_id=mid,
                  status=MessageStatus.PENDING)
    await gw.db.save_message(broken)
    await bot.render_outgoing_message(broken)
    assert _finals(link, sid) == []
    assert bot.active_turn_snapshot(sid) is not None  # still typing while the self-healing retry runs
    healing_msg = _msg(sid, MessageRole.SYSTEM, "File transmission failed", sender="System",
                       status=MessageStatus.PROCESSED, metadata={"is_self_healing_retry": True})
    await gw.db.save_message(healing_msg)
    healed = _msg(sid, MessageRole.ASSISTANT, "哼！", sender="Kesoku", parent_id=healing_msg.id,
                  status=MessageStatus.PENDING)
    await gw.db.save_message(healed)
    await bot.render_outgoing_message(healed)
    assert [f["payload"]["turn_id"] for f in _finals(link, sid)] == [healing_msg.id]
    # The retry finalized under a different turn id, yet the session is no longer marked as typing.
    assert bot.active_turn_snapshot(sid) is None


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_v038_notification_final_keeps_turn_and_turn_stop_clears(translator_env) -> None:
    bot, gw = translator_env["bot"], translator_env["gw"]
    sid = await create_session(translator_env)
    mid = await _send(translator_env, sid, "t3")
    note = _msg(sid, MessageRole.ASSISTANT, "📌 note", sender="Notification", parent_id=mid,
                status=MessageStatus.PENDING)
    await gw.db.save_message(note)
    await bot.render_outgoing_message(note)
    assert bot.active_turn_snapshot(sid) is not None
    stop = await rpc(translator_env, "turn.stop", {"session_id": sid})
    assert stop["payload"]["stopped"] is True
    assert bot.active_turn_snapshot(sid) is None


# origin: tests/gateway/chatbot/app/test_app_v037_delivery_translator.py
async def test_v038_delivery_failure_clears_active_turn(translator_env, monkeypatch) -> None:
    bot, gw = translator_env["bot"], translator_env["gw"]
    sid = await create_session(translator_env)
    mid = await _send(translator_env, sid, "t4")

    async def boom(message: Message) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(bot, "_deliver_final", boom)
    reply = _msg(sid, MessageRole.ASSISTANT, "x", sender="Kesoku", parent_id=mid, status=MessageStatus.PENDING)
    await gw.db.save_message(reply)
    await bot.render_outgoing_message(reply)
    assert bot.active_turn_snapshot(sid) is None


# ---------------------------------------------------------------------------- v0.3.13 session notes & translator voice


async def test_session_notes_get_put_and_live_state_sync(env) -> None:
    bot, link = env["bot"], env["link"]
    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid

    # Missing notes.md returns empty content (never NOT_FOUND).
    initial = await rpc(env, "session.notes.get", {"session_id": sid})
    assert initial["type"] == "session.notes.get.result"
    assert initial["payload"] == {"session_id": sid, "content": "", "bytes": 0, "updated_at": None}

    # Saving notes writes atomically and emits incremental `state.sync` with `notes.md`.
    note_text = "# Session Rules\n- Keep responses concise 📌\n"
    put_res = await rpc(env, "session.notes.put", {"session_id": sid, "content": note_text})
    assert put_res["type"] == "session.notes.put.result"
    assert put_res["payload"]["session_id"] == sid
    assert put_res["payload"]["content"] == note_text
    assert put_res["payload"]["bytes"] == len(note_text.encode("utf-8"))
    assert isinstance(put_res["payload"]["updated_at"], float)

    syncs = [e for e in link.of_type("state.sync") if e["session_id"] == sid]
    assert syncs and syncs[-1]["payload"]["files"] == {"notes.md": note_text}
    assert syncs[-1]["payload"]["removed"] == []

    # `session.notes.get` and `session.open` reflect the persisted notes.md.
    got = await rpc(env, "session.notes.get", {"session_id": sid})
    assert got["payload"]["content"] == note_text
    assert got["payload"]["bytes"] == len(note_text.encode("utf-8"))
    assert got["payload"]["updated_at"] == pytest.approx(put_res["payload"]["updated_at"])

    snap = (await rpc(env, "session.open", {"session_id": sid}))["payload"]
    assert snap["state_files"]["notes.md"] == note_text

    # Simulated AI `update_file` modifying `$STAGING_DIR/notes.md` emits a new `state.sync` upon scan.
    agent_note = "# Session Rules\n- Updated by agent tool call\n- Track quest clues\n"
    (staging / "notes.md").write_text(agent_note, encoding="utf-8")
    scan_payload = await bot.state_watcher.scan(sid)
    assert scan_payload is not None
    assert scan_payload["files"] == {"notes.md": agent_note}

    # Whitespace-only `session.notes.put` unlinks `notes.md` and emits `removed: ["notes.md"]`.
    cleared = await rpc(env, "session.notes.put", {"session_id": sid, "content": "   \n\t  "})
    assert cleared["type"] == "session.notes.put.result"
    assert cleared["payload"]["content"] == ""
    assert cleared["payload"]["bytes"] == 0
    assert isinstance(cleared["payload"]["updated_at"], float)
    assert not (staging / "notes.md").exists()

    last_sync = [e for e in link.of_type("state.sync") if e["session_id"] == sid][-1]["payload"]
    assert last_sync["removed"] == ["notes.md"]
    assert last_sync["files"] == {}

    after_clear = await rpc(env, "session.notes.get", {"session_id": sid})
    assert after_clear["payload"] == {"session_id": sid, "content": "", "bytes": 0, "updated_at": None}


async def test_session_notes_limits_validation_and_symlink_jail(env) -> None:
    bot = env["bot"]
    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid
    staging.mkdir(parents=True, exist_ok=True)

    # Validation errors on bad session_id, missing session, non-string content, and >64 KiB payloads.
    assert (await rpc(env, "session.notes.get", {"session_id": "../bad"}))["payload"]["code"] == "BAD_REQUEST"
    assert (await rpc(env, "session.notes.get", {"session_id": "deadbeef"}))["payload"]["code"] == "NOT_FOUND"
    assert (await rpc(env, "session.notes.put", {"session_id": "deadbeef", "content": "hi"}))["payload"]["code"] == (
        "NOT_FOUND"
    )
    for bad_content in (None, 42, ["list"], {"a": 1}):
        err = await rpc(env, "session.notes.put", {"session_id": sid, "content": bad_content})
        assert err["type"] == "error" and err["payload"]["code"] == "BAD_REQUEST"

    # Full 64 KiB succeeds and is NOT truncated at the 8 KiB context/*.md cap.
    max_bytes = LIMITS["SESSION_NOTES_MAX_BYTES"]
    assert max_bytes == 64 * 1024 and NOTES_TEXT_MAX_BYTES == 64 * 1024
    exact_64k = "a" * max_bytes
    ok_64k = await rpc(env, "session.notes.put", {"session_id": sid, "content": exact_64k})
    assert ok_64k["type"] == "session.notes.put.result" and ok_64k["payload"]["bytes"] == max_bytes
    snap = (await rpc(env, "session.open", {"session_id": sid}))["payload"]
    assert len(snap["state_files"]["notes.md"]) == max_bytes

    # 64 KiB + 1 byte (ASCII and multibyte UTF-8) is rejected.
    too_big_ascii = await rpc(env, "session.notes.put", {"session_id": sid, "content": "a" * (max_bytes + 1)})
    assert too_big_ascii["payload"]["code"] == "BAD_REQUEST"
    too_big_utf8 = await rpc(env, "session.notes.put", {"session_id": sid, "content": "中" * ((max_bytes // 3) + 2)})
    assert too_big_utf8["payload"]["code"] == "BAD_REQUEST"

    # Oversized notes.md written directly on disk is safely truncated to <= 64 KiB UTF-8 on read & state_watcher.
    (staging / "notes.md").write_text("中" * 25000, encoding="utf-8")
    trunc_get = (await rpc(env, "session.notes.get", {"session_id": sid}))["payload"]
    assert 0 < trunc_get["bytes"] <= max_bytes
    assert trunc_get["bytes"] == len(trunc_get["content"].encode("utf-8"))
    scan = await bot.state_watcher.scan(sid)
    assert scan is not None and len(scan["files"]["notes.md"].encode("utf-8")) <= max_bytes

    # Symlink escape outside staging is ignored on get/watcher and rejected on put without touching target.
    outside = env["tmp"] / "outside_secret.md"
    outside.write_text("TOP SECRET", encoding="utf-8")
    os.unlink(staging / "notes.md")
    os.symlink(outside, staging / "notes.md")

    sym_get = (await rpc(env, "session.notes.get", {"session_id": sid}))["payload"]
    assert sym_get == {"session_id": sid, "content": "", "bytes": 0, "updated_at": None}
    sym_scan = await bot.state_watcher.scan(sid)
    assert sym_scan is not None and "notes.md" in sym_scan["removed"]
    sym_put = await rpc(env, "session.notes.put", {"session_id": sid, "content": "attempt overwrite"})
    assert sym_put["type"] == "error" and sym_put["payload"]["code"] == "BAD_REQUEST"
    assert outside.read_text(encoding="utf-8") == "TOP SECRET"


async def test_state_watcher_notes_md_never_starved_by_max_state_files(env) -> None:
    bot = env["bot"]
    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid
    game_dir = staging / "game"
    ctx_dir = staging / "context"
    game_dir.mkdir(parents=True, exist_ok=True)
    ctx_dir.mkdir(parents=True, exist_ok=True)

    (staging / "notes.md").write_text("pinned priority note", encoding="utf-8")
    for idx in range(MAX_STATE_FILES + 6):
        (game_dir / f"g{idx:03d}.json").write_text(json.dumps({"i": idx}), encoding="utf-8")
    (ctx_dir / "10-extra.md").write_text("extra", encoding="utf-8")

    payload = await bot.state_watcher.scan(sid)
    assert payload is not None
    assert payload["files"]["notes.md"] == "pinned priority note"
    assert len(payload["files"]) == MAX_STATE_FILES


async def test_translator_pure_audio_inbound_prompt_preserves_clean_user_bubble(translator_env, monkeypatch) -> None:
    bot, link, gw = translator_env["bot"], translator_env["link"], translator_env["gw"]
    posted: list[Message] = []
    orig_post = gw.post

    async def capture(msg: Message) -> Message:
        posted.append(msg)
        return await orig_post(msg)

    monkeypatch.setattr(gw, "post", capture)

    async def fake_get(url: str, max_bytes: int) -> bytes:
        key = url.split("store.invalid/", 1)[1].split("?", 1)[0]
        return link.objects[key]["data"]

    bot.media._http_get = fake_get

    def _stage_blob(sid: str, ext: str, data: bytes = b"FAKEAUDIO") -> str:
        obj = f"media/{sid}/{uuid.uuid4()}.{ext}"
        link.objects[obj] = {"data": data, "device": "dev-1", "issued_at": time.time()}
        return obj

    # 1. Deferred translator session (`pending_mode == "translator"`): first message is pure audio (`text == ""`).
    prefs = {"native_lang": "zh-CN", "foreign_langs": ["ja", "en"], "src_lang": "ja", "dst_lang": "zh-CN"}
    sid = await create_session(translator_env, mode="translator", prefs=prefs)
    obj1 = _stage_blob(sid, "webm")
    ack1 = await rpc(
        translator_env,
        "message.send",
        {
            "session_id": sid,
            "client_msg_id": "aud-1",
            "text": "",
            "attachments": [{"object": obj1, "mime": "audio/webm;codecs=opus", "filename": "voice1.webm", "bytes": 9}],
        },
    )
    assert ack1["type"] == "message.ack", ack1
    msg1 = posted[-1]
    assert "进入通用极速翻译模式：母语 zh-CN ⇄ 常用外语 ja（常用外语：ja、en），零寒暄。" in msg1.content
    assert TRANSLATOR_PURE_AUDIO_PROMPT in msg1.content
    assert "Attachments:\n[Attachment: voice1.webm" in msg1.content
    assert msg1.metadata["display_text"] == ""
    user_evt1 = [e for e in link.of_type("message.user") if e["session_id"] == sid][-1]["payload"]
    assert user_evt1["text"] == ""
    assert len(user_evt1["attachments"]) == 1
    hist1 = (await rpc(translator_env, "history.page", {"session_id": sid}))["payload"]["messages"]
    assert hist1[-1]["text"] == "" and len(hist1[-1]["attachments"]) == 1

    # 2. Active translator session (`game/games.json` active, `pending_mode` already cleared): pure audio turn.
    game_dir = translator_env["tmp"] / "sessions" / sid / "game"
    game_dir.mkdir(parents=True, exist_ok=True)
    (game_dir / "games.json").write_text(json.dumps([{"game": "translator", "active": True}]), encoding="utf-8")
    obj2 = _stage_blob(sid, "m4a")
    ack2 = await rpc(
        translator_env,
        "message.send",
        {
            "session_id": sid,
            "client_msg_id": "aud-2",
            "text": "",
            "attachments": [{"object": obj2, "mime": "AUDIO/MP4", "filename": "voice2.m4a", "bytes": 9}],
        },
    )
    assert ack2["type"] == "message.ack", ack2
    msg2 = posted[-1]
    assert msg2.content.startswith(f"{TRANSLATOR_PURE_AUDIO_PROMPT}\n\nAttachments:\n")
    assert "进入通用极速翻译模式" not in msg2.content
    assert msg2.metadata["display_text"] == ""
    user_evt2 = [e for e in link.of_type("message.user") if e["session_id"] == sid][-1]["payload"]
    assert user_evt2["text"] == ""

    # 3. Audio WITH explicit text in translator mode keeps user text without injecting the pure-audio prompt.
    obj3 = _stage_blob(sid, "wav")
    await rpc(
        translator_env,
        "message.send",
        {
            "session_id": sid,
            "client_msg_id": "aud-3",
            "text": "请翻成敬语",
            "attachments": [{"object": obj3, "mime": "audio/wav", "filename": "voice3.wav", "bytes": 9}],
        },
    )
    assert posted[-1].content.startswith("请翻成敬语\n\nAttachments:\n")
    assert TRANSLATOR_PURE_AUDIO_PROMPT not in posted[-1].content
    assert posted[-1].metadata["display_text"] == "请翻成敬语"

    # 4. Pure image attachment in translator mode does NOT inject the pure-audio prompt.
    obj4 = _stage_blob(sid, "jpg", b"\xff\xd8\xff")
    await rpc(
        translator_env,
        "message.send",
        {
            "session_id": sid,
            "client_msg_id": "img-1",
            "text": "",
            "attachments": [{"object": obj4, "mime": "image/jpeg", "filename": "menu.jpg", "bytes": 3}],
        },
    )
    assert posted[-1].content.startswith("Attachments:\n")
    assert TRANSLATOR_PURE_AUDIO_PROMPT not in posted[-1].content

    # 5. Pure audio in plain `chat` mode (and in a cross-channel session) does NOT inject the translator prompt.
    sid_chat = await create_session(translator_env)
    obj5 = _stage_blob(sid_chat, "webm")
    await rpc(
        translator_env,
        "message.send",
        {
            "session_id": sid_chat,
            "client_msg_id": "chat-aud-1",
            "text": "",
            "attachments": [{"object": obj5, "mime": "audio/webm", "filename": "chat.webm", "bytes": 9}],
        },
    )
    assert posted[-1].content.startswith("Attachments:\n")
    assert TRANSLATOR_PURE_AUDIO_PROMPT not in posted[-1].content

    cross_sess = await gw.create_session(title="Cross", chatbot_id="discord", channel_id="d-chan", role="helper")
    cross_game = translator_env["tmp"] / "sessions" / cross_sess.workspace_name / "game"
    cross_game.mkdir(parents=True, exist_ok=True)
    (cross_game / "games.json").write_text(json.dumps([{"game": "translator", "active": True}]), encoding="utf-8")
    obj6 = _stage_blob(cross_sess.id, "webm")
    await rpc(
        translator_env,
        "message.send",
        {
            "session_id": cross_sess.id,
            "client_msg_id": "cross-aud-1",
            "text": "",
            "attachments": [{"object": obj6, "mime": "audio/webm", "filename": "cross.webm", "bytes": 9}],
        },
    )
    assert posted[-1].content.startswith("Attachments:\n")
    assert TRANSLATOR_PURE_AUDIO_PROMPT not in posted[-1].content


# ---------------------------------------------------------------------------------------------- v0.3.14


async def test_session_create_and_rename_lock_manual_titles(env) -> None:
    """Custom create titles and renames are stored with `title_source = "manual"`; defaults stay unlocked."""
    db = env["gw"].db
    default_sid = await create_session(env)
    assert (await db.get_session(default_sid)).title_source == ""
    placeholder_sid = await create_session(env, title="新对话")
    assert (await db.get_session(placeholder_sid)).title_source == ""
    custom_sid = await create_session(env, title="  My Trip  ")
    sess = await db.get_session(custom_sid)
    assert (sess.title, sess.title_source) == ("My Trip", "manual")

    reply = await rpc(env, "session.rename", {"session_id": default_sid, "title": "Renamed"})
    assert reply["payload"]["title"] == "Renamed"
    sess = await db.get_session(default_sid)
    assert (sess.title, sess.title_source) == ("Renamed", "manual")


async def test_media_hydrate_reuses_uploads_across_sessions(env, monkeypatch) -> None:
    """Repeated hydration of an unchanged file re-signs the cached object instead of re-uploading bytes."""
    link: FakeLink = env["link"]
    sid1 = await create_session(env)
    sid2 = await create_session(env)
    avatar = env["tmp"] / "characters" / "roster" / "images" / "a.png"
    avatar.parent.mkdir(parents=True)
    avatar.write_bytes(b"\x89PNG avatar")
    ref = "characters/roster/images/a.png"

    first = (await rpc(env, "media.hydrate", {"session_id": sid1, "ref": ref}))["payload"]
    assert len(link.objects) == 1
    second = (await rpc(env, "media.hydrate", {"session_id": sid2, "ref": ref, "placeholder_id": "ph1"}))["payload"]
    assert len(link.objects) == 1
    assert second["object"] == first["object"] and second["url"].endswith("?sig=1")
    assert second["mime"] == "image/png" and second["bytes"] == len(b"\x89PNG avatar")
    assert second["session_id"] == sid2 and second["placeholder_id"] == "ph1"

    # A changed file (new size/mtime) is uploaded again.
    avatar.write_bytes(b"\x89PNG avatar v2!")
    third = (await rpc(env, "media.hydrate", {"session_id": sid1, "ref": ref}))["payload"]
    assert third["object"] != first["object"] and len(link.objects) == 2

    # Expired entries and objects that can no longer be signed fall back to a fresh upload.
    # Hydrate reuse lives in the MediaManager's single LRU under a session-independent scope.
    cache = env["bot"].media._upload_cache
    assert all(k[3] == commands_mod.HYDRATE_CACHE_SCOPE for k in cache)
    key = list(cache)[-1]  # most recently used = the current file version
    cache[key]["uploaded_at"] = time.time() - commands_mod.HYDRATE_UPLOAD_CACHE_TTL_SECONDS - 1
    await rpc(env, "media.hydrate", {"session_id": sid2, "ref": ref})
    assert len(link.objects) == 3
    link.objects.clear()
    healed = (await rpc(env, "media.hydrate", {"session_id": sid2, "ref": ref}))["payload"]
    assert healed["object"] in link.objects and cache[key]["object"] == healed["object"]

    # The cache is a bounded LRU.
    monkeypatch.setattr(media_mod, "UPLOAD_CACHE_MAX_ENTRIES", 2)
    staging = env["tmp"] / "sessions" / sid1
    staging.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (staging / f"f{i}.png").write_bytes(b"x" * (i + 1))
        await rpc(env, "media.hydrate", {"session_id": sid1, "ref": f"f{i}.png"})
    assert len(cache) == 2 and all(k[0].endswith(("f1.png", "f2.png")) for k in cache)


async def test_history_question_choice_items_from_workspace_registry(env) -> None:
    """Session history enriches character-candidate questions with avatar cards from `$AWD/characters`."""
    tmp = env["tmp"]
    reg = tmp / "characters" / "roster"
    (reg / "images").mkdir(parents=True)
    (reg / "images" / "a.jpg").write_bytes(b"a")
    (reg / "images" / "b.jpg").write_bytes(b"b")
    (reg / "characters.json").write_text(
        json.dumps(
            [
                {"name": "Alpha One", "aliases": ["阿尔法"], "image": "characters/roster/images/a.jpg"},
                {"name": "Beta Two", "aliases": ["贝塔"], "image": "characters/roster/images/b.jpg"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sid = await create_session(env)
    gw = env["gw"]
    await gw.db.save_message(
        Message(
            session_id=sid,
            chatbot_id="app",
            channel_id=f"app:{sid}",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            content=(
                "```ui\ntype: select\nprompt: 选谁？\noptions:\n"
                "  - 阿尔法（队长）\n  - 贝塔（副手）\n  - 换一批\n```"
            ),
            status=MessageStatus.DELIVERED,
        )
    )
    hist = (await rpc(env, "history.page", {"session_id": sid}))["payload"]["messages"]
    q = next(s for m in hist if m["role"] == "assistant" for s in m["segments"] if s["type"] == "question")
    assert q["layout"] == "cards"
    assert q["choice_items"][0]["avatar"] == "characters/roster/images/a.jpg"
    assert q["choice_items"][2] == {"value": "换一批", "name": "换一批"}


async def test_reply_channel_stamped_server_side_from_resolved_mode(env) -> None:
    """The adapter stamps `metadata.reply_channel` only from the server-resolved mode; client keys are dropped."""
    bot, gw = env["bot"], env["gw"]
    bot.app_config.modes.append(
        AppModeConfig(id="guided", title="Guided", launch_prompt="begin guided", reply_channel=True)
    )
    sid = await create_session(env)
    posted: list[Message] = []
    orig_post = gw.post

    async def capture(msg: Message) -> Message:
        posted.append(msg)
        return await orig_post(msg)

    gw.post = capture

    # Chat mode: a client-supplied key is dropped, and an llm_override carrying it is rejected outright.
    await bot.ingest_app_message(
        session_id=sid, device_id="dev-1", client_msg_id="c-chat", text="hello", attachments=[],
        llm_override=None, extra_metadata={"reply_channel": True},
    )
    assert "reply_channel" not in posted[-1].metadata
    bad = await rpc(
        env,
        "message.send",
        {"session_id": sid, "client_msg_id": "c-ovr", "text": "x", "llm_override": {"reply_channel": True}},
    )
    assert bad["payload"]["code"] == "BAD_REQUEST"
    rejected = await rpc(env, "model.set", {"session_id": sid, "llm_override": {"reply_channel": False}})
    assert rejected["payload"]["code"] == "BAD_REQUEST"

    # Launching the opted-in mode stamps the flag on the launch message.
    await rpc(env, "mode.start", {"session_id": sid, "mode_id": "guided"})
    assert posted[-1].metadata["mode_hint"] == "guided"
    assert posted[-1].metadata["reply_channel"] is True

    # While the mode is active, every message is stamped; a model switch or client key cannot turn it off.
    _write_games(env["tmp"], sid, [{"id": "guided#1", "game": "guided", "active": True}])
    await rpc(
        env,
        "message.send",
        {"session_id": sid, "client_msg_id": "c-g1", "text": "next", "llm_override": {"model_name": "model-a"}},
    )
    assert posted[-1].metadata["reply_channel"] is True
    assert posted[-1].metadata["llm_override"] == {"model_name": "model-a"}
    await bot.ingest_app_message(
        session_id=sid, device_id="dev-1", client_msg_id="c-g2", text="again", attachments=[],
        llm_override=None, extra_metadata={"reply_channel": False},
    )
    assert posted[-1].metadata["reply_channel"] is True

    # Stopping the mode (mode_hint="chat") turns it off again.
    await rpc(env, "mode.stop", {"session_id": sid})
    assert posted[-1].metadata["mode_hint"] == "chat"
    assert "reply_channel" not in posted[-1].metadata


# ---------------------------------------------------------------------------------------------- audit fixes


class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.parametrize(
    ("exc", "code", "message"),
    [
        (TimeoutError(), "INTERNAL", "request timed out"),
        (ConnectionError("engine link lost"), "ENGINE_OFFLINE", "relay link is not connected"),
    ],
)
async def test_expected_request_failures_map_to_codes_without_traceback(env, monkeypatch, exc, code, message) -> None:
    """Timeouts / link loss reply with an existing wire code and log one WARNING line without exc_info."""
    bot = env["bot"]

    async def failing(ctx, payload):
        raise exc

    monkeypatch.setitem(bot._handlers, "session.list", failing)
    collector = _RecordCollector()
    adapter_logger = logging.getLogger("kesoku.gateway.chatbot.app.adapter")
    adapter_logger.addHandler(collector)
    try:
        reply = await rpc(env, "session.list", {})
    finally:
        adapter_logger.removeHandler(collector)
    assert reply["type"] == "error"
    assert reply["payload"]["code"] == code and reply["payload"]["message"] == message
    assert [r.levelno for r in collector.records] == [logging.WARNING]
    assert collector.records[0].exc_info is None


async def test_unexpected_request_failure_keeps_error_traceback(env, monkeypatch) -> None:
    bot = env["bot"]

    async def failing(ctx, payload):
        raise RuntimeError("boom")

    monkeypatch.setitem(bot._handlers, "session.list", failing)
    collector = _RecordCollector()
    adapter_logger = logging.getLogger("kesoku.gateway.chatbot.app.adapter")
    adapter_logger.addHandler(collector)
    try:
        reply = await rpc(env, "session.list", {})
    finally:
        adapter_logger.removeHandler(collector)
    assert reply["payload"]["code"] == "INTERNAL" and reply["payload"]["message"] == "internal error"
    assert [r.levelno for r in collector.records] == [logging.ERROR]
    assert collector.records[0].exc_info is not None


async def test_media_hydrate_single_flight_shares_one_upload(env, monkeypatch) -> None:
    """N concurrent hydrates of the same unchanged file perform exactly one upload."""
    link: FakeLink = env["link"]
    sids = [await create_session(env) for _ in range(3)]
    art = env["tmp"] / "characters" / "roster" / "images" / "shared.png"
    art.parent.mkdir(parents=True)
    art.write_bytes(b"\x89PNG shared")
    gate = asyncio.Event()
    uploads: list[str] = []
    real_binary = link.request_binary

    async def slow_binary(frame: bytes, request_id: str, timeout: float | None = None) -> dict[str, Any]:
        uploads.append(request_id)
        await gate.wait()
        return await real_binary(frame, request_id, timeout)

    monkeypatch.setattr(link, "request_binary", slow_binary)
    ref = "characters/roster/images/shared.png"
    tasks = [
        asyncio.create_task(rpc(env, "media.hydrate", {"session_id": sids[i % 3], "ref": ref})) for i in range(6)
    ]
    await asyncio.sleep(0.2)
    assert len(uploads) == 1
    gate.set()
    replies = await asyncio.gather(*tasks)
    assert len(uploads) == 1 and len(link.objects) == 1
    assert all(r["type"] == "media.ready" for r in replies)
    assert {r["payload"]["object"] for r in replies} == set(link.objects)
    assert env["bot"].requests._hydrate_inflight == {}


async def test_media_hydrate_single_flight_survives_waiter_cancellation(env, monkeypatch) -> None:
    link: FakeLink = env["link"]
    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "c.png").write_bytes(b"c")
    gate = asyncio.Event()
    real_binary = link.request_binary

    async def slow_binary(frame: bytes, request_id: str, timeout: float | None = None) -> dict[str, Any]:
        await gate.wait()
        return await real_binary(frame, request_id, timeout)

    monkeypatch.setattr(link, "request_binary", slow_binary)
    first = asyncio.create_task(rpc(env, "media.hydrate", {"session_id": sid, "ref": "c.png"}))
    await asyncio.sleep(0.1)
    second = asyncio.create_task(rpc(env, "media.hydrate", {"session_id": sid, "ref": "c.png"}))
    await asyncio.sleep(0.1)
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    gate.set()
    reply = await second
    assert reply["type"] == "media.ready" and len(link.objects) == 1


async def test_media_hydrate_uploads_are_bounded(env, monkeypatch) -> None:
    """At most `HYDRATE_MAX_CONCURRENCY` distinct hydrate uploads run concurrently."""
    link: FakeLink = env["link"]
    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid
    staging.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (staging / f"b{i}.png").write_bytes(b"y" * (i + 1))
    active = 0
    peak = 0
    real_binary = link.request_binary

    async def tracked_binary(frame: bytes, request_id: str, timeout: float | None = None) -> dict[str, Any]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.05)
            return await real_binary(frame, request_id, timeout)
        finally:
            active -= 1

    monkeypatch.setattr(link, "request_binary", tracked_binary)
    replies = await asyncio.gather(
        *(rpc(env, "media.hydrate", {"session_id": sid, "ref": f"b{i}.png"}) for i in range(10))
    )
    assert all(r["type"] == "media.ready" for r in replies)
    assert len(link.objects) == 10
    assert peak == commands_mod.HYDRATE_MAX_CONCURRENCY


async def test_media_hydrate_offline_fails_fast(env) -> None:
    link: FakeLink = env["link"]
    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "o.png").write_bytes(b"o")
    link.connected = False
    reply = await rpc(env, "media.hydrate", {"session_id": sid, "ref": "o.png"})
    assert reply["type"] == "error" and reply["payload"]["code"] == "ENGINE_OFFLINE"


async def test_session_list_ordered_by_updated_at_after_final_and_cross_channel_send(env) -> None:
    bot, gw = env["bot"], env["gw"]
    s_old = await create_session(env, title="Old")
    s_new = await create_session(env, title="New")
    await gw.db.update_session_updated_at(s_old, 100.0)
    await gw.db.update_session_updated_at(s_new, 200.0)

    # Assistant final delivery bumps updated_at so s_old moves ahead of s_new.
    final = Message(
        session_id=s_old,
        chatbot_id="app",
        channel_id=f"app:{s_old}",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content="reply",
        timestamp=300.0,
        status=MessageStatus.PENDING,
    )
    await gw.db.save_message(final)
    await bot.render_outgoing_message(final)
    listed = (await rpc(env, "session.list", {}))["payload"]["sessions"]
    assert [s["session_id"] for s in listed[:2]] == [s_old, s_new]
    assert listed[0]["updated_at"] == pytest.approx(300.0)

    # Cross-channel session replied to from app also bumps updated_at to the top.
    cross = await gw.create_session(title="Discord", chatbot_id="discord", channel_id="chan-upd", role="helper")
    await gw.db.update_session_updated_at(cross.id, 50.0)
    ack = await rpc(env, "message.send", {"session_id": cross.id, "client_msg_id": "xc1", "text": "bump"})
    assert ack["type"] == "message.ack"
    listed_all = (await rpc(env, "session.list", {"all_channels": True}))["payload"]["sessions"]
    assert listed_all[0]["session_id"] == cross.id
    assert listed_all[0]["updated_at"] > 300.0


# ---------------------------------------------------------------------------------- hydrate, mode defaults & roster


async def test_character_hydrate_prefers_avatar_and_awd_relative_paths(env) -> None:
    tmp = env["tmp"]
    sid = await create_session(env)
    (tmp / "secret.png").write_bytes(b"SECRET")

    # Malformed subdir must not break valid subdirs.
    bad_dir = tmp / "characters" / "00_broken"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "characters.json").write_text("{broken json", encoding="utf-8")

    group_dir = tmp / "characters" / "01_cast"
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "hero_avatar.jpg").write_bytes(b"AVATAR_BYTES")
    (group_dir / "hero_full.png").write_bytes(b"FULL_IMAGE_BYTES")
    (group_dir / "legacy.png").write_bytes(b"LEGACY_SUBDIR_BYTES")
    os.symlink(tmp / "secret.png", group_dir / "escape.png")

    (group_dir / "characters.json").write_text(
        json.dumps(
            [
                {
                    "name": "Heroine Alpha",
                    "trigger": "Alpha Trigger",
                    "aliases": ["Alpha"],
                    "avatar": "characters/01_cast/hero_avatar.jpg",
                    "image": "characters/01_cast/hero_full.png",
                },
                {
                    "name": "Legacy Hero",
                    "avatar": "characters/01_cast/missing.jpg",
                    "image": "legacy.png",
                },
                {
                    "name": "Evil Traversal",
                    "avatar": "../../secret.png",
                    "image": "escape.png",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 1. Prefers `avatar` resolved against `$AWD` first.
    res_avatar = await rpc(env, "media.hydrate", {"session_id": sid, "ref": "Alpha"})
    assert res_avatar["type"] == "media.ready"
    obj_avatar = res_avatar["payload"]["object"]
    assert env["link"].objects[obj_avatar]["data"] == b"AVATAR_BYTES"

    # 2. Falls back to `image` resolved against `characters/<sub>` when `avatar` is missing.
    res_legacy = await rpc(env, "media.hydrate", {"session_id": sid, "ref": "Legacy Hero"})
    assert res_legacy["type"] == "media.ready"
    obj_legacy = res_legacy["payload"]["object"]
    assert env["link"].objects[obj_legacy]["data"] == b"LEGACY_SUBDIR_BYTES"

    # 3. Path traversal and escaping symlinks are rejected.
    res_evil = await rpc(env, "media.hydrate", {"session_id": sid, "ref": "Evil Traversal"})
    assert res_evil["type"] == "error" and res_evil["payload"]["code"] == "NOT_FOUND"


async def test_prefs_schema_text_rule_and_4kib_cap(env) -> None:
    sid = await create_session(env)
    env["bot"].app_config.modes.append(
        AppModeConfig(
            id="theater",
            title="Theater",
            launch_prompt="start theater",
            prefs_schema={"policy": "text", "short_tag": "string", "extra_a": "text", "extra_b": "text"},
        )
    )

    # 600 Chinese chars = 1800 UTF-8 bytes fits cleanly under "text" and 4 KiB cap.
    cn_600 = "角" * 600
    ok = await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": {"policy": cn_600, "short_tag": "x" * 200}})
    assert ok["type"] == "mode.prefs.set.result"
    assert ok["payload"]["prefs"]["policy"] == cn_600

    # "text" rejects >600 chars, non-string, and NUL bytes.
    for bad in ("角" * 601, 123, "ok\x00bad"):
        err = await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": {"policy": bad}})
        assert err["type"] == "error" and err["payload"]["code"] == "BAD_REQUEST", bad

    # "string" stays capped at 200 chars.
    err_str = await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": {"short_tag": "x" * 201}})
    assert err_str["type"] == "error" and err_str["payload"]["code"] == "BAD_REQUEST"

    # Combined prefs exceeding 4096 UTF-8 bytes are rejected (3 * 600 * 3 = 5400 bytes > 4096).
    err_cap = await rpc(
        env,
        "mode.prefs.set",
        {"session_id": sid, "prefs": {"policy": "角" * 600, "extra_a": "角" * 600, "extra_b": "角" * 600}},
    )
    assert err_cap["type"] == "error" and err_cap["payload"]["code"] == "BAD_REQUEST"


async def test_mode_defaults_crud_and_autoseed(env, monkeypatch) -> None:
    bot, gw = env["bot"], env["gw"]
    bot.app_config.modes.append(
        AppModeConfig(
            id="theater",
            title="Theater",
            launch_prompt="start theater",
            prefs_schema={"casting_policy": "text", "tone": ["light", "dark"]},
        )
    )

    # Empty initially.
    init_all = await rpc(env, "mode.defaults.get", {})
    assert init_all["type"] == "mode.defaults.get.result"
    assert init_all["payload"] == {"defaults": {}}
    init_one = await rpc(env, "mode.defaults.get", {"mode_id": "theater"})
    assert init_one["payload"] == {"defaults": {}}

    # Validation errors on set / get.
    assert (await rpc(env, "mode.defaults.set", {"mode_id": "unknown", "prefs": {"tone": "light"}}))["payload"][
        "code"
    ] == "BAD_REQUEST"
    assert (await rpc(env, "mode.defaults.set", {"mode_id": "theater", "prefs": "bad"}))["payload"][
        "code"
    ] == "BAD_REQUEST"
    assert (await rpc(env, "mode.defaults.set", {"mode_id": "theater", "prefs": {"tone": "invalid"}}))["payload"][
        "code"
    ] == "BAD_REQUEST"
    assert (await rpc(env, "mode.defaults.set", {"mode_id": "theater", "prefs": {"level": "hard"}}))["payload"][
        "code"
    ] == "BAD_REQUEST"
    assert (await rpc(env, "mode.defaults.get", {"mode_id": "../bad"}))["payload"]["code"] == "BAD_REQUEST"

    # Set defaults for two modes and query both filtered and unfiltered.
    set_t = await rpc(
        env,
        "mode.defaults.set",
        {"mode_id": "theater", "prefs": {"casting_policy": "ask_all", "tone": "dark"}},
    )
    assert set_t["type"] == "mode.defaults.set.result"
    assert set_t["payload"] == {"mode_id": "theater", "prefs": {"casting_policy": "ask_all", "tone": "dark"}}
    await rpc(env, "mode.defaults.set", {"mode_id": "quiz", "prefs": {"level": "hard"}})

    got_all = (await rpc(env, "mode.defaults.get", {}))["payload"]["defaults"]
    assert got_all == {"theater": {"casting_policy": "ask_all", "tone": "dark"}, "quiz": {"level": "hard"}}
    got_t = (await rpc(env, "mode.defaults.get", {"mode_id": "theater"}))["payload"]["defaults"]
    assert got_t == {"theater": {"casting_policy": "ask_all", "tone": "dark"}}

    # Merge + None deletion.
    merged_t = await rpc(env, "mode.defaults.set", {"mode_id": "theater", "prefs": {"tone": None}})
    assert merged_t["payload"] == {"mode_id": "theater", "prefs": {"casting_policy": "ask_all"}}

    # 1. Auto-seed on session.create (deferred mode, default).
    create_reply = await rpc(env, "session.create", {"mode": "theater"})
    assert create_reply["type"] == "session.snapshot"
    sid_defer = create_reply["payload"]["session_id"]
    assert create_reply["payload"]["mode_prefs"] == {"casting_policy": "ask_all"}
    prefs_md = (env["tmp"] / "sessions" / sid_defer / "context" / "05-app-prefs.md").read_text(encoding="utf-8")
    assert "`casting_policy`: ask_all" in prefs_md

    # 2. Explicit prefs on session.create override stored defaults while preserving other default keys.
    await rpc(env, "mode.defaults.set", {"mode_id": "theater", "prefs": {"tone": "dark"}})
    create_override = await rpc(env, "session.create", {"mode": "theater", "prefs": {"tone": "light"}})
    assert create_override["payload"]["mode_prefs"] == {"casting_policy": "ask_all", "tone": "light"}

    # 3. Auto-seed on mode.start (both immediate and deferred) without clobbering existing session prefs.
    posted: list[Message] = []
    orig_post = gw.post

    async def capture(msg: Message) -> Message:
        posted.append(msg)
        return await orig_post(msg)

    monkeypatch.setattr(gw, "post", capture)

    sid_start = await create_session(env)
    await rpc(env, "mode.start", {"session_id": sid_start, "mode_id": "theater", "defer": False})
    assert posted[-1].metadata["mode_prefs"] == {"casting_policy": "ask_all", "tone": "dark"}
    assert "`casting_policy`: ask_all" in (
        env["tmp"] / "sessions" / sid_start / "context" / "05-app-prefs.md"
    ).read_text(encoding="utf-8")

    # Deferred mode.start on an existing session with custom `tone` preserves `tone` and fills missing `casting_policy`.
    sid_partial = await create_session(env)
    await rpc(env, "mode.prefs.set", {"session_id": sid_partial, "prefs": {"tone": "light"}})
    await rpc(env, "mode.start", {"session_id": sid_partial, "mode_id": "theater", "defer": True})
    assert bot.sync_db.get_app_session_prefs(sid_partial)["mode_prefs"] == {
        "casting_policy": "ask_all",
        "tone": "light",
    }


async def test_character_list_and_roster_update(env) -> None:
    import fcntl
    import threading

    tmp = env["tmp"]
    (tmp / "outside.png").write_bytes(b"OUTSIDE")

    # Set up character registry with two groups, avatar/image priority, and security traps.
    g_b = tmp / "characters" / "group_b"
    g_b.mkdir(parents=True, exist_ok=True)
    (g_b / "beta_img.png").write_bytes(b"BETA")
    (g_b / "characters.json").write_text(
        json.dumps(
            [
                {"name": "Scout Beta", "aliases": ["Beta"], "image": "beta_img.png"},
                {"name": "No Asset", "trigger": "NoAssetRef", "group": "custom_group", "avatar": "../../outside.png"},
            ]
        ),
        encoding="utf-8",
    )

    g_a = tmp / "characters" / "group_a"
    g_a.mkdir(parents=True, exist_ok=True)
    (g_a / "alpha_av.jpg").write_bytes(b"ALPHA_AV")
    (g_a / "alpha_img.png").write_bytes(b"ALPHA_IMG")
    (g_a / "characters.json").write_text(
        json.dumps(
            [
                {
                    "name": "Heroine Alpha",
                    "trigger": "Alpha Canonical",
                    "aliases": ["Alpha", "主角A"],
                    "group": "A-Team",
                    "avatar": "characters/group_a/alpha_av.jpg",
                    "image": "characters/group_a/alpha_img.png",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 1. character.list deterministic & jailed.
    listed = await rpc(env, "character.list", {})
    assert listed["type"] == "character.list.result"
    chars = listed["payload"]["characters"]
    assert [c["name"] for c in chars] == ["Heroine Alpha", "No Asset", "Scout Beta"]
    assert chars[0] == {
        "name": "Heroine Alpha",
        "ref": "Alpha Canonical",
        "aliases": ["Alpha", "主角A"],
        "group": "A-Team",
        "avatar": "characters/group_a/alpha_av.jpg",
    }
    assert chars[1]["avatar"] == "" and chars[1]["group"] == "custom_group" and chars[1]["ref"] == "NoAssetRef"
    assert chars[2]["avatar"] == "characters/group_b/beta_img.png" and chars[2]["group"] == "group_b"

    # 2. roster.update on base & transformed members + 60-roster.md sync + state.sync.
    sid = await create_session(env)
    staging = tmp / "sessions" / sid
    (staging / "game").mkdir(parents=True, exist_ok=True)

    # Missing roster.json -> NOT_FOUND.
    missing_file = await rpc(env, "roster.update", {"session_id": sid, "key": "Lead", "ref": "Alpha"})
    assert missing_file["type"] == "error" and missing_file["payload"]["code"] == "NOT_FOUND"

    roster_data = {
        "members": {
            "Lead": {"name": "Lead", "prompt": "old look", "ref": ""},
            "Fox": {
                "name": "Fox",
                "form_state": "transformed",
                "ref": "OldRef",
                "prompt": "old fox form",
                "base_appearance": {"prompt": "human base", "ref": "OldRef"},
                "transformed_appearance": {"prompt": "old fox form", "ref": "OldRef"},
            },
        }
    }
    roster_path = staging / "game" / "roster.json"
    roster_path.write_text(json.dumps(roster_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update base member: resolves alias -> canonical trigger, escapes pipe/newlines in 60-roster.md, sets fusion.
    upd = await rpc(
        env,
        "roster.update",
        {
            "session_id": sid,
            "key": "Lead",
            "ref": "alpha",
            "prompt": "silver coat | scarf\nwarm smile",
            "fusion": True,
        },
    )
    assert upd["type"] == "roster.update.result"
    member = upd["payload"]["member"]
    assert member["ref"] == "Alpha Canonical"
    assert member["base_appearance"] == {"prompt": "silver coat | scarf\nwarm smile", "ref": "Alpha Canonical"}
    assert member["prompt"] == "silver coat | scarf\nwarm smile"
    assert member["fusion"] is True

    roster_md = (staging / "context" / "60-roster.md").read_text(encoding="utf-8")
    assert "# 登场角色与形象设定 (Roster)" in roster_md
    assert "| 角色名称 (Name) | 参考图绑定 (Ref) | 形态 | 当前形象与设定 Prompt | 基础外表 |" in roster_md
    assert r"silver coat \| scarf warm smile" in roster_md
    assert len(roster_md.splitlines()) <= 40

    # state.sync emitted immediately with updated roster.json and 60-roster.md.
    syncs = [e for e in env["link"].of_type("state.sync") if e["session_id"] == sid]
    assert syncs and syncs[-1]["payload"]["files"]["game/roster.json"]["members"]["Lead"]["ref"] == "Alpha Canonical"
    assert "context/60-roster.md" in syncs[-1]["payload"]["files"]

    # Clear ref + unset fusion.
    cleared = await rpc(env, "roster.update", {"session_id": sid, "key": "Lead", "ref": "  ", "fusion": False})
    assert cleared["payload"]["member"]["ref"] == ""
    assert cleared["payload"]["member"]["base_appearance"]["ref"] == ""
    assert "fusion" not in cleared["payload"]["member"]

    # Non-transformed member with only transform_prompt -> BAD_REQUEST.
    bad_tf = await rpc(env, "roster.update", {"session_id": sid, "key": "Lead", "transform_prompt": "cat form"})
    assert bad_tf["type"] == "error" and bad_tf["payload"]["code"] == "BAD_REQUEST"

    # Transformed member: update ref, base prompt, and transform_prompt.
    upd_tf = await rpc(
        env,
        "roster.update",
        {
            "session_id": sid,
            "key": "Fox",
            "ref": "Custom Unregistered Ref",
            "prompt": "updated human base",
            "transform_prompt": "fluffy white fox",
        },
    )
    assert upd_tf["type"] == "roster.update.result"
    fox = upd_tf["payload"]["member"]
    assert fox["ref"] == "Custom Unregistered Ref"
    assert fox["base_appearance"] == {"prompt": "updated human base", "ref": "Custom Unregistered Ref"}
    assert fox["transformed_appearance"] == {"prompt": "fluffy white fox", "ref": "Custom Unregistered Ref"}
    assert fox["prompt"] == "fluffy white fox"
    roster_md_tf = (staging / "context" / "60-roster.md").read_text(encoding="utf-8")
    assert "| Fox | Custom Unregistered Ref | 🔮已变身 | fluffy white fox | updated human base |" in roster_md_tf

    # 3. Verify exclusive flock on <staging>/game/.gm.lock blocks concurrent update until released.
    lock_path = staging / "game" / ".gm.lock"
    assert lock_path.is_file()
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    released = threading.Event()
    try:
        task = asyncio.create_task(
            rpc(env, "roster.update", {"session_id": sid, "key": "Lead", "prompt": "after lock"})
        )
        await asyncio.sleep(0.05)
        assert not task.done(), "roster.update must wait while .gm.lock is exclusively locked"
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        released.set()
        res_locked = await asyncio.wait_for(task, timeout=2.0)
        assert res_locked["type"] == "roster.update.result"
        assert res_locked["payload"]["member"]["prompt"] == "after lock"
    finally:
        if not released.is_set():
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    # 4. Large roster (>40 members) stays <= 40 lines with omitted footer.
    big_members = {f"Char{i:02d}": {"name": f"Char{i:02d}", "ref": f"Ref{i:02d}", "prompt": "look"} for i in range(45)}
    roster_path.write_text(json.dumps({"members": big_members}), encoding="utf-8")
    big_res = await rpc(env, "roster.update", {"session_id": sid, "key": "Char00", "prompt": "updated"})
    assert big_res["type"] == "roster.update.result"
    big_md = (staging / "context" / "60-roster.md").read_text(encoding="utf-8")
    assert len(big_md.splitlines()) <= 40
    assert "未列出" in big_md

    # 5. Security & input validation edge cases.
    for bad_payload in (
        {"session_id": sid, "key": "Char00"},
        {"session_id": sid, "key": "", "ref": "x"},
        {"session_id": sid, "key": "k" * 121, "ref": "x"},
        {"session_id": sid, "key": "bad\x00key", "ref": "x"},
        {"session_id": sid, "key": "Char00", "ref": "r" * 161},
        {"session_id": sid, "key": "Char00", "ref": "a\x00b"},
        {"session_id": sid, "key": "Char00", "prompt": "p" * 2001},
        {"session_id": sid, "key": "Char00", "prompt": "a\x00b"},
        {"session_id": sid, "key": "Char00", "transform_prompt": "t" * 2001},
        {"session_id": sid, "key": "Char00", "transform_prompt": "a\x00b"},
        {"session_id": sid, "key": "Char00", "fusion": "yes"},
    ):
        err = await rpc(env, "roster.update", bad_payload)
        assert err["type"] == "error" and err["payload"]["code"] == "BAD_REQUEST", bad_payload

    not_found_key = await rpc(env, "roster.update", {"session_id": sid, "key": "NoSuchChar", "ref": "x"})
    assert not_found_key["type"] == "error" and not_found_key["payload"]["code"] == "NOT_FOUND"

    # Symlinked roster.json escaping staging is refused.
    outside_roster = tmp / "outside_roster.json"
    outside_roster.write_text(json.dumps({"members": {"Char00": {"name": "Char00"}}}), encoding="utf-8")
    roster_path.unlink()
    os.symlink(outside_roster, roster_path)
    sym_err = await rpc(env, "roster.update", {"session_id": sid, "key": "Char00", "ref": "x"})
    assert sym_err["type"] == "error" and sym_err["payload"]["code"] == "NOT_FOUND"

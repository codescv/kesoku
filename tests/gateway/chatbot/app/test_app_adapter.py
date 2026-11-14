"""End-to-end tests of the app channel adapter's wire protocol (fake relay link, real SQLite + gateway)."""

import asyncio
import json
import os
import time
import uuid
from typing import Any

import pytest

import kesoku.config
from kesoku.agent.turn_executor import TurnExecutor
from kesoku.config import AppConfig, AppModeConfig, KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message
from kesoku.gateway.chatbot.app.adapter import AppChatbot
from kesoku.gateway.chatbot.app.protocol import (
    ProtocolError,
    decode_binary_frame,
    encode_binary_frame,
    make_envelope,
    new_id,
    parse_envelope,
    reply_type_for,
)
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

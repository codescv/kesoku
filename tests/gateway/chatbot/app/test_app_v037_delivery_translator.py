"""v0.3.7: pre-flight missing-media self-healing (no duplicate finals) and translator native/foreign language prefs."""

import asyncio
import json
from typing import Any

import pytest

import kesoku.config
from kesoku.config import AppConfig, AppModeConfig, KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message
from kesoku.gateway.chatbot.app.adapter import AppChatbot
from kesoku.gateway.chatbot.app.segments import MediaJob
from kesoku.gateway.chatbot.base import DeliveryAbortedError
from kesoku.gateway.gateway import Gateway
from tests.gateway.chatbot.app.test_app_adapter import FakeLink, create_session, rpc

TRANSLATOR_DEFAULT_PROMPT = "进入通用极速翻译模式：日语⇄简中，零寒暄。"


@pytest.fixture
async def env(tmp_path, monkeypatch):
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


# ---------------------------------------------------------------------------------------------- Item 5


async def test_missing_voice_first_attempt_self_heals_without_duplicate_final(env) -> None:
    bot, link, gw = env["bot"], env["link"], env["gw"]
    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid
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

    hist = (await rpc(env, "history.page", {"session_id": sid}))["payload"]["messages"]
    ids = [m["message_id"] for m in hist]
    assert broken.id not in ids and healed.id in ids and user.id in ids


async def test_self_healing_retry_with_still_missing_media_is_delivered_once(env) -> None:
    bot, link, gw = env["bot"], env["link"], env["gw"]
    sid = await create_session(env)
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


async def test_preflight_accepts_existing_files_and_deferred_sidecars(env) -> None:
    bot, gw = env["bot"], env["gw"]
    sid = await create_session(env)
    staging = env["tmp"] / "sessions" / sid
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


async def test_card_ref_resolvable_only_via_hydration_roots_never_aborts(env) -> None:
    bot, link, gw = env["bot"], env["link"], env["gw"]
    sid = await create_session(env)
    art = env["tmp"] / "games" / "pokemon_assistant" / "data" / "art"
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


async def test_history_page_hides_error_assistant_but_keeps_error_user_messages(env) -> None:
    gw = env["gw"]
    sid = await create_session(env)
    u = _msg(sid, MessageRole.USER, "user text", sender="dev-1", status=MessageStatus.ERROR)
    a_err = _msg(sid, MessageRole.ASSISTANT, "aborted", sender="Kesoku", status=MessageStatus.ERROR)
    a_ok = _msg(sid, MessageRole.ASSISTANT, "fine", sender="Kesoku", status=MessageStatus.DELIVERED)
    for m in (u, a_err, a_ok):
        await gw.db.save_message(m)
    visible = [m.id for m in env["bot"].sync_db.get_messages_page(sid, visible_only=True)]
    assert u.id in visible and a_ok.id in visible and a_err.id not in visible
    everything = [m.id for m in env["bot"].sync_db.get_messages_page(sid, visible_only=False)]
    assert a_err.id in everything


# ---------------------------------------------------------------------------------------------- Item 4 (backend)


async def test_prefs_schema_list_rule(env) -> None:
    sid = await create_session(env)
    ok = await rpc(env, "mode.prefs.set", {"session_id": sid,
                                           "prefs": {"native_lang": "zh-CN", "foreign_langs": ["ja", "en", "ko"]}})
    assert ok["payload"]["prefs"] == {"native_lang": "zh-CN", "foreign_langs": ["ja", "en", "ko"]}
    for bad in ("ja,en", ["x" * 41], [f"l{i}" for i in range(17)], ["ja", 3]):
        err = await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": {"foreign_langs": bad}})
        assert err["type"] == "error" and err["payload"]["code"] == "BAD_REQUEST", bad
    edge = await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": {"foreign_langs": ["y" * 40] * 16}})
    assert edge["type"] == "mode.prefs.set.result", edge
    bad_formality = await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": {"formality": "rude"}})
    assert bad_formality["payload"]["code"] == "BAD_REQUEST"


async def test_prefs_sync_updates_translator_json(env) -> None:
    sid = await create_session(env)
    game = env["tmp"] / "sessions" / sid / "game"
    game.mkdir(parents=True, exist_ok=True)
    (game / "translator.json").write_text(json.dumps({"src_lang": "ja", "dst_lang": "zh-CN", "turns": 3}),
                                          encoding="utf-8")
    prefs = {"native_lang": "en", "foreign_langs": ["ja", "fr"], "src_lang": "fr", "dst_lang": "en",
             "formality": "casual"}
    await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": prefs})
    data = json.loads((game / "translator.json").read_text(encoding="utf-8"))
    assert data == {**prefs, "turns": 3}
    md = (env["tmp"] / "sessions" / sid / "context" / "05-app-prefs.md").read_text(encoding="utf-8")
    assert "`native_lang`: en" in md


async def test_prefs_sync_does_not_create_translator_json(env) -> None:
    sid = await create_session(env)
    await rpc(env, "mode.prefs.set", {"session_id": sid, "prefs": {"native_lang": "ja"}})
    assert not (env["tmp"] / "sessions" / sid / "game" / "translator.json").exists()


async def _start_translator(env, monkeypatch, sid: str, prefs: dict[str, Any] | None = None) -> str:
    sent: list[str] = []

    async def fake_ingest(**kw: Any) -> dict[str, Any]:
        sent.append(kw["text"])
        return {"ok": True}

    monkeypatch.setattr(env["bot"], "ingest_app_message", fake_ingest)
    payload: dict[str, Any] = {"session_id": sid, "mode_id": "translator"}
    if prefs is not None:
        payload["prefs"] = prefs
    reply = await rpc(env, "mode.start", payload)
    assert reply["type"] == "mode.start.result", reply
    return sent[-1]


async def test_translator_start_uses_default_prompt_without_language_prefs(env, monkeypatch) -> None:
    sid = await create_session(env)
    assert await _start_translator(env, monkeypatch, sid) == TRANSLATOR_DEFAULT_PROMPT


async def test_translator_start_builds_prompt_from_stored_prefs_and_syncs_staging(env, monkeypatch) -> None:
    sid = await create_session(env)
    game = env["tmp"] / "sessions" / sid / "game"
    game.mkdir(parents=True, exist_ok=True)
    (game / "translator.json").write_text(json.dumps({"src_lang": "ja", "dst_lang": "zh-CN"}), encoding="utf-8")
    await asyncio.to_thread(
        env["bot"].sync_db.upsert_app_session_prefs, sid,
        mode_prefs={"native_lang": "zh-CN", "foreign_langs": ["en", "ja"], "src_lang": "zh-CN", "dst_lang": "ko"},
    )
    prompt = await _start_translator(env, monkeypatch, sid)
    assert prompt == "进入通用极速翻译模式：母语 zh-CN ⇄ 常用外语 ko（常用外语：en、ja），零寒暄。"
    data = json.loads((game / "translator.json").read_text(encoding="utf-8"))
    assert data["dst_lang"] == "ko" and data["native_lang"] == "zh-CN" and data["foreign_langs"] == ["en", "ja"]


async def test_translator_start_payload_prefs_override_and_persist(env, monkeypatch) -> None:
    sid = await create_session(env)
    prompt = await _start_translator(env, monkeypatch, sid, {"native_lang": "ja", "foreign_langs": ["en"]})
    assert prompt == "进入通用极速翻译模式：母语 ja ⇄ 常用外语 en，零寒暄。"
    stored = env["bot"].sync_db.get_app_session_prefs(sid)["mode_prefs"]
    assert stored == {"native_lang": "ja", "foreign_langs": ["en"]}
    md = (env["tmp"] / "sessions" / sid / "context" / "05-app-prefs.md").read_text(encoding="utf-8")
    assert "`native_lang`: ja" in md

    # Explicit foreign<->foreign pair (neither side is the native language).
    prompt = await _start_translator(env, monkeypatch, sid, {"src_lang": "en", "dst_lang": "fr"})
    assert prompt == "进入通用极速翻译模式：母语 ja，当前外语对 en ⇄ fr，零寒暄。"

    for bad_prefs in ({"formality": "rude"}, {"unknown_key": "x"}, {"foreign_langs": "ja"}):
        bad = await rpc(env, "mode.start", {"session_id": sid, "mode_id": "translator", "prefs": bad_prefs})
        assert bad["type"] == "error" and bad["payload"]["code"] == "BAD_REQUEST", bad_prefs
    not_obj = await rpc(env, "mode.start", {"session_id": sid, "mode_id": "translator", "prefs": ["x"]})
    assert not_obj["payload"]["code"] == "BAD_REQUEST"
    # Rejected prefs never overwrite the stored ones.
    assert env["bot"].sync_db.get_app_session_prefs(sid)["mode_prefs"]["native_lang"] == "ja"


async def test_translator_start_src_dst_only_defaults_native(env, monkeypatch) -> None:
    sid = await create_session(env)
    prompt = await _start_translator(env, monkeypatch, sid, {"src_lang": "ja", "dst_lang": "zh-CN"})
    assert prompt == "进入通用极速翻译模式：母语 zh-CN ⇄ 常用外语 ja，零寒暄。"


async def test_translator_native_only_prompt_avoids_native_as_foreign(env, monkeypatch) -> None:
    sid = await create_session(env)
    prompt = await _start_translator(env, monkeypatch, sid, {"native_lang": "ja"})
    assert prompt == "进入通用极速翻译模式：母语 ja ⇄ 常用外语 en，零寒暄。"


async def test_session_create_with_translator_mode_and_prefs(env, monkeypatch) -> None:
    sent: list[str] = []

    async def fake_ingest(**kw: Any) -> dict[str, Any]:
        sent.append(kw["text"])
        return {"ok": True}

    monkeypatch.setattr(env["bot"], "ingest_app_message", fake_ingest)
    prefs = {"native_lang": "zh-CN", "foreign_langs": ["ja", "en", "ko"], "src_lang": "ja", "dst_lang": "zh-CN"}
    sid = await create_session(env, mode="translator", prefs=prefs)
    assert env["bot"].sync_db.get_app_session_prefs(sid)["mode_prefs"] == prefs
    assert sent == ["进入通用极速翻译模式：母语 zh-CN ⇄ 常用外语 ja（常用外语：ja、en、ko），零寒暄。"]

    # chat-mode session.create just persists prefs.
    sid2 = await create_session(env, prefs={"native_lang": "en"})
    assert env["bot"].sync_db.get_app_session_prefs(sid2)["mode_prefs"] == {"native_lang": "en"}
    assert len(sent) == 1

    # Invalid prefs are rejected before any session is created.
    before = len((await rpc(env, "session.list", {"limit": 50}))["payload"]["sessions"])
    bad = await rpc(env, "session.create", {"mode": "translator", "prefs": {"formality": "rude"}})
    assert bad["type"] == "error" and bad["payload"]["code"] == "BAD_REQUEST"
    assert len((await rpc(env, "session.list", {"limit": 50}))["payload"]["sessions"]) == before

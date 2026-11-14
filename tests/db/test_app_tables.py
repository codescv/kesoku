"""Tests for the app channel / memory proposal DAOs (`kesoku.db.app_tables`, migration v4)."""

import sqlite3
import time

import pytest

from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db.app_tables import APP_SCHEMA_VERSION
from kesoku.db.manager import DatabaseManager
from kesoku.db.models import Message, Session


@pytest.fixture
def db(tmp_path):
    mgr = DatabaseManager(str(tmp_path / "app.db"))
    mgr.init_tables()
    return mgr


def _session(db, sid, *, updated_at, role="default", title=None):
    db.create_session(Session(id=sid, title=title or sid, created_at=updated_at, updated_at=updated_at,
                              role_name=role))


def _msg(db, sid, ts, *, role=MessageRole.USER, mtype=MessageType.TEXT, sender="user", content="hi"):
    m = Message(session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", sender=sender, role=role, type=mtype,
                content=content, timestamp=ts, status=MessageStatus.PROCESSED)
    db.save_message(m)
    return m


# ---------------------------------------------------------------- migration


def test_v4_tables_exist_and_user_version(db):
    with db.connection_provider.connection() as conn:
        names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    expected = {
        "app_devices", "app_pairings", "app_outbox", "app_session_prefs", "app_mode_memory_prefs",
        "memory_proposals", "memory_proposal_blocklist", "avatar_jobs",
    }
    assert expected <= names
    assert version == APP_SCHEMA_VERSION == 4


def test_migration_is_idempotent_and_only_seeds_chat(db):
    db.init_tables()  # second run must not fail or duplicate
    with db.connection_provider.connection() as conn:
        db._apply_app_v4_migration(conn)
    prefs = db.list_app_mode_memory_prefs()
    assert list(prefs) == ["chat"]
    assert prefs["chat"]["memory_policy"] == "full"


def test_seed_mode_prefs_never_overwrites_user_choice(db):
    db.set_app_mode_memory_policy("chat", "isolated", ts=5.0)
    inserted = db.seed_app_mode_memory_prefs({"chat": "full", "quiz": "confirm_only", "bad": "nope"}, ts=9.0)

    assert inserted == 1
    assert db.get_app_mode_memory_policy("chat") == "isolated"
    assert db.get_app_mode_memory_policy("quiz") == "confirm_only"
    assert db.get_app_mode_memory_policy("bad") is None
    assert db.seed_app_mode_memory_prefs({"quiz": "full"}) == 0
    assert db.get_app_mode_memory_policy("quiz") == "confirm_only"


def test_set_mode_policy_upserts_and_validates(db):
    ts = db.set_app_mode_memory_policy("quiz", "full", ts=1.0)
    assert ts == 1.0
    db.set_app_mode_memory_policy("quiz", "confirm_only", ts=2.0)
    assert db.list_app_mode_memory_prefs()["quiz"] == {"memory_policy": "confirm_only", "updated_at": 2.0}
    with pytest.raises(ValueError):
        db.set_app_mode_memory_policy("quiz", "always")


def test_check_constraint_rejects_bad_policy(db):
    with db.connection_provider.connection() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    "INSERT INTO app_mode_memory_prefs (mode_id, memory_policy, updated_at) VALUES (?, ?, ?)",
                    ("x", "sometimes", 1.0),
                )


# ---------------------------------------------------------------- outbox


def test_outbox_seq_monotonic_per_session(db):
    seqs_a = [db.append_app_outbox("aaaa0001", "message.user", {"i": i}, f"evt_a{i}", ts=100.0 + i) for i in range(3)]
    seqs_b = [db.append_app_outbox("bbbb0002", "message.final", {"i": i}, f"evt_b{i}") for i in range(2)]

    assert seqs_a == [1, 2, 3]
    assert seqs_b == [1, 2]
    after = db.get_app_outbox_after("aaaa0001", 1)
    assert [e["seq"] for e in after] == [2, 3]
    assert after[0] == {"session_id": "aaaa0001", "seq": 2, "type": "message.user", "payload": {"i": 1},
                        "ts": 101.0, "event_id": "evt_a1"}
    assert [e["seq"] for e in db.get_app_outbox_after("aaaa0001", 0, limit=2)] == [1, 2]
    assert db.get_app_outbox_bounds("aaaa0001") == (1, 3)
    assert db.get_app_outbox_bounds("nope") == (0, 0)
    assert db.get_app_outbox_heads(["aaaa0001", "bbbb0002", "cccc0003", ""]) == {
        "aaaa0001": 3, "bbbb0002": 2, "cccc0003": 0,
    }
    assert db.get_app_outbox_heads([]) == {}


def test_prune_keeps_head_and_seq_never_restarts(db):
    for i in range(10):
        db.append_app_outbox("aaaa0001", "turn.event", {"i": i}, f"e{i}", ts=float(i))
    db.append_app_outbox("bbbb0002", "turn.event", {}, "old", ts=0.0)

    removed = db.prune_app_outbox(max_per_session=3, cutoff_ts=1e12)  # everything is "old"

    assert removed == 9  # aaaa: 7 by count + 2 by age; bbbb keeps its only (head) row
    assert db.get_app_outbox_bounds("aaaa0001") == (10, 10)
    assert db.get_app_outbox_bounds("bbbb0002") == (1, 1)
    assert db.append_app_outbox("aaaa0001", "turn.event", {}, "next") == 11
    assert db.append_app_outbox("bbbb0002", "turn.event", {}, "next") == 2

    for i in range(4):
        db.append_app_outbox("cccc0003", "turn.event", {}, f"c{i}", ts=time.time())
    db.prune_app_outbox(max_per_session=2, cutoff_ts=0.0)
    assert db.get_app_outbox_bounds("cccc0003") == (3, 4)

    db.delete_app_outbox_session("cccc0003")
    assert db.get_app_outbox_bounds("cccc0003") == (0, 0)


# ---------------------------------------------------------------- session prefs


def test_session_prefs_upsert_partial_and_whitelist(db):
    _session(db, "aaaa0001", updated_at=1.0)
    assert db.get_app_session_prefs("aaaa0001") is None
    db.upsert_app_session_prefs("aaaa0001", llm_override={"llm": "x"}, origin_channel="app")
    db.upsert_app_session_prefs("aaaa0001", mode_prefs={"k": 1})
    prefs = db.get_app_session_prefs("aaaa0001")
    assert prefs["llm_override"] == {"llm": "x"}
    assert prefs["mode_prefs"] == {"k": 1}
    assert prefs["origin_channel"] == "app"
    db.upsert_app_session_prefs("aaaa0001", llm_override=None)
    assert db.get_app_session_prefs("aaaa0001")["llm_override"] is None
    with pytest.raises(ValueError):
        db.upsert_app_session_prefs("aaaa0001", **{"updated_at = 0, mode_prefs": "x"})


# ---------------------------------------------------------------- proposals


def _proposal(db, pid, *, created_at=1.0, expires_at=None, role="alice", fp=None):
    db.insert_memory_proposal(
        proposal_id=pid, session_id="aaaa0001", message_id="m1", role=role, source_mode="quiz",
        ops=[{"file": "user/profile.md", "section": "s", "text": "likes tea", "tags": []}],
        fingerprint=fp or f"fp-{pid}", created_at=created_at,
        expires_at=expires_at if expires_at is not None else time.time() + 3600,
    )


def test_proposal_resolve_only_from_pending(db):
    _proposal(db, "p1")
    got = db.get_memory_proposal("p1")
    assert got["status"] == "pending" and got["ops"][0]["text"] == "likes tea" and got["resolved_at"] is None

    assert db.resolve_memory_proposal("p1", "dismissed", ts=5.0, device_id="d1") is True
    assert db.resolve_memory_proposal("p1", "applied") is False
    got = db.get_memory_proposal("p1")
    assert got["status"] == "dismissed" and got["resolved_at"] == 5.0 and got["resolved_by_device"] == "d1"
    with pytest.raises(ValueError):
        db.resolve_memory_proposal("p1", "pending")
    with pytest.raises(ValueError):
        db.resolve_memory_proposal("p1", "bogus")
    assert db.get_memory_proposal("missing") is None


def test_proposal_claim_and_finish(db):
    _proposal(db, "p1")
    assert db.claim_memory_proposal("p1") is True
    assert db.claim_memory_proposal("p1") is False  # concurrent confirm loses
    assert db.resolve_memory_proposal("p1", "dismissed") is False  # not pending while applying

    db.finish_memory_proposal("p1", "pending")  # failure path
    assert db.get_memory_proposal("p1")["status"] == "pending"

    assert db.claim_memory_proposal("p1") is True
    db.finish_memory_proposal("p1", "applied", ts=7.0, device_id="d2", applied_ops=[{"file": "a.md", "text": "t"}])
    got = db.get_memory_proposal("p1")
    assert got["status"] == "applied" and got["applied_ops"] == [{"file": "a.md", "text": "t"}]
    db.finish_memory_proposal("p1", "pending")  # not applying any more: no-op
    assert db.get_memory_proposal("p1")["status"] == "applied"


def test_proposal_list_and_expire(db):
    now = time.time()
    _proposal(db, "old", created_at=1.0, expires_at=now - 10)
    _proposal(db, "new", created_at=2.0)
    _proposal(db, "done", created_at=3.0, expires_at=now - 10)
    db.resolve_memory_proposal("done", "applied")

    assert db.expire_memory_proposals(now=now) == ["old"]
    assert db.get_memory_proposal("old")["status"] == "expired"
    assert db.get_memory_proposal("done")["status"] == "applied"
    assert [p["proposal_id"] for p in db.list_memory_proposals()] == ["done", "new", "old"]
    assert [p["proposal_id"] for p in db.list_memory_proposals(status="pending")] == ["new"]
    assert len(db.list_memory_proposals(limit=1)) == 1
    assert db.expire_memory_proposals(now=now) == []


def test_proposal_blocklist(db):
    assert db.add_memory_proposal_blocklist(["fa", "fb", "", "fa"], "alice") == 2
    assert db.add_memory_proposal_blocklist(["fc"], "bob") == 1
    assert db.get_blocklisted_fingerprints(["fa", "fc", "zz", ""]) == {"fa", "fc"}
    assert db.get_blocklisted_fingerprints([]) == set()
    assert db.clear_memory_proposal_blocklist("alice") == 2
    assert db.get_blocklisted_fingerprints(["fa", "fb", "fc"]) == {"fc"}
    assert db.clear_memory_proposal_blocklist() == 1


# ---------------------------------------------------------------- devices / pairings


def test_devices_upsert_revoke_and_inactive(db):
    db.upsert_app_device("d1", "Phone", "client", platform="ios", created_at=100.0)
    db.upsert_app_device("d2", "Tablet", "client", created_at=200.0)
    db.upsert_app_device("d1", "Phone 2", "client")  # refresh name, keep platform/created_at
    d1 = db.get_app_device("d1")
    assert d1["name"] == "Phone 2" and d1["platform"] == "ios" and d1["created_at"] == 100.0

    assert db.revoke_app_device("d1", ts=300.0) is True
    assert db.revoke_app_device("d1") is False
    db.upsert_app_device("d1", "Phone 3", "client")
    assert db.get_app_device("d1")["revoked_at"] == 300.0  # re-upsert keeps revocation
    assert db.list_revoked_app_device_ids() == ["d1"]
    assert [d["device_id"] for d in db.list_app_devices()] == ["d2", "d1"]
    assert [d["device_id"] for d in db.list_app_devices(include_revoked=False)] == ["d2"]

    db.upsert_app_device("d3", "Laptop", "client", created_at=50.0)
    db.touch_app_device("d3", ts=1000.0)
    assert db.revoke_inactive_app_devices(cutoff_ts=500.0, now=600.0) == ["d2"]
    assert db.get_app_device("d2")["revoked_at"] == 600.0
    assert db.get_app_device("d3")["revoked_at"] is None
    assert db.get_app_device("nope") is None


def test_pairings_single_active_offer_and_redeem(db):
    now = time.time()
    db.insert_app_pairing("h1", "s1", "first", now, now + 120)
    db.insert_app_pairing("h2", "s2", "second", now + 1, now + 120)

    assert db.get_app_pairing("h1") is None  # older unredeemed offer dropped
    active = db.get_active_app_pairing(now=now + 2)
    assert active["code_hash"] == "h2" and active["salt"] == "s2"
    assert db.mark_app_pairing_redeemed("h2", "dev_x") is True
    assert db.mark_app_pairing_redeemed("h2", "dev_y") is False
    assert db.get_app_pairing("h2")["redeemed_device_id"] == "dev_x"
    assert db.get_active_app_pairing(now=now + 2) is None

    db.insert_app_pairing("h3", "s3", "third", now, now + 10)
    assert db.get_app_pairing("h2") is not None  # redeemed rows survive new offers
    assert db.get_active_app_pairing(now=now + 11) is None  # expired
    assert db.delete_expired_app_pairings(now=now + 11) == 1
    assert db.get_app_pairing("h2") is not None
    db.insert_app_pairing("h4", "s4", "fourth", now, now + 10)
    db.delete_app_pairing("h4")
    assert db.get_app_pairing("h4") is None


# ---------------------------------------------------------------- sessions / history


def test_list_sessions_by_chatbots_bindings_role_and_cursor(db):
    _session(db, "s0000001", updated_at=10.0, role="alice")
    _session(db, "s0000002", updated_at=20.0, role="bob")
    _session(db, "s0000003", updated_at=20.0, role=None)
    _session(db, "s0000004", updated_at=30.0, role="alice")
    _session(db, "s0000005", updated_at=40.0)  # unbound
    db.bind_channel_session_if_absent("app", "app:s0000001", "s0000001")
    db.bind_channel_session_if_absent("discord", "chan-1", "s0000001")
    db.bind_channel_session_if_absent("app", "app:s0000002", "s0000002")
    db.bind_channel_session_if_absent("wechat", "w1", "s0000003")
    db.bind_channel_session_if_absent("app", "app:s0000004", "s0000004")

    app_only = db.list_sessions_by_chatbots(["app"])
    assert [s["session_id"] for s in app_only] == ["s0000004", "s0000002", "s0000001"]
    assert app_only[2]["chatbot_ids"] == ["app", "discord"]

    everything = db.list_sessions_by_chatbots(["app", "discord", "wechat"])
    assert [s["session_id"] for s in everything] == ["s0000004", "s0000003", "s0000002", "s0000001"]
    assert everything[1]["role"] == "default"  # NULL role -> default

    assert [s["session_id"] for s in db.list_sessions_by_chatbots(["app", "wechat"], role="default")] == ["s0000003"]
    assert [s["session_id"] for s in db.list_sessions_by_chatbots(["app"], role="alice")] == ["s0000004", "s0000001"]

    page1 = db.list_sessions_by_chatbots(["app", "wechat"], limit=2)
    assert [s["session_id"] for s in page1] == ["s0000004", "s0000003"]
    cursor = (page1[-1]["updated_at"], page1[-1]["session_id"])
    page2 = db.list_sessions_by_chatbots(["app", "wechat"], cursor=cursor, limit=2)
    assert [s["session_id"] for s in page2] == ["s0000002", "s0000001"]
    assert db.list_sessions_by_chatbots([]) == []
    assert db.list_sessions_by_chatbots([""]) == []


def test_get_messages_page_visible_only_before_ts_chronological(db):
    _session(db, "aaaa0001", updated_at=1.0)
    _msg(db, "aaaa0001", 1.0, content="u1")
    _msg(db, "aaaa0001", 2.0, role=MessageRole.ASSISTANT, mtype=MessageType.THOUGHT, content="thinking")
    _msg(db, "aaaa0001", 3.0, role=MessageRole.ASSISTANT, content="a1")
    _msg(db, "aaaa0001", 4.0, role=MessageRole.TOOL, mtype=MessageType.TOOL_RESULT, content="tool")
    _msg(db, "aaaa0001", 5.0, role=MessageRole.ASSISTANT, sender="Notification", content="note")
    _msg(db, "aaaa0001", 6.0, content="u2")

    page = db.get_messages_page("aaaa0001")
    assert [m.content for m in page] == ["u1", "a1", "u2"]
    assert [m.content for m in db.get_messages_page("aaaa0001", limit=2)] == ["a1", "u2"]
    assert [m.content for m in db.get_messages_page("aaaa0001", before_ts=6.0)] == ["u1", "a1"]
    assert [m.content for m in db.get_messages_page("aaaa0001", before_ts=3.0, visible_only=False)] == [
        "u1", "thinking",
    ]
    assert len(db.get_messages_page("aaaa0001", visible_only=False)) == 6


def test_bind_channel_session_if_absent_never_touches_existing(db):
    _session(db, "aaaa0001", updated_at=1.0)
    _session(db, "bbbb0002", updated_at=2.0)
    assert db.bind_channel_session_if_absent("app", "app:aaaa0001", "aaaa0001") is True
    assert db.bind_channel_session_if_absent("app", "app:aaaa0001", "bbbb0002") is False
    assert db.list_channel_bindings("aaaa0001") == [("app", "app:aaaa0001")]
    assert db.list_channel_bindings("bbbb0002") == []


def test_update_session_role_and_reassign_role_references(db):
    _session(db, "aaaa0001", updated_at=1.0, role="temp_role")
    _session(db, "bbbb0002", updated_at=2.0, role="alice")
    db.set_channel_role("discord", "chan-1", "temp_role")
    db.set_channel_role("discord", "chan-2", "alice")

    changed = db.reassign_role_references("temp_role", "default")

    assert changed == 2
    assert db.get_session("aaaa0001").role_name == "default"
    assert db.get_session("bbbb0002").role_name == "alice"
    assert db.get_channel_role("discord", "chan-1") == "default"
    assert db.get_channel_role("discord", "chan-2") == "alice"
    assert db.reassign_role_references("temp_role", "default") == 0
    db.update_session_role("bbbb0002", "bob")
    assert db.get_session("bbbb0002").role_name == "bob"


# ---------------------------------------------------------------- avatar jobs


def test_avatar_jobs_crud_rate_count_and_expiry(db):
    db.insert_avatar_job(job_id="j1", device_id="d1", target_type="user", target_id="user", style_preset="pixel",
                         prompt_hint=None, source_path="/x/source.png", created_at=100.0, expires_at=200.0)
    db.insert_avatar_job(job_id="j2", device_id="d1", target_type="role", target_id="alice",
                         style_preset="watercolor", prompt_hint="hat", source_path="/y", created_at=150.0,
                         candidates=[{"index": 0, "status": "pending"}])
    db.insert_avatar_job(job_id="j3", device_id="d2", target_type="user", target_id="user", style_preset="pixel",
                         prompt_hint=None, source_path="/z", created_at=150.0)

    job = db.get_avatar_job("j2")
    assert job["candidates"] == [{"index": 0, "status": "pending"}] and job["status"] == "running"
    assert job["expires_at"] == 150.0 + 86400.0
    db.update_avatar_job("j2", candidates=[{"index": 0, "status": "ready"}], status="done")
    job = db.get_avatar_job("j2")
    assert job["candidates"][0]["status"] == "ready" and job["status"] == "done"
    db.update_avatar_job("j2", status="failed")
    assert db.get_avatar_job("j2")["candidates"][0]["status"] == "ready"

    assert db.count_recent_avatar_jobs("d1", since_ts=120.0) == 1
    assert db.count_recent_avatar_jobs("d1", since_ts=0.0) == 2
    assert db.count_recent_avatar_jobs("d3", since_ts=0.0) == 0
    assert [j["job_id"] for j in db.list_expired_avatar_jobs(now=250.0)] == ["j1"]
    db.delete_avatar_job("j1")
    assert db.get_avatar_job("j1") is None

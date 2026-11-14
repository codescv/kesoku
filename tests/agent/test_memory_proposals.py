"""Tests for `confirm_only` memory proposals: staging, confirmation, blocklist and consolidator integration."""

import json
import os
import stat
import time
from typing import Any

import pytest

from kesoku.agent.context_policy import (
    POLICY_RANK,
    most_restrictive_policy,
    normalize_policy,
    resolve_context_policy,
    resolve_memory_policy,
)
from kesoku.agent.llm import BaseLLM, LLMResponse
from kesoku.agent.memory_consolidator import EpisodeExtract, MemoryConsolidator, format_turn_for_extraction
from kesoku.agent.memory_proposals import (
    ProposalNotFoundError,
    ProposalNotPendingError,
    ProposalStore,
    ProposalValidationError,
    op_fingerprint,
)
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, Session


def _cfg(tmp_path) -> tuple[KesokuConfig, DatabaseManager]:
    db_path = str(tmp_path / "p.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, roles_dir=str(tmp_path / "roles")))
    cfg.agent_working_dir = str(tmp_path)
    return cfg, db


def _op(text: str, file: str = "user/profile.md", section: str = "Facts") -> dict[str, Any]:
    return {"op": "upsert_bullet", "file": file, "section": section, "text": text, "tags": ["#direct"]}


# ---------------------------------------------------------------------------------------------- policy rank


def test_four_state_policy_rank_and_confirm_globs() -> None:
    assert POLICY_RANK == {"full": 0, "real_facts_only": 1, "confirm_only": 2, "isolated": 3}
    assert normalize_policy(" Confirm_Only ") == "confirm_only"
    assert most_restrictive_policy("full", "real_facts_only", "confirm_only") == "confirm_only"
    assert most_restrictive_policy("confirm_only", "isolated") == "isolated"
    cfg = KesokuConfig()
    cfg.agent.memory_confirm_contexts = ["game-q*"]
    cfg.agent.memory_filtered_contexts = ["game-*"]
    assert resolve_context_policy(["game-quiz#1"], {}, cfg) == "confirm_only"
    assert resolve_context_policy(["game-story#1"], {}, cfg) == "real_facts_only"
    assert resolve_context_policy(["game-x"], {"game-x": "confirm_only"}, cfg) == "confirm_only"


def test_resolve_memory_policy_override_is_authoritative() -> None:
    assert resolve_memory_policy("s", "isolated", lambda sid: "full") == "full"
    assert resolve_memory_policy("s", "full", lambda sid: "isolated") == "isolated"
    assert resolve_memory_policy("s", "real_facts_only", lambda sid: None) == "real_facts_only"
    assert resolve_memory_policy("s", "full", lambda sid: "bogus") == "full"

    def boom(sid: str) -> str:
        raise RuntimeError("x")

    assert resolve_memory_policy("s", "confirm_only", boom) == "confirm_only"
    assert resolve_memory_policy("s", None, None) == "full"


def test_confirm_only_turn_rendered_with_real_facts_frame() -> None:
    msg = Message(session_id="s", chatbot_id="c", channel_id="c", sender="u", role=MessageRole.USER, content="hi")
    assert "[scenario frame — keep only real-world facts]" in format_turn_for_extraction([msg], "confirm_only")


# ---------------------------------------------------------------------------------------------- ProposalStore


def test_stage_filters_and_dedupes(tmp_path) -> None:
    cfg, db = _cfg(tmp_path)
    store = ProposalStore(db, cfg)
    p = store.stage_sync(
        session_id="s1",
        role="tutor",
        ops=[
            _op("Allergic to shellfish"),
            _op("allergic   to SHELLFISH"),  # normalized duplicate
            {"op": "retract", "file": "user/profile.md", "match": "x"},
            _op("escape", file="../../outside.md"),
            _op("x" * 900, file="topics/long.md"),
            "garbage",
        ],
        message_id="m1",
        source_mode="quiz",
    )
    assert p is not None and p["status"] == "pending" and p["source_mode"] == "quiz"
    assert [o["text"][:10] for o in p["ops"]] == ["Allergic t", "xxxxxxxxxx"]
    assert len(p["ops"][1]["text"]) == 500
    assert p["expires_at"] - p["created_at"] == pytest.approx(7 * 86400)
    # Same facts already pending -> nothing new staged.
    assert store.stage_sync(session_id="s2", role="tutor", ops=[_op("Allergic to shellfish")]) is None
    # Different role is independent.
    assert store.stage_sync(session_id="s2", role="other", ops=[_op("Allergic to shellfish")]) is not None
    assert store.stage_sync(session_id="s3", role="tutor", ops=[{"op": "retract", "file": "a.md"}]) is None


def test_confirm_writes_only_via_memory_writer_and_edits_are_validated(tmp_path) -> None:
    cfg, db = _cfg(tmp_path)
    store = ProposalStore(db, cfg)
    p = store.stage_sync(
        session_id="s1", role="tutor", ops=[_op("Lives in Osaka"), _op("Plays piano", "topics/hobby.md")]
    )
    with pytest.raises(ProposalValidationError):
        store.confirm_sync(p["proposal_id"], edited_ops=[_op("Lives in Kyoto", file="topics/other.md")])
    with pytest.raises(ProposalValidationError):
        store.confirm_sync(p["proposal_id"], edited_ops=[_op("a"), _op("b"), _op("c")])
    with pytest.raises(ProposalValidationError):
        store.confirm_sync(p["proposal_id"], edited_ops=[_op("y" * 501)])
    res = store.confirm_sync(p["proposal_id"], edited_ops=[_op("Lives in Kyoto")], device_id="dev-1")
    assert res["status"] == "applied" and res["applied_files"] == ["user/profile.md"]
    text = (tmp_path / "roles" / "tutor" / "memories" / "user" / "profile.md").read_text(encoding="utf-8")
    assert "Lives in Kyoto" in text and "Osaka" not in text
    assert not (tmp_path / "roles" / "tutor" / "memories" / "topics" / "hobby.md").exists()
    stored = db.get_memory_proposal(p["proposal_id"])
    assert stored["status"] == "applied" and stored["resolved_by_device"] == "dev-1"
    assert stored["applied_ops"][0]["text"] == "Lives in Kyoto"
    with pytest.raises(ProposalNotPendingError):
        store.confirm_sync(p["proposal_id"])
    with pytest.raises(ProposalNotFoundError):
        store.confirm_sync("mp_missing")


def test_dismiss_blocklists_fingerprints_and_expiry(tmp_path) -> None:
    now = [1_800_000_000.0]
    cfg, db = _cfg(tmp_path)
    store = ProposalStore(db, cfg, clock=lambda: now[0])
    p = store.stage_sync(session_id="s1", role="tutor", ops=[_op("Hates cilantro")])
    res = store.dismiss_sync(p["proposal_id"], device_id="d")
    assert res["status"] == "dismissed"
    assert db.get_blocklisted_fingerprints([op_fingerprint(_op("hates  CILANTRO"))])
    assert store.stage_sync(session_id="s9", role="tutor", ops=[_op("Hates cilantro")]) is None
    with pytest.raises(ProposalNotPendingError):
        store.dismiss_sync(p["proposal_id"])
    db.clear_memory_proposal_blocklist()
    assert store.stage_sync(session_id="s9", role="tutor", ops=[_op("Hates cilantro")]) is not None

    q = store.stage_sync(session_id="s1", role="tutor", ops=[_op("Moving next month")])
    now[0] += 7 * 86400 + 1
    expired = store.expire_sync()
    assert q["proposal_id"] in {e["proposal_id"] for e in expired}
    with pytest.raises(ProposalNotPendingError):
        store.confirm_sync(q["proposal_id"])


async def test_stage_notifies_listeners(tmp_path) -> None:
    cfg, db = _cfg(tmp_path)
    seen: list[dict[str, Any]] = []

    async def listener(p: dict[str, Any]) -> None:
        seen.append(p)

    async def broken(p: dict[str, Any]) -> None:
        raise RuntimeError("listener failure must not propagate")

    store = ProposalStore(db, cfg, listeners=[broken, listener])
    p = await store.stage(session_id="s", role="tutor", ops=[_op("Likes jazz")])
    assert seen and seen[0]["proposal_id"] == p["proposal_id"]


# ---------------------------------------------------------------------------------------------- consolidator


def _save_turn(db: DatabaseManager, sid: str, idx: int, user: str, asst: str, ts: float, policy: str) -> Message:
    u = Message(id=f"{sid}_u{idx}", session_id=sid, chatbot_id="app", channel_id=f"app:{sid}", sender="dev",
                role=MessageRole.USER, type=MessageType.TEXT, content=user, timestamp=ts,
                status=MessageStatus.PROCESSED, metadata={"context_policy": policy})
    db.save_message(u)
    db.save_message(Message(id=f"{sid}_a{idx}", session_id=sid, chatbot_id="app", channel_id=f"app:{sid}",
                            sender="assistant", role=MessageRole.ASSISTANT, type=MessageType.TEXT, content=asst,
                            timestamp=ts + 0.5, status=MessageStatus.DELIVERED, parent_id=u.id))
    return u


class ScriptedExtractor(BaseLLM):
    """Returns ops depending on which user facts are visible in the prompt."""

    def __init__(self, facts: dict[str, dict[str, Any]]) -> None:
        self.facts = facts
        self.prompts: list[str] = []

    async def generate(self, prompt: str | None = None, **kwargs: Any) -> LLMResponse:
        text = prompt or ""
        self.prompts.append(text)
        ops = [op for marker, op in self.facts.items() if f"USER: {marker}" in text]
        payload = {"episode_summary": "summary", "open_threads": [], "ops": ops}
        return LLMResponse(content=json.dumps(payload, ensure_ascii=False), prompt_tokens=10, completion_tokens=5,
                           total_tokens=15)


async def test_confirm_only_never_touches_memory_dir(tmp_path) -> None:
    cfg, db = _cfg(tmp_path)
    mem_dir = tmp_path / "roles" / "tutor" / "memories"
    (mem_dir / "user").mkdir(parents=True)
    (mem_dir / "user" / "profile.md").write_text("---\nvisibility: PINNED\n---\n- existing fact\n", encoding="utf-8")
    llm = ScriptedExtractor({"I am allergic to peanuts": _op("Allergic to peanuts")})
    ctx = KesokuContext(config=cfg, llm=llm, db=db)
    staged: list[dict[str, Any]] = []

    async def listener(p: dict[str, Any]) -> None:
        staged.append(p)

    ctx.memory_proposal_listeners.append(listener)
    ctx.memory_mode_provider = lambda sid: "quiz"
    consolidator = MemoryConsolidator(ctx, prewarm_embeddings=False, enable_timers=False)
    now = time.time() - 100
    db.create_session(Session(id="cs000001", title="quiz", created_at=now, updated_at=now, role_name="tutor"))
    for i in range(10):
        _save_turn(db, "cs000001", i, "I am allergic to peanuts" if i == 3 else f"answer {i}", "ok", now + i,
                   "confirm_only")

    before = {p: os.stat(p).st_mtime_ns for p in [str(mem_dir), str(mem_dir / "user"), str(mem_dir / "user" /
                                                                                         "profile.md")]}
    # Read-only mount simulation: any write attempt into $MEMORY_DIR would raise PermissionError.
    os.chmod(mem_dir / "user", stat.S_IRUSR | stat.S_IXUSR)
    os.chmod(mem_dir, stat.S_IRUSR | stat.S_IXUSR)
    try:
        outcome = await consolidator.run_episode_extract(EpisodeExtract(role="tutor", session_id="cs000001"))
    finally:
        os.chmod(mem_dir, 0o755)
        os.chmod(mem_dir / "user", 0o755)
    assert outcome.status == "ok", outcome
    assert outcome.ops_applied == 0 and outcome.proposal_id
    assert sorted(os.listdir(mem_dir)) == ["user"] or sorted(os.listdir(mem_dir)) == [".memory.lock", "user"]
    assert {p: os.stat(p).st_mtime_ns for p in before} == before
    assert "Allergic to peanuts" not in (mem_dir / "user" / "profile.md").read_text(encoding="utf-8")
    assert len(llm.prompts) == 1 and "[scenario frame — keep only real-world facts]" in llm.prompts[0]
    assert staged and staged[0]["proposal_id"] == outcome.proposal_id and staged[0]["source_mode"] == "quiz"
    assert staged[0]["message_id"] == "cs000001_u9"

    # The only write path: explicit confirmation.
    store = ProposalStore(db, cfg)
    store.confirm_sync(outcome.proposal_id)
    assert "Allergic to peanuts" in (mem_dir / "user" / "profile.md").read_text(encoding="utf-8")
    await consolidator.stop(wait_seconds=1.0)


async def test_mixed_window_applies_auto_turns_and_stages_confirm_turns(tmp_path) -> None:
    cfg, db = _cfg(tmp_path)
    llm = ScriptedExtractor(
        {
            "I work as a nurse": _op("Works as a nurse"),
            "My sister lives in Lyon": _op("Sister lives in Lyon", file="topics/family.md"),
        }
    )
    ctx = KesokuContext(config=cfg, llm=llm, db=db)
    consolidator = MemoryConsolidator(ctx, prewarm_embeddings=False, enable_timers=False)
    now = time.time() - 100
    db.create_session(Session(id="mx000001", title="mixed", created_at=now, updated_at=now, role_name="tutor"))
    _save_turn(db, "mx000001", 1, "I work as a nurse", "nice", now + 1, "full")
    _save_turn(db, "mx000001", 2, "My sister lives in Lyon", "cool", now + 2, "confirm_only")
    _save_turn(db, "mx000001", 3, "secret dungeon stuff", "ok", now + 3, "isolated")
    outcome = await consolidator.run_episode_extract(EpisodeExtract(role="tutor", session_id="mx000001"))
    assert outcome.status == "ok"
    assert len(llm.prompts) == 2
    auto_prompt, confirm_prompt = llm.prompts
    assert "I work as a nurse" in auto_prompt and "My sister lives in Lyon" not in auto_prompt
    assert "My sister lives in Lyon" in confirm_prompt and "I work as a nurse" not in confirm_prompt
    assert "secret dungeon stuff" not in auto_prompt + confirm_prompt
    mem_dir = tmp_path / "roles" / "tutor" / "memories"
    assert "Works as a nurse" in (mem_dir / "user" / "profile.md").read_text(encoding="utf-8")
    assert not (mem_dir / "topics" / "family.md").exists()
    pending = db.list_memory_proposals(status="pending")
    assert len(pending) == 1 and pending[0]["ops"][0]["text"] == "Sister lives in Lyon"
    assert outcome.proposal_id == pending[0]["proposal_id"]
    await consolidator.stop(wait_seconds=1.0)


async def test_blocklisted_fact_is_never_reproposed_by_consolidator(tmp_path) -> None:
    cfg, db = _cfg(tmp_path)
    db.add_memory_proposal_blocklist([op_fingerprint(_op("Allergic to peanuts"))], "tutor")
    llm = ScriptedExtractor({"peanuts again": _op("Allergic to peanuts")})
    ctx = KesokuContext(config=cfg, llm=llm, db=db)
    consolidator = MemoryConsolidator(ctx, prewarm_embeddings=False, enable_timers=False)
    now = time.time() - 100
    db.create_session(Session(id="bl000001", title="b", created_at=now, updated_at=now, role_name="tutor"))
    _save_turn(db, "bl000001", 1, "peanuts again", "ok", now + 1, "confirm_only")
    outcome = await consolidator.run_episode_extract(EpisodeExtract(role="tutor", session_id="bl000001"))
    assert outcome.status == "ok" and outcome.proposal_id == ""
    assert db.list_memory_proposals() == []
    await consolidator.stop(wait_seconds=1.0)

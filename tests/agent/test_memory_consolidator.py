"""Unit tests for `MemoryConsolidator`, safe extraction window, triggers, consolidation, and CLI."""

import json
import time
from typing import Any

import pytest
from typer.testing import CliRunner

from kesoku.agent.llm import BaseLLM, LLMResponse
from kesoku.agent.memory_consolidator import (
    ConsolidateFile,
    EpisodeExtract,
    MemoryConsolidator,
    consolidate_file_sync,
    resolve_extraction_thinking_level,
)
from kesoku.cli.entrypoint import app
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, Session, SummaryNode

runner = CliRunner()


def _save_turn(
    db: DatabaseManager,
    *,
    sid: str,
    turn_idx: int,
    user_text: str,
    asst_text: str,
    ts: float,
    policy: str = "full",
    user_status: str = MessageStatus.PROCESSED,
    asst_status: str = MessageStatus.PENDING,
) -> tuple[Message, Message]:
    u = Message(
        id=f"{sid}_u{turn_idx}",
        session_id=sid,
        chatbot_id="discord",
        channel_id="ch1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content=user_text,
        timestamp=ts,
        status=user_status,
        metadata={"context_policy": policy},
    )
    db.save_message(u)
    a = Message(
        id=f"{sid}_a{turn_idx}",
        session_id=sid,
        chatbot_id="discord",
        channel_id="ch1",
        sender="assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content=asst_text,
        timestamp=ts + 0.5,
        status=asst_status,
        parent_id=u.id,
    )
    db.save_message(a)
    return u, a


def test_safe_window_excludes_in_flight_pending_agent_and_processing_while_keeping_assistant_pending(tmp_path) -> None:
    """Verify `get_unextracted_messages` stops strictly before `pending_agent`/`processing` USER rows (`C4`, `R2-1`)."""
    db = DatabaseManager(str(tmp_path / "window.db"))
    db.init_tables()
    now = 1_760_000_000.0
    db.create_session(Session(id="s_win", title="Window", created_at=now, updated_at=now, role_name="tutor"))

    # Turn 1: completed user (`processed`), assistant still in `MessageStatus.PENDING` delivery -> MUST be included!
    _save_turn(
        db,
        sid="s_win",
        turn_idx=1,
        user_text="我对牛奶乳糖不耐受，以后推荐豆奶",
        asst_text="没问题，无糖豆奶很适合你",
        ts=now + 1,
        user_status=MessageStatus.PROCESSED,
        asst_status=MessageStatus.PENDING,
    )

    # Same-millisecond tie-break check on Turn 2
    m2a = Message(
        id="s_win_u2a",
        session_id="s_win",
        chatbot_id="discord",
        channel_id="ch1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="补充一句",
        timestamp=now + 5,
        status=MessageStatus.PROCESSED,
    )
    m2b = Message(
        id="s_win_u2b",
        session_id="s_win",
        chatbot_id="discord",
        channel_id="ch1",
        sender="assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="收到补充",
        timestamp=now + 5,  # identical millisecond timestamp
        status=MessageStatus.DELIVERED,
    )
    db.save_message(m2a)
    db.save_message(m2b)

    # Turn 3: in-flight queued USER turn (`MessageStatus.PENDING_AGENT`) -> MUST NOT be swept!
    in_flight = Message(
        id="s_win_u3_inflight",
        session_id="s_win",
        chatbot_id="discord",
        channel_id="ch1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="下一轮正在排队的问题",
        timestamp=now + 10,
        status=MessageStatus.PENDING_AGENT,
    )
    db.save_message(in_flight)

    window = db.get_unextracted_messages("s_win")
    assert [m.id for m in window] == ["s_win_u1", "s_win_a1", "s_win_u2a", "s_win_u2b"]

    # Advancing cursor to `s_win_u2a` (`ts = now + 5`) still fetches `s_win_u2b` at identical `ts = now + 5`!
    rem = db.get_unextracted_messages("s_win", after_ts=now + 5, after_id="s_win_u2a")
    assert [m.id for m in rem] == ["s_win_u2b"]


def test_thinking_level_mapping_never_returns_none() -> None:
    """Verify `resolve_extraction_thinking_level` maps Gemini -> `minimal` and Claude -> `low` (`H4`)."""
    assert resolve_extraction_thinking_level("gemini", "minimal") == "minimal"
    assert resolve_extraction_thinking_level("gemini", None) == "minimal"
    assert resolve_extraction_thinking_level("claude", "minimal") == "low"
    assert resolve_extraction_thinking_level("claude", None) == "low"


@pytest.mark.asyncio
async def test_all_isolated_zero_llm_cost_and_real_facts_only_with_compaction_hint(tmp_path, monkeypatch) -> None:
    """Verify all-`isolated` window costs 0 LLM calls, while `real_facts_only` + T3 hint extracts real facts."""
    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", lambda text: [0.5] * 384)
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [[0.5] * 384 for _ in texts])

    db_path = str(tmp_path / "consolidator.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    roles_dir = tmp_path / "roles"
    custom_rubric = tmp_path / "custom_rubric.md"
    custom_rubric.write_text("EXTRA RUBRIC: Never save fictional dungeon loot.", encoding="utf-8")

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, roles_dir=str(roles_dir)))
    cfg.agent_working_dir = str(tmp_path)
    cfg.agent.memory_extraction_prompts = [str(custom_rubric)]

    captured_prompts: list[str] = []

    class ExtractorMockLLM(BaseLLM):
        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            user_prompt = "\n".join([system_prompt or "", prompt or "", *(m.content for m in (history or []))])
            captured_prompts.append(user_prompt)
            payload = {
                "episode_summary": "用户在德里出差，有乳糖不耐受，订了周三返程航班",
                "open_threads": ["机场快线时刻表"],
                "ops": [
                    {
                        "op": "upsert_bullet",
                        "file": "user/profile.md",
                        "section": "饮食",
                        "text": "乳糖不耐受，避免纯牛奶制品",
                        "tags": ["#direct"],
                        "pinned": True,
                    }
                ],
            }
            return LLMResponse(
                content=json.dumps(payload, ensure_ascii=False),
                prompt_tokens=300,
                completion_tokens=80,
                total_tokens=380,
            )

    llm = ExtractorMockLLM()
    ctx = KesokuContext(config=cfg, llm=llm, db=db)
    consolidator = MemoryConsolidator(ctx, prewarm_embeddings=False, enable_timers=False)

    now = 1_760_000_000.0
    # 1. Pure `isolated` session -> 0 LLM calls, 0 tokens, cursor advanced!
    db.create_session(Session(id="iso_trpg_01", title="地下城战役", created_at=now, updated_at=now, role_name="tutor"))
    _save_turn(
        db,
        sid="iso_trpg_01",
        turn_idx=1,
        user_text="我的精灵法师对秘银过敏，住在月光森林",
        asst_text="记下了你的角色设定",
        ts=now + 1,
        policy="isolated",
    )
    iso_outcome = await consolidator.run_episode_extract(EpisodeExtract(role="tutor", session_id="iso_trpg_01"))
    assert iso_outcome.status == "skipped_isolated"
    assert len(captured_prompts) == 0
    assert not (roles_dir / "tutor" / "memories" / "user" / "profile.md").exists()

    # 2. `real_facts_only` session + T3 compaction hint -> LLM sees custom rubric, scenario frame prefix, and learnings!
    db.create_session(
        Session(id="rfo_comp_02", title="德里行程", created_at=now + 20, updated_at=now + 20, role_name="tutor")
    )
    _save_turn(
        db,
        sid="rfo_comp_02",
        turn_idx=1,
        user_text="我这周在德里出差，顺便提醒一下我乳糖不耐受",
        asst_text="好的，帮你避开含奶油的咖喱",
        ts=now + 21,
        policy="real_facts_only",
    )
    db.insert_summary_node(
        SummaryNode(
            id="node_hint_1",
            session_id="rfo_comp_02",
            level=0,
            summary=json.dumps({"timeline": ["- 出差讨论"], "learnings": ["用户周三晚航班返程"]}, ensure_ascii=False),
            start_timestamp=now + 20,
            end_timestamp=now + 22,
            token_count=20,
            source_token_count=100,
            parent_id=None,
            created_at=now + 23,
        )
    )
    consolidator.attach_compaction_hint("rfo_comp_02", "node_hint_1")

    rfo_outcome = await consolidator.run_episode_extract(EpisodeExtract(role="tutor", session_id="rfo_comp_02"))
    assert rfo_outcome.status == "ok"
    assert rfo_outcome.ops_applied == 1
    assert len(captured_prompts) == 1
    assert "EXTRA RUBRIC: Never save fictional dungeon loot." in captured_prompts[0]
    assert "[scenario frame — keep only real-world facts]" in captured_prompts[0]
    assert "用户周三晚航班返程" in captured_prompts[0]

    profile_text = (roles_dir / "tutor" / "memories" / "user" / "profile.md").read_text(encoding="utf-8")
    assert "乳糖不耐受" in profile_text
    digest = db.get_session_digest("rfo_comp_02")
    assert digest is not None
    assert "德里出差" in digest["summary"]

    await consolidator.stop(wait_seconds=2.0)


@pytest.mark.asyncio
async def test_triggers_debounce_budget_backoff_and_consolidate_file_splitter(tmp_path, monkeypatch) -> None:
    """Verify T1 signal/clicked_option/debounce, daily budget vs T2 bypass, failure backoff, and `ConsolidateFile`."""
    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", lambda text: [0.3] * 384)
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [[0.3] * 384 for _ in texts])

    db_path = str(tmp_path / "trig.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    roles_dir = tmp_path / "roles"
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, roles_dir=str(roles_dir)))
    cfg.agent_working_dir = str(tmp_path)
    cfg.agent.memory_extraction_daily_token_budget = 100  # small budget to test exhaustion & T2 bypass

    should_fail = False

    class ToggleLLM(BaseLLM):
        async def generate(self, **kwargs: Any) -> LLMResponse:
            if should_fail:
                raise RuntimeError("Simulated LLM outage")
            return LLMResponse(
                content='{"episode_summary": "摘要", "open_threads": [], "ops": []}',
                prompt_tokens=120,
                completion_tokens=30,
                total_tokens=150,
            )

    ctx = KesokuContext(config=cfg, llm=ToggleLLM(), db=db)
    consolidator = MemoryConsolidator(ctx, prewarm_embeddings=False, enable_timers=False)

    now = 1_760_000_000.0
    db.create_session(Session(id="s_trig", title="Triggers", created_at=now, updated_at=now, role_name="coder"))
    u1, _ = _save_turn(
        db,
        sid="s_trig",
        turn_idx=1,
        user_text="记住以后所有测试都用 pytest 运行",
        asst_text="好的",
        ts=now + 1,
    )

    # 1. Clicked option with signal keyword -> does NOT enqueue immediate T1 (`H4`)
    enqueued_jobs: list[EpisodeExtract] = []
    monkeypatch.setattr(consolidator, "enqueue", lambda job: enqueued_jobs.append(job) or True)  # type: ignore[assignment]

    await consolidator.notify_turn(session_id="s_trig", role="coder", user_msg=u1, clicked_option=True)
    assert enqueued_jobs == []

    # 2. Real human turn matching signal regex -> enqueues `signal` immediately; second signal within 60s debounces!
    await consolidator.notify_turn(session_id="s_trig", role="coder", user_msg=u1, clicked_option=False)
    assert len(enqueued_jobs) == 1 and enqueued_jobs[0].reason == "signal"

    await consolidator.notify_turn(session_id="s_trig", role="coder", user_msg=u1, clicked_option=False)
    assert len(enqueued_jobs) == 1  # debounced!

    # 3. Failure keeps cursor unadvanced & increments `failures`; 3 failures defer non-nightly runs (`C4`)
    should_fail = True
    out_err = await consolidator.run_episode_extract(EpisodeExtract(role="coder", session_id="s_trig"))
    assert out_err.status == "error"
    cur_after_err = db.get_extraction_cursor("s_trig")
    assert cur_after_err is not None
    assert cur_after_err["failures"] == 1
    assert cur_after_err["last_extracted_msg_id"] == ""

    # Immediate retry hits exponential backoff
    out_backoff = await consolidator.run_episode_extract(EpisodeExtract(role="coder", session_id="s_trig"))
    assert out_backoff.status == "backoff"

    # Reset failures and run once (spends 150 tokens > 100 daily budget)
    should_fail = False
    db.upsert_extraction_cursor("s_trig", "coder", failures=0, last_run_at=0.0)
    out_ok = await consolidator.run_episode_extract(EpisodeExtract(role="coder", session_id="s_trig"))
    assert out_ok.status == "ok"

    # Add Turn 2; regular T1 job is now blocked by exhausted daily budget, whereas T2 (`bypass_budget=True`) runs!
    _save_turn(
        db,
        sid="s_trig",
        turn_idx=2,
        user_text="我下周搬到西雅图办公室工作，请记录时区变化",
        asst_text="已了解",
        ts=now + 30,
    )
    out_budget = await consolidator.run_episode_extract(
        EpisodeExtract(role="coder", session_id="s_trig", reason="batch")
    )
    assert out_budget.status == "budget_exhausted"

    out_t2 = await consolidator.run_episode_extract(
        EpisodeExtract(role="coder", session_id="s_trig", reason="close", bypass_budget=True)
    )
    assert out_t2.status == "ok"

    # 4. `ConsolidateFile` splits >12 KiB topic files and demotes over-budget PINNED bullets
    mem_dir = roles_dir / "coder" / "memories"
    (mem_dir / "user").mkdir(parents=True, exist_ok=True)
    distinct_pinned = [
        f"- note_{i:02d}_topic_{i * 13}_detail_{i * 29}_ extra_{i * 47}_context_{i * 71} #inferred"
        for i in range(25)
    ]
    pinned_lines = "\n".join(distinct_pinned)
    (mem_dir / "user" / "profile.md").write_text(
        f"---\ndescription: User profile\nvisibility: PINNED\n---\n"
        f"## 核心\n- 必留直接偏好：对花生严重过敏 #direct\n## 次要\n{pinned_lines}\n",
        encoding="utf-8",
    )
    demoted_touched = consolidate_file_sync(
        db,
        "coder",
        "user/profile.md",
        str(mem_dir),
        max_file_bytes=12288,
        pinned_max_chars=500,
    )
    assert "user/profile.md" in demoted_touched
    assert any(p.startswith("topics/") for p in demoted_touched)
    assert (mem_dir / "topics" / "次要.md").is_file()

    sec_a = "\n".join(
        f"- alpha_{i:03d}_module_{i * 17}_index_{i * 31}_cache_{i * 53}_shard_{i * 97} #direct"
        for i in range(60)
    )
    sec_b = "\n".join(
        f"- beta_{i:03d}_deploy_{i * 19}_canary_{i * 37}_probe_{i * 61}_drain_{i * 89} #direct"
        for i in range(60)
    )
    (mem_dir / "handbook.md").write_text(
        f"---\ndescription: Engineering handbook\n---\n## 架构模块\n{sec_a}\n\n## 部署模块\n{sec_b}\n",
        encoding="utf-8",
    )
    split_touched = consolidate_file_sync(
        db,
        "coder",
        "handbook.md",
        str(mem_dir),
        max_file_bytes=4096,
    )
    assert any(p.startswith("handbook/") for p in split_touched)
    assert ConsolidateFile(role="coder", rel_path="handbook.md").rel_path == "handbook.md"

    await consolidator.stop(wait_seconds=2.0)


def test_kesoku_memory_cli_subcommands(tmp_path, monkeypatch) -> None:
    """Verify `kesoku memory status|reindex|backfill-fts|backfill --dry-run|consolidate` CLI commands."""
    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", lambda text: [0.2] * 384)
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [[0.2] * 384 for _ in texts])

    res_init = runner.invoke(app, ["init", "-w", str(tmp_path)])
    assert res_init.exit_code == 0
    cfg_path = tmp_path / "config.toml"
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(
            db_path=str(tmp_path / "kesoku.db"),
            roles_dir=str(tmp_path / "roles"),
            sessions_dir=str(tmp_path / "sessions"),
        )
    )
    db = DatabaseManager(cfg.workspace.db_path)
    db.init_tables()

    mem_dir = tmp_path / "roles" / "tutor" / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "grammar.md").write_text(
        "---\ndescription: Japanese particles\n---\n## 助词\n- に vs で 区分 #direct\n",
        encoding="utf-8",
    )

    now = time.time()
    db.create_session(
        Session(id="s_cli_bf", title="CLI Backfill", created_at=now - 60, updated_at=now, role_name="tutor")
    )
    _save_turn(
        db,
        sid="s_cli_bf",
        turn_idx=1,
        user_text="我住在东京品川区，平时坐山手线通勤",
        asst_text="品川站换乘非常方便",
        ts=now - 30,
    )

    # 1. reindex
    r_reindex = runner.invoke(app, ["memory", "reindex", "--role", "tutor", "-c", str(cfg_path)])
    assert r_reindex.exit_code == 0

    # 2. backfill-fts
    r_fts = runner.invoke(app, ["memory", "backfill-fts", "-c", str(cfg_path)])
    assert r_fts.exit_code == 0
    assert len(db.fts_search_messages("tutor", "品川")) == 2

    # 3. status
    r_status = runner.invoke(app, ["memory", "status", "--role", "tutor", "-c", str(cfg_path)])
    assert r_status.exit_code == 0
    assert "tutor" in r_status.stdout

    # 4. consolidate
    r_cons = runner.invoke(app, ["memory", "consolidate", "--role", "tutor", "-c", str(cfg_path)])
    assert r_cons.exit_code == 0

    # 5. backfill --dry-run with patched LLM
    class CliDryRunLLM(BaseLLM):
        async def generate(self, **kwargs: Any) -> LLMResponse:
            payload = {
                "episode_summary": "住在品川",
                "open_threads": [],
                "ops": [
                    {
                        "op": "upsert_bullet",
                        "file": "user/profile.md",
                        "section": "住址",
                        "text": "住在东京品川区",
                        "tags": ["#direct"],
                        "pinned": True,
                    }
                ],
            }
            return LLMResponse(
                content=json.dumps(payload, ensure_ascii=False),
                prompt_tokens=50,
                completion_tokens=30,
                total_tokens=80,
            )

    monkeypatch.setattr("kesoku.context.KesokuContext.get_llm", lambda *a, **kw: CliDryRunLLM())
    r_bf = runner.invoke(
        app,
        ["memory", "backfill", "--role", "tutor", "--since", "2020-01-01", "--dry-run", "-c", str(cfg_path)],
    )
    assert r_bf.exit_code == 0
    assert not (mem_dir / "user" / "profile.md").exists()  # dry-run never writes!

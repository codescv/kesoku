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
        f"- note_{i:02d}_topic_{i * 13}_detail_{i * 29}_ extra_{i * 47}_context_{i * 71} #inferred" for i in range(25)
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
        f"- alpha_{i:03d}_module_{i * 17}_index_{i * 31}_cache_{i * 53}_shard_{i * 97} #direct" for i in range(60)
    )
    sec_b = "\n".join(
        f"- beta_{i:03d}_deploy_{i * 19}_canary_{i * 37}_probe_{i * 61}_drain_{i * 89} #direct" for i in range(60)
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


def test_get_unextracted_messages_trims_to_clean_turn_boundary(tmp_path) -> None:
    """Verify `get_unextracted_messages` trims mid-turn slices back to the last USER boundary."""
    db = DatabaseManager(str(tmp_path / "trim.db"))
    db.init_tables()
    now = 1_760_000_000.0
    db.create_session(Session(id="s_trim", title="Trim", created_at=now, updated_at=now, role_name="coder"))

    def _m(mid: str, role: str, mtype: str, ts: float) -> None:
        db.save_message(
            Message(
                id=mid,
                session_id="s_trim",
                chatbot_id="cli",
                channel_id="ch1",
                sender="user" if role == MessageRole.USER else "assistant",
                role=role,
                type=mtype,
                content=f"msg {mid}",
                timestamp=ts,
                status=MessageStatus.PROCESSED if role == MessageRole.USER else MessageStatus.DELIVERED,
            )
        )

    # Turn 1: 4 rows (u1, tc1, tr1, a1)
    _m("u1", MessageRole.USER, MessageType.TEXT, now + 1)
    _m("tc1", MessageRole.TOOL, MessageType.TOOL_CALL, now + 2)
    _m("tr1", MessageRole.TOOL, MessageType.TOOL_RESULT, now + 3)
    _m("a1", MessageRole.ASSISTANT, MessageType.TEXT, now + 4)
    # Turn 2: 4 rows (u2, tc2, tr2, a2) -> limit=6 would cut right between tc2 and tr2!
    _m("u2", MessageRole.USER, MessageType.TEXT, now + 5)
    _m("tc2", MessageRole.TOOL, MessageType.TOOL_CALL, now + 6)
    _m("tr2", MessageRole.TOOL, MessageType.TOOL_RESULT, now + 7)
    _m("a2", MessageRole.ASSISTANT, MessageType.TEXT, now + 8)

    w1 = db.get_unextracted_messages("s_trim", limit=6)
    assert [m.id for m in w1] == ["u1", "tc1", "tr1", "a1"]

    w2 = db.get_unextracted_messages("s_trim", after_ts=w1[-1].timestamp, after_id=w1[-1].id, limit=6)
    assert [m.id for m in w2] == ["u2", "tc2", "tr2", "a2"]

    # Single giant turn exceeding limit keeps all `limit` rows so forward progress is guaranteed
    db.create_session(Session(id="s_giant", title="Giant", created_at=now, updated_at=now, role_name="coder"))
    db.save_message(
        Message(
            id="gu1",
            session_id="s_giant",
            chatbot_id="cli",
            channel_id="ch1",
            sender="user",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="giant",
            timestamp=now + 1,
            status=MessageStatus.PROCESSED,
        )
    )
    for idx in range(6):
        db.save_message(
            Message(
                id=f"gt_{idx}",
                session_id="s_giant",
                chatbot_id="cli",
                channel_id="ch1",
                sender="tool",
                role=MessageRole.TOOL,
                type=MessageType.TOOL_RESULT,
                content="out",
                timestamp=now + 2 + idx,
                status=MessageStatus.DELIVERED,
            )
        )
    giant_w1 = db.get_unextracted_messages("s_giant", limit=4)
    assert len(giant_w1) == 4 and giant_w1[0].id == "gu1"


@pytest.mark.asyncio
async def test_multi_window_backfill_carries_prior_markers_ignores_late_disk_and_runs_all_dry_run_windows(
    tmp_path, monkeypatch
) -> None:
    """Verify >200-msg sessions carry `prior_tool_calls`, ignore late `disk_entries`, and run all dry-run windows."""
    import tomli_w
    from rich.console import Console

    from kesoku.agent.tools.memory import session_search, session_view
    from kesoku.agent.tools.registry import ToolContext
    from kesoku.cli.memory import _run_backfill_async, run_backfill_fts_sync
    from kesoku.gateway.gateway import Gateway

    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", lambda text: [0.2] * 384)
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [[0.2] * 384 for _ in texts])

    cfg_dict = {
        "workspace": {"db_path": "kesoku.db", "roles_dir": "roles", "sessions_dir": "sessions"},
        "agent": {
            "memory_isolated_contexts": ["stage-*", "trpg-*"],
            "memory_filtered_contexts": ["companion-*"],
            "memory_backfill_context_markers": [
                {
                    "pattern": r"ctl\s+start\s+trpg\b",
                    "policy": "isolated",
                    "until": r"ctl\s+stop\s+trpg\b",
                },
                {
                    "pattern": r"ctl\s+start\s+companion\b",
                    "policy": "real_facts_only",
                    "until": r"ctl\s+stop\s+companion\b",
                },
                {
                    "pattern": r"ctl\s+start\s+stage\b",
                    "policy": "isolated",
                    "until": r"ctl\s+stop\s+stage\b",
                },
            ],
        },
    }
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(tomli_w.dumps(cfg_dict), encoding="utf-8")

    from kesoku.config import load_config

    cfg = load_config(str(cfg_path))
    db = DatabaseManager(cfg.workspace.db_path)
    db.init_tables()
    now = 1_760_000_000.0

    # Session 1 (`fab0e5a7`-style, >200 msgs = 115 turns / 232 rows):
    # Turn 1 starts `companion` (`real_facts_only`), tours Taj Mahal & mentions `小笨鸟` in Window 1 & Window 2,
    # and Turn 115 at the very end starts `stage` (`isolated`), leaving `10-stage#2.md` in relative `sessions/`!
    sid_mixed = "fab0e5a7_agra"
    db.create_session(
        Session(id=sid_mixed, title="泰姬陵之行", created_at=now, updated_at=now + 500, role_name="tutor")
    )
    mixed_ctx_dir = tmp_path / "sessions" / sid_mixed / "context"
    mixed_ctx_dir.mkdir(parents=True)
    (mixed_ctx_dir / "10-stage#2.md").write_text("Stage contest leftover on disk\n", encoding="utf-8")

    for i in range(1, 116):
        ts_base = now + i * 3.0
        u_text = "泰姬陵照片与小笨鸟昵称起源" if i in (5, 108) else f"阿格拉旅行见闻记录第 {i} 段"
        if i == 115:
            u_text = "进入虚拟舞台比赛"
        db.save_message(
            Message(
                id=f"mx_u_{i:03d}",
                session_id=sid_mixed,
                chatbot_id="discord",
                channel_id="ch_agra",
                sender="user",
                role=MessageRole.USER,
                type=MessageType.TEXT,
                content=u_text,
                timestamp=ts_base,
                status=MessageStatus.PROCESSED,
            )
        )
        if i == 1:
            db.save_message(
                Message(
                    id="mx_tc_001",
                    session_id=sid_mixed,
                    chatbot_id="discord",
                    channel_id="ch_agra",
                    sender="run_shell_command",
                    role=MessageRole.TOOL,
                    type=MessageType.TOOL_CALL,
                    content="ctl start companion",
                    timestamp=ts_base + 0.2,
                    status=MessageStatus.RESPONDED,
                    metadata={"tool_name": "run_shell_command", "tool_arguments": {"command": "ctl start companion"}},
                )
            )
        elif i == 115:
            db.save_message(
                Message(
                    id="mx_tc_115",
                    session_id=sid_mixed,
                    chatbot_id="discord",
                    channel_id="ch_agra",
                    sender="run_shell_command",
                    role=MessageRole.TOOL,
                    type=MessageType.TOOL_CALL,
                    content="ctl start stage",
                    timestamp=ts_base + 0.2,
                    status=MessageStatus.RESPONDED,
                    metadata={"tool_name": "run_shell_command", "tool_arguments": {"command": "ctl start stage"}},
                )
            )
        db.save_message(
            Message(
                id=f"mx_a_{i:03d}",
                session_id=sid_mixed,
                chatbot_id="discord",
                channel_id="ch_agra",
                sender="assistant",
                role=MessageRole.ASSISTANT,
                type=MessageType.TEXT,
                content="虚拟舞台虚构评委台词" if i == 115 else f"回复第 {i} 段",
                timestamp=ts_base + 1.0,
                status=MessageStatus.DELIVERED,
            )
        )

    # Session 2 (`f4157b08`-style, >200 msgs = 110 turns / 221 rows):
    # Turn 1 starts `trpg` (`isolated`) and stops BEFORE session end (so `sessions/` has NO leftover file on disk!).
    # Window 2 has zero TOOL_CALLs and MUST still inherit `isolated` from Window 1 via `prior_tool_calls`!
    sid_trpg = "f4157b08_trpg"
    db.create_session(
        Session(id=sid_trpg, title="长篇跑团", created_at=now + 600, updated_at=now + 1200, role_name="tutor")
    )
    for i in range(1, 111):
        ts_base = now + 600 + i * 3.0
        db.save_message(
            Message(
                id=f"tr_u_{i:03d}",
                session_id=sid_trpg,
                chatbot_id="discord",
                channel_id="ch_trpg",
                sender="user",
                role=MessageRole.USER,
                type=MessageType.TEXT,
                content=f"精灵弓箭手第 {i} 轮行动",
                timestamp=ts_base,
                status=MessageStatus.PROCESSED,
            )
        )
        if i == 1:
            db.save_message(
                Message(
                    id="tr_tc_001",
                    session_id=sid_trpg,
                    chatbot_id="discord",
                    channel_id="ch_trpg",
                    sender="run_shell_command",
                    role=MessageRole.TOOL,
                    type=MessageType.TOOL_CALL,
                    content="ctl start trpg",
                    timestamp=ts_base + 0.2,
                    status=MessageStatus.RESPONDED,
                    metadata={"tool_name": "run_shell_command", "tool_arguments": {"command": "ctl start trpg"}},
                )
            )
        db.save_message(
            Message(
                id=f"tr_a_{i:03d}",
                session_id=sid_trpg,
                chatbot_id="discord",
                channel_id="ch_trpg",
                sender="assistant",
                role=MessageRole.ASSISTANT,
                type=MessageType.TEXT,
                content=f"地下城叙述 {i}",
                timestamp=ts_base + 1.0,
                status=MessageStatus.DELIVERED,
            )
        )

    # 1. Run `backfill-fts` -> `fab0e5a7_agra` searchable by `泰姬陵`/`小笨鸟` with `include_isolated=False`!
    run_backfill_fts_sync(cfg, db)
    agra_digest = db.get_session_digest(sid_mixed)
    assert agra_digest is not None
    assert agra_digest["context_policy"] == "real_facts_only"
    assert "进入虚拟舞台比赛" not in agra_digest["last_user_msgs"]

    gw = Gateway(context=KesokuContext(config=cfg, db=db))
    await gw.db.set_channel_role("discord", "ch_cur", "tutor")
    await gw.db.create_session(
        Session(id="cur_sess_01", title="Now", created_at=now + 2000, updated_at=now + 2000, role_name="tutor")
    )
    await gw.db.set_active_session_for_channel("discord", "ch_cur", "cur_sess_01")
    t_ctx = ToolContext(
        session_id="cur_sess_01",
        session_workspace="cur_sess_01",
        chatbot_id="discord",
        channel_id="ch_cur",
        gateway=gw,
    )
    s_out = await session_search(query="小笨鸟", context=t_ctx)
    assert "#session:fab0e5a7" in s_out
    v_out = await session_view(session_id="#session:fab0e5a7", around_message_id="mx_u_108", context=t_ctx)
    assert "小笨鸟" in v_out
    assert "虚拟舞台虚构评委台词" not in v_out  # Turn 115 (`isolated`) stays hidden!

    # 2. Multi-window `--dry-run` traverses both windows of `fab0e5a7_agra` and skips both of `f4157b08_trpg`!
    llm_calls: list[str] = []

    class MultiWinLLM(BaseLLM):
        async def generate(self, prompt: str | None = None, **kwargs: Any) -> LLMResponse:
            llm_calls.append(prompt or "")
            payload = {
                "episode_summary": "阿格拉泰姬陵旅行",
                "open_threads": [],
                "ops": [
                    {
                        "op": "upsert_bullet",
                        "file": "travel.md",
                        "section": "行程",
                        "text": "游览了阿格拉泰姬陵",
                        "tags": ["#direct"],
                    }
                ],
            }
            return LLMResponse(
                content=json.dumps(payload, ensure_ascii=False),
                prompt_tokens=100,
                completion_tokens=40,
                total_tokens=140,
            )

    monkeypatch.setattr("kesoku.context.KesokuContext.get_llm", lambda *a, **kw: MultiWinLLM())
    dry_stats = await _run_backfill_async(
        cfg,
        db,
        role="tutor",
        since_ts=0.0,
        dry_run=True,
        budget_tokens=100_000,
        console=Console(quiet=True),
    )
    assert dry_stats["windows_extracted"] == 2  # Window 1 + Window 2 of `fab0e5a7_agra`!
    assert dry_stats["isolated_skipped"] == 2  # Window 1 + Window 2 of `f4157b08_trpg` (0 LLM tokens)!
    assert len(llm_calls) == 2
    # Window 2 of `fab0e5a7_agra` carried `real_facts_only` forward AND redacted Turn 115 (`isolated`) as `[omitted]`!
    assert "[scenario frame — keep only real-world facts]" in llm_calls[1]
    assert "虚拟舞台虚构评委台词" not in llm_calls[1]
    assert db.get_extraction_cursor(sid_mixed) is None  # dry-run left SQLite cursors untouched!

    # 3. Live `backfill` also completes all windows cleanly across trimmed <200-msg turn boundaries
    live_stats = await _run_backfill_async(
        cfg,
        db,
        role="tutor",
        since_ts=0.0,
        dry_run=False,
        budget_tokens=100_000,
        console=Console(quiet=True),
    )
    assert live_stats["windows_extracted"] == 2
    assert live_stats["isolated_skipped"] == 2
    final_cur = db.get_extraction_cursor(sid_mixed)
    assert final_cur is not None and final_cur["last_extracted_msg_id"] == "mx_a_115"
    final_trpg_cur = db.get_extraction_cursor(sid_trpg)
    assert final_trpg_cur is not None and final_trpg_cur["last_extracted_msg_id"] == "tr_a_110"


@pytest.mark.asyncio
async def test_minimal_thinking_unsupported_falls_back_to_low(tmp_path, monkeypatch) -> None:
    """Verify Gemini 400 `THINKING_LEVEL_MINIMAL` auto-falls back to `thinking_level='low'` and caches it."""
    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", lambda text: [0.1] * 384)
    monkeypatch.setattr("kesoku.utils.embedding.get_embeddings", lambda texts: [[0.1] * 384 for _ in texts])

    db_path = str(tmp_path / "fallback.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, roles_dir=str(tmp_path / "roles")))
    cfg.agent_working_dir = str(tmp_path)

    seen_levels: list[str] = []

    class StrictFlashLLM(BaseLLM):
        async def generate(self, **kwargs: Any) -> LLMResponse:
            lvl = getattr(self, "_last_extraction_thinking_level", "minimal")
            seen_levels.append(lvl)
            if lvl == "minimal":
                raise RuntimeError("400 INVALID_ARGUMENT: Thinking level is unsupported: THINKING_LEVEL_MINIMAL")
            return LLMResponse(
                content='{"episode_summary": "ok", "open_threads": [], "ops": []}',
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
            )

    ctx = KesokuContext(config=cfg, llm=StrictFlashLLM(), db=db)
    consolidator = MemoryConsolidator(ctx, prewarm_embeddings=False, enable_timers=False)

    now = 1_760_000_000.0
    db.create_session(Session(id="s_fb1", title="FB1", created_at=now, updated_at=now, role_name="coder"))
    _save_turn(db, sid="s_fb1", turn_idx=1, user_text="测试回退机制", asst_text="好的", ts=now + 1)
    out1 = await consolidator.run_episode_extract(EpisodeExtract(role="coder", session_id="s_fb1"))
    assert out1.status == "ok"
    assert seen_levels == ["minimal", "low"]

    # Subsequent extraction goes directly to `low` without another 400 round-trip
    _save_turn(db, sid="s_fb1", turn_idx=2, user_text="第二轮测试", asst_text="好的", ts=now + 5)
    out2 = await consolidator.run_episode_extract(EpisodeExtract(role="coder", session_id="s_fb1"))
    assert out2.status == "ok"
    assert seen_levels == ["minimal", "low", "low"]
    await consolidator.stop(wait_seconds=1.0)


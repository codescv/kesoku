"""Unit tests for Tier 0b `<recent_sessions>`, turn-start digest upsert, and `_memory_snapshot`."""

import time
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kesoku.agent.llm import BaseLLM, LLMResponse, ToolCallRequest
from kesoku.agent.session_digest import (
    DigestUpdater,
    fetch_recent_digests,
    format_recent_sessions_block,
    should_render,
)
from kesoku.agent.turn_executor import TurnExecutor
from kesoku.agent.turn_logger import TurnLogger
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, Session
from kesoku.gateway.gateway import Gateway


def test_turn_start_digest_solves_10s_switch_and_parallel_thread_and_excludes_cronjob(tmp_path) -> None:
    """Verify `touch_at_turn_start_sync` exposes Session A's message before LLM reply and skips cronjobs."""
    db = DatabaseManager(str(tmp_path / "digest.db"))
    db.init_tables()
    now = 1_760_000_000.0

    sess_a = Session(id="sessa111_tokyo", title="东京行程咨询", created_at=now, updated_at=now, role_name="traveler")
    sess_b = Session(
        id="sessb222_ramen", title="拉面店推荐", created_at=now + 10, updated_at=now + 10, role_name="traveler"
    )
    sess_cron = Session(
        id="cron9999_daily", title="Cronjob Morning", created_at=now + 5, updated_at=now + 5, role_name="traveler"
    )
    for s in (sess_a, sess_b, sess_cron):
        db.create_session(s)

    # 1. Session A starts a turn ("我下周去东京出差住新宿") — LLM has NOT replied yet (e.g. still generating TTS/media)
    msg_a = Message(
        id="ma_1",
        session_id="sessa111_tokyo",
        chatbot_id="discord",
        channel_id="thread_a",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="我下周去东京出差住新宿，帮我看看交通",
        timestamp=now + 1,
    )
    DigestUpdater.touch_at_turn_start_sync(db, sess_a, "traveler", msg_a)

    # 2. Cronjob message fires at `now + 5` — must NEVER enter `session_digests` (`H2`)
    msg_cron = Message(
        id="mc_1",
        session_id="cron9999_daily",
        chatbot_id="cronjob",
        channel_id="ch_cron",
        sender="Cronjob",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="自动晨报推送",
        timestamp=now + 5,
        metadata={"is_cronjob": True},
    )
    DigestUpdater.touch_at_turn_start_sync(db, sess_cron, "traveler", msg_cron)
    assert db.get_session_digest("cron9999_daily") is None

    # 3. 10 seconds later (`now + 10`), Session B bootstraps and immediately sees Session A's last user message!
    rows = fetch_recent_digests(db, "traveler", current_session_id="sessb222_ramen", now_ts=now + 10)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sessa111_tokyo"
    assert should_render(is_bootstrap=True, prev_user_ts=None, rows=rows) is True

    block = format_recent_sessions_block("traveler", rows, max_chars=1200)
    assert "<recent_sessions " in block
    assert "#session:sessa111" in block
    assert "去东京出差住新宿" in block


def test_should_render_consecutive_vs_interleaved_and_isolated_title_only(tmp_path) -> None:
    """Verify `<recent_sessions>` skips consecutive turns, renders on interleave, and hides `[isolated]` body."""
    db = DatabaseManager(str(tmp_path / "render.db"))
    db.init_tables()
    now = 1_760_000_000.0

    db.create_session(
        Session(id="sib_full_01", title="日常问答", created_at=now - 300, updated_at=now - 300, role_name="tutor")
    )
    db.create_session(
        Session(id="sib_iso_02", title="奇幻地下城战役", created_at=now - 200, updated_at=now - 200, role_name="tutor")
    )

    db.upsert_session_digest(
        "sib_full_01",
        "tutor",
        chatbot_id="discord",
        channel_id="c1",
        title="日常问答",
        started_at=now - 300,
        last_active_at=now - 100,
        turn_count=5,
        last_user_msgs=["乳糖不耐受喝什么"],
        context_policy="full",
        summary="讨论无乳糖饮品推荐",
    )
    db.upsert_session_digest(
        "sib_iso_02",
        "tutor",
        chatbot_id="discord",
        channel_id="c2",
        title="奇幻地下城战役",
        started_at=now - 200,
        last_active_at=now - 50,
        turn_count=12,
        last_user_msgs=["施放火球术"],
        context_policy="isolated",
        summary="不应泄露的虚拟剧情摘要",
    )

    rows = fetch_recent_digests(db, "tutor", current_session_id="cur_sess", now_ts=now)

    # Consecutive turn where `prev_user_ts` (`now - 10`) is AFTER all sibling `last_active_at` -> False (0 chars!)
    assert should_render(is_bootstrap=False, prev_user_ts=now - 10, rows=rows) is False

    # Cross-session interleave where a sibling was active at `now - 50 > prev_user_ts` (`now - 80`) -> True!
    assert should_render(is_bootstrap=False, prev_user_ts=now - 80, rows=rows) is True

    block = format_recent_sessions_block("tutor", rows, max_chars=1200)
    assert len(block) <= 1200
    assert "讨论无乳糖饮品推荐" in block
    assert "乳糖不耐受喝什么" in block
    # Isolated session shows title + `[isolated]`, NEVER `summary` or `last_user`!
    assert '"奇幻地下城战役"' in block
    assert "[isolated]" in block
    assert "施放火球术" not in block
    assert "不应泄露的虚拟剧情摘要" not in block


@pytest.mark.asyncio
async def test_turn_executor_memory_snapshot_byte_identical_across_tool_rounds(tmp_path, monkeypatch) -> None:
    """Verify `TurnExecutor._memory_snapshot` stays byte-identical across R1..R2 despite mid-turn memory edits."""
    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", lambda text: [1.0] + [0.0] * 383)

    db_path = str(tmp_path / "snap.db")
    db = DatabaseManager(db_path)
    db.init_tables()

    roles_dir = tmp_path / "roles"
    user_mem_dir = roles_dir / "tutor" / "memories" / "user"
    user_mem_dir.mkdir(parents=True)
    profile_path = user_mem_dir / "profile.md"
    profile_path.write_text(
        "---\ndescription: Core profile\nvisibility: PINNED\n---\n## 档案\n- 初始档案：常住大阪 #direct\n",
        encoding="utf-8",
    )

    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, roles_dir=str(roles_dir)))
    gw = Gateway(context=KesokuContext(config=cfg, db=db))
    await gw.db.set_channel_role("cli", "ch1", "tutor")

    now = time.time()
    # Sibling session so `<recent_sessions>` also renders on R1
    await gw.create_session("sib_0001", title="Sibling", role="tutor")
    db.upsert_session_digest(
        "sib_0001",
        "tutor",
        chatbot_id="cli",
        channel_id="ch_sib",
        title="上周课程",
        started_at=now - 600,
        last_active_at=now - 300,
        turn_count=4,
        last_user_msgs=["复习第10课"],
        context_policy="full",
    )

    await gw.create_session("sess_r1r2", title="Active", role="tutor")
    await gw.db.set_active_session_for_channel("cli", "ch1", "sess_r1r2")
    user_msg = Message(
        id="u_turn1",
        session_id="sess_r1r2",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="帮我查一下学习进度",
        status=MessageStatus.PENDING_AGENT,
        timestamp=now,
    )
    await gw.post(user_msg)

    round_user_wrappers: list[str] = []

    class TwoRoundLLM(BaseLLM):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self,
            prompt: str | None = None,
            system_prompt: str | None = None,
            history: list[Message] | None = None,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            self.calls += 1
            first_user = next(m for m in (history or []) if m.role == MessageRole.USER)
            round_user_wrappers.append(first_user.content)
            if self.calls == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCallRequest(id="tc_1", name="dummy_tool", arguments={})],
                    total_tokens=10,
                )
            return LLMResponse(content="已完成查询", total_tokens=10)

    from kesoku.agent.tool_runner import ToolRunner
    from kesoku.agent.tools.registry import ToolContext, ToolRegistry

    def dummy_tool() -> str:
        """Mutate pinned memory on disk and bump version mid-turn between R1 and R2."""
        from kesoku.agent.memory_store import bump_memory_index_version

        profile_path.write_text(
            "---\ndescription: Core profile\nvisibility: PINNED\n---\n## 档案\n- 轮中修改：搬到京都 #direct\n",
            encoding="utf-8",
        )
        bump_memory_index_version("tutor")
        return "ok"

    llm = TwoRoundLLM()
    reg = ToolRegistry()
    reg.register(dummy_tool)
    t_ctx = ToolContext(
        session_id="sess_r1r2",
        session_workspace="sess_r1r2",
        chatbot_id="cli",
        channel_id="ch1",
        gateway=gw,
    )
    tool_runner = ToolRunner(reg, t_ctx)
    turn_logger = MagicMock(spec=TurnLogger)

    ctx = KesokuContext(config=cfg, llm=llm, db=db)
    executor = TurnExecutor("sess_r1r2", gw, tool_runner, turn_logger, context=ctx)

    worker = MagicMock()
    type(worker).running = PropertyMock(return_value=True)

    async def _no_pivot(m: Message) -> Message:
        return m

    worker.drain_queue_and_pivot.side_effect = _no_pivot
    worker.queue_empty.return_value = True

    staging_dir = str(tmp_path / "sessions" / "sess_r1r2")
    with (
        patch("kesoku.context.get_config", return_value=cfg),
        patch("kesoku.utils.path.get_config", return_value=cfg),
    ):
        await executor.process_turn(current_msg=user_msg, worker=worker, session_staging_dir=staging_dir)

    assert len(round_user_wrappers) == 2
    assert "<memory_pinned" in round_user_wrappers[0]
    assert "<recent_sessions" in round_user_wrappers[0]
    # Ordering check: <memory_pinned> precedes <recent_sessions> which precedes <current_message>
    assert round_user_wrappers[0].index("<memory_pinned") < round_user_wrappers[0].index("<recent_sessions")
    assert round_user_wrappers[0].index("<recent_sessions") < round_user_wrappers[0].index("<current_message")
    # Byte-identical across R1 and R2 despite mid-turn memory mutation (`H1-dev`)!
    assert round_user_wrappers[0] == round_user_wrappers[1]

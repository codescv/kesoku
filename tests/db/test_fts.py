"""Unit tests for SQLite FTS5 indexing/querying and `session_search` / `session_view` guards."""

import time

import pytest

from kesoku.agent.tools.memory import session_search, session_view
from kesoku.agent.tools.registry import ToolContext, default_registry
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message, Session
from kesoku.gateway.gateway import Gateway
from kesoku.utils.text import fts_tokenize


def _index_msg(
    db: DatabaseManager,
    *,
    mid: str,
    sid: str,
    persona_role: str,
    policy: str,
    speaker: str,
    ts: float,
    content: str,
) -> None:
    db.save_message(
        Message(
            id=mid,
            session_id=sid,
            chatbot_id="discord",
            channel_id=f"chan_{sid}",
            sender=speaker,
            role=MessageRole.USER if speaker == "user" else MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content=content,
            timestamp=ts,
            status=MessageStatus.RESPONDED,
            metadata={"context_policy": policy},
        )
    )
    db.fts_insert_messages(
        [
            {
                "message_id": mid,
                "session_id": sid,
                "persona_role": persona_role,
                "context_policy": policy,
                "speaker_role": speaker,
                "ts": ts,
                "raw_excerpt": content[:400],
                "text": fts_tokenize(content, for_query=False),
            }
        ]
    )


def test_messages_fts_cjk_substrings_trailing_char_furigana_and_scoping(tmp_path) -> None:
    """Verify CJK substring `新宿` in `住新宿`, `猫*` final-char match, furigana, role scoping, and isolation."""
    db = DatabaseManager(str(tmp_path / "fts.db"))
    db.init_tables()
    now = time.time()

    for sid, role in (("s_tutor_1", "tutor"), ("s_tutor_iso", "tutor"), ("s_other", "coder")):
        db.create_session(Session(id=sid, title=sid, created_at=now, updated_at=now, role_name=role))

    _index_msg(
        db,
        mid="m_shinjuku",
        sid="s_tutor_1",
        persona_role="tutor",
        policy="full",
        speaker="user",
        ts=now - 50,
        content="我这次出差住新宿车站旁边，还养了一只黑猫",
    )
    _index_msg(
        db,
        mid="m_furi",
        sid="s_tutor_1",
        persona_role="tutor",
        policy="real_facts_only",
        speaker="assistant",
        ts=now - 40,
        content="渋谷(しぶや)から秋葉原(あきはばら)まで電車で行けます。",
    )
    _index_msg(
        db,
        mid="m_iso",
        sid="s_tutor_iso",
        persona_role="tutor",
        policy="isolated",
        speaker="user",
        ts=now - 30,
        content="在奇幻王国的住新宿旅馆里发现了一只黑猫",
    )
    _index_msg(
        db,
        mid="m_other_role",
        sid="s_other",
        persona_role="coder",
        policy="full",
        speaker="user",
        ts=now - 20,
        content="coder 住新宿调试代码",
    )

    # 1. `新宿` matches `住新宿`, scoped strictly inside MATCH to `tutor` and excluding `isolated` by default
    hits = db.fts_search_messages("tutor", "新宿")
    assert [h["message_id"] for h in hits] == ["m_shinjuku"]

    # 2. `include_isolated=True` surfaces the isolated row for `tutor`, still never leaking `coder`
    hits_with_iso = db.fts_search_messages("tutor", "新宿", include_isolated=True)
    assert {h["message_id"] for h in hits_with_iso} == {"m_shinjuku", "m_iso"}

    # 3. Single-char prefix `猫` (`猫*`) matches `养了一只黑猫` where `猫` is the final character (`R2-3`)
    cat_hits = db.fts_search_messages("tutor", "猫")
    assert [h["message_id"] for h in cat_hits] == ["m_shinjuku"]

    # 4. Kanji query `秋葉原` matches furigana-annotated `秋葉原(あきはばら)` (`C3`)
    furi_hits = db.fts_search_messages("tutor", "秋葉原")
    assert [h["message_id"] for h in furi_hits] == ["m_furi"]

    # 5. Deleting a session cleans up its FTS rows
    db.delete_session("s_tutor_1")
    assert db.fts_search_messages("tutor", "新宿") == []


@pytest.mark.asyncio
async def test_session_search_long_message_hydration_and_session_view_guards(tmp_path, monkeypatch) -> None:
    """Verify `session_search` PK snippet hydration (>400 chars) and `session_view` authorization guards."""
    monkeypatch.setattr("kesoku.utils.embedding.get_embedding", lambda text: [0.0] * 384)

    names = [f.__name__ for f in default_registry.get_tools_list()]
    assert "session_search" in names
    assert "session_view" in names

    db_path = str(tmp_path / "tools_fts.db")
    db = DatabaseManager(db_path)
    db.init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=db_path, roles_dir=str(tmp_path / "roles")))
    gw = Gateway(context=KesokuContext(config=cfg, db=db))

    now = time.time()
    await gw.db.set_channel_role("discord", "ch_active", "tutor")

    # Active session (excluded from self-hits in session_search)
    await gw.db.create_session(
        Session(id="active01_full", title="Current", created_at=now, updated_at=now, role_name="tutor")
    )
    await gw.db.set_active_session_for_channel("discord", "ch_active", "active01_full")

    # Sibling 1 (tutor, full): hit appears AFTER 450 chars so 400-char raw_excerpt lacks it (`R2-3`)
    await gw.db.create_session(
        Session(
            id="sib11111_lesson", title="标日第12课复习", created_at=now - 600, updated_at=now - 500, role_name="tutor"
        )
    )
    db.upsert_session_digest(
        "sib11111_lesson",
        "tutor",
        chatbot_id="discord",
        channel_id="ch_sib1",
        title="标日第12课复习",
        started_at=now - 600,
        last_active_at=now - 500,
        turn_count=3,
        last_user_msgs=["开始下一课"],
        context_policy="full",
        summary="复习了第12课语法并准备进入第13课",
    )
    long_prefix = "这是一段很长的课堂背景铺垫文字。" * 35  # > 500 chars
    long_msg_text = f"{long_prefix}关键知识点是小笨鸟贴纸和授受动词用法。"
    _index_msg(
        db,
        mid="m_long_hit",
        sid="sib11111_lesson",
        persona_role="tutor",
        policy="full",
        speaker="assistant",
        ts=now - 500,
        content=long_msg_text,
    )

    # Sibling 2 (tutor, isolated): must be blocked unless include_isolated=True
    await gw.db.create_session(
        Session(
            id="iso22222_quest", title="海岛奇幻战役", created_at=now - 400, updated_at=now - 300, role_name="tutor"
        )
    )
    db.upsert_session_digest(
        "iso22222_quest",
        "tutor",
        chatbot_id="discord",
        channel_id="ch_iso",
        title="海岛奇幻战役",
        started_at=now - 400,
        last_active_at=now - 300,
        turn_count=5,
        last_user_msgs=["打开宝箱"],
        context_policy="isolated",
    )
    _index_msg(
        db,
        mid="m_iso_secret",
        sid="iso22222_quest",
        persona_role="tutor",
        policy="isolated",
        speaker="user",
        ts=now - 300,
        content="发现了神秘黑曜石钥匙",
    )

    # Sibling 3 (different role `coder`): must NEVER be viewable from `tutor`
    await gw.db.create_session(
        Session(
            id="cod33333_work", title="Coder Session", created_at=now - 200, updated_at=now - 100, role_name="coder"
        )
    )
    db.upsert_session_digest(
        "cod33333_work",
        "coder",
        chatbot_id="discord",
        channel_id="ch_coder",
        title="Coder Session",
        started_at=now - 200,
        last_active_at=now - 100,
        turn_count=2,
        last_user_msgs=["部署服务"],
        context_policy="full",
    )
    _index_msg(
        db,
        mid="m_coder_msg",
        sid="cod33333_work",
        persona_role="coder",
        policy="full",
        speaker="user",
        ts=now - 100,
        content="机密部署密钥",
    )

    # Sibling 4 (cronjob session): must NEVER be viewable (`H2`, `M2`)
    await gw.db.create_session(
        Session(id="crn44444_job", title="Cronjob Check", created_at=now - 90, updated_at=now - 80, role_name="tutor")
    )
    db.save_message(
        Message(
            id="m_cron_1",
            session_id="crn44444_job",
            chatbot_id="cronjob",
            channel_id="ch_cron",
            sender="Cronjob",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="定时提醒任务",
            timestamp=now - 80,
            status=MessageStatus.RESPONDED,
            metadata={"is_cronjob": True},
        )
    )

    ctx = ToolContext(
        session_id="active01_full",
        session_workspace="active01_full",
        chatbot_id="discord",
        channel_id="ch_active",
        gateway=gw,
    )

    # 1. `session_search` hydrates full message via PK lookup so snippet contains `小笨鸟贴纸` past char 400
    search_out = await session_search(query="小笨鸟贴纸", context=ctx)
    assert "#session:sib11111" in search_out
    assert "小笨鸟贴纸" in search_out
    assert "[msg:m_long_hit]" in search_out

    # 2. `session_view` succeeds on authorized sibling via `#session:<id8>` handle
    view_ok = await session_view(session_id="#session:sib11111", around_message_id="m_long_hit", context=ctx)
    assert "标日第12课复习" in view_ok
    assert "小笨鸟贴纸" in view_ok

    # 3. `session_view` rejects isolated session by default (`not found`), allows with `include_isolated=True`
    view_iso_denied = await session_view(session_id="iso22222_quest", context=ctx)
    assert "not found" in view_iso_denied
    view_iso_allowed = await session_view(session_id="iso22222_quest", include_isolated=True, context=ctx)
    assert "黑曜石钥匙" in view_iso_allowed

    # 4. `session_view` rejects cross-role session even if `include_isolated=True`
    view_cross_role = await session_view(session_id="cod33333_work", include_isolated=True, context=ctx)
    assert "not found" in view_cross_role

    # 5. `session_view` rejects cronjob session
    view_cron = await session_view(session_id="crn44444_job", include_isolated=True, context=ctx)
    assert "not found" in view_cron

"""Unit tests for SessionNamer."""

from pathlib import Path

import pytest

from kesoku.agent.llm import LLMResponse, MockLLM
from kesoku.agent.namer import SessionNamer, parse_title_response
from kesoku.config import KesokuConfig, WorkspaceConfig
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.context import KesokuContext
from kesoku.db import DatabaseManager, Message
from kesoku.gateway.gateway import Gateway


@pytest.fixture
def temp_db(tmp_path: pytest.TempPathFactory) -> str:
    """Provide a temporary database path."""
    return str(tmp_path / "test_namer.db")


@pytest.mark.asyncio
async def test_session_namer_success(temp_db: str) -> None:
    """Test successful auto-naming."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    ctx = KesokuContext(config=cfg)
    gw = Gateway(context=ctx)

    await gw.create_session("sess1", title="New Session")

    # Seed first turn
    msg_user = Message(
        session_id="sess1",
        chatbot_id="discord",
        channel_id="ch1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello, who are you?",
        status=MessageStatus.PROCESSED,
    )
    msg_assistant = Message(
        session_id="sess1",
        chatbot_id="discord",
        channel_id="ch1",
        sender="assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="I am Kesoku AI.",
        status=MessageStatus.DELIVERED,
    )
    await gw.db.save_message(msg_user)
    await gw.db.save_message(msg_assistant)

    llm = MockLLM(responses=[LLMResponse(content="Kesoku Introduction", tool_calls=[])])

    namer = SessionNamer(db=gw.db, gateway=gw, llm=llm)

    success = await namer.auto_rename_session("sess1")
    assert success is True

    # Verify session title updated in DB
    session = await gw.db.get_session("sess1")
    assert session.title == "Kesoku Introduction"

    # Verify rename message posted
    messages = await gw.db.get_messages_by_filters(filters={"session_id": "sess1"})
    rename_msgs = [m for m in messages if m.role == MessageRole.SYSTEM and m.type == MessageType.SESSION_RENAME]
    assert len(rename_msgs) == 1
    assert rename_msgs[0].content == "Kesoku Introduction"
    assert rename_msgs[0].chatbot_id == "discord"
    assert rename_msgs[0].channel_id == "ch1"


@pytest.mark.asyncio
async def test_session_namer_wrong_turns_count(temp_db: str) -> None:
    """Test that naming is skipped if turns count is not 1 or 2."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    ctx = KesokuContext(config=cfg)
    gw = Gateway(context=ctx)

    await gw.create_session("sess1", title="New Session")

    namer = SessionNamer(db=gw.db, gateway=gw, llm=MockLLM())

    # 0 turns
    success = await namer.auto_rename_session("sess1")
    assert success is False

    # Seed 3 turns (3 user messages)
    for i in range(3):
        msg = Message(
            session_id="sess1",
            chatbot_id="discord",
            channel_id="ch1",
            sender="user",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=f"msg {i}",
            status=MessageStatus.PROCESSED,
        )
        await gw.db.save_message(msg)

    success = await namer.auto_rename_session("sess1")
    assert success is False


@pytest.mark.asyncio
async def test_session_namer_empty_response(temp_db: str) -> None:
    """Test that naming fails if LLM returns empty response."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    ctx = KesokuContext(config=cfg)
    gw = Gateway(context=ctx)

    await gw.create_session("sess1", title="New Session")

    msg_user = Message(
        session_id="sess1",
        chatbot_id="discord",
        channel_id="ch1",
        sender="user",
        role=MessageRole.USER,
        type=MessageType.TEXT,
        content="Hello",
        status=MessageStatus.PROCESSED,
    )
    msg_assistant = Message(
        session_id="sess1",
        chatbot_id="discord",
        channel_id="ch1",
        sender="assistant",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="Hi",
        status=MessageStatus.DELIVERED,
    )
    await gw.db.save_message(msg_user)
    await gw.db.save_message(msg_assistant)

    llm = MockLLM(responses=[LLMResponse(content="", tool_calls=[])])

    namer = SessionNamer(db=gw.db, gateway=gw, llm=llm)

    success = await namer.auto_rename_session("sess1")
    assert success is False

    # Title should not be updated
    session = await gw.db.get_session("sess1")
    assert session.title == "New Session"


# ---------------------------------------------------------------------------------------------------------------
# v0.3.14 hybrid naming: title_source locks, context frontmatter, provisional upgrades.


async def _gateway(temp_db: str) -> Gateway:
    """Build a Gateway over a fresh temp database."""
    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(workspace=WorkspaceConfig(db_path=temp_db))
    return Gateway(context=KesokuContext(config=cfg))


async def _seed_turn(gw: Gateway, user: str, assistant: str, session_id: str = "sess1") -> None:
    """Persist one user/assistant exchange."""
    for role, content, status in (
        (MessageRole.USER, user, MessageStatus.PROCESSED),
        (MessageRole.ASSISTANT, assistant, MessageStatus.DELIVERED),
    ):
        await gw.db.save_message(
            Message(
                session_id=session_id,
                chatbot_id="discord",
                channel_id="ch1",
                sender=role.value,
                role=role,
                type=MessageType.TEXT,
                content=content,
                status=status,
            )
        )


async def _renames(gw: Gateway, session_id: str = "sess1") -> list[str]:
    """Return contents of posted SESSION_RENAME messages."""
    msgs = await gw.db.get_messages_by_filters(filters={"session_id": session_id})
    return [m.content for m in msgs if m.type == MessageType.SESSION_RENAME]


def _write_context(staging: Path, name: str, title: str, body: str = "Scene body") -> None:
    """Write `context/<name>` with a `session_title` frontmatter."""
    ctx_dir = staging / "context"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    (ctx_dir / name).write_text(f"---\nsession_title: {title}\n---\n{body}\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_json_reply_sets_auto_source(temp_db: str) -> None:
    """A JSON reply with provisional=false stores the title with source 'auto'."""
    gw = await _gateway(temp_db)
    await gw.create_session("sess1", title="New Session")
    await _seed_turn(gw, "Tell me about tide pools", "Tide pools are ...")
    llm = MockLLM(responses=[LLMResponse(content='```json\n{"title": "潮汐池生态", "provisional": false}\n```')])
    assert await SessionNamer(db=gw.db, gateway=gw, llm=llm).auto_rename_session("sess1") is True
    sess = await gw.db.get_session("sess1")
    assert (sess.title, sess.title_source) == ("潮汐池生态", "auto")
    assert await _renames(gw) == ["潮汐池生态"]


@pytest.mark.asyncio
async def test_provisional_title_upgrades_to_auto_then_locks(temp_db: str) -> None:
    """A provisional title is replaced on a later turn; once 'auto' the LLM is not consulted again."""
    gw = await _gateway(temp_db)
    await gw.create_session("sess1", title="New Session")
    await _seed_turn(gw, "start", "Pick a setting: A / B / C")
    llm = MockLLM(
        responses=[
            LLMResponse(content='{"title": "Setting Menu", "provisional": true}'),
            LLMResponse(content='{"title": "Harbor Mystery", "provisional": false}'),
            LLMResponse(content='{"title": "SHOULD NOT BE USED", "provisional": false}'),
        ]
    )
    namer = SessionNamer(db=gw.db, gateway=gw, llm=llm)
    assert await namer.auto_rename_session("sess1") is True
    assert (await gw.db.get_session("sess1")).title_source == "provisional"

    await _seed_turn(gw, "B please", "The harbor fog rolls in ...")
    assert await namer.auto_rename_session("sess1") is True
    sess = await gw.db.get_session("sess1")
    assert (sess.title, sess.title_source) == ("Harbor Mystery", "auto")

    await _seed_turn(gw, "go on", "...")
    assert await namer.auto_rename_session("sess1") is False
    assert (await gw.db.get_session("sess1")).title == "Harbor Mystery"
    assert await _renames(gw) == ["Setting Menu", "Harbor Mystery"]


@pytest.mark.asyncio
async def test_manual_title_is_never_overwritten(temp_db: str, tmp_path: Path) -> None:
    """A manual title blocks both context and LLM naming."""
    gw = await _gateway(temp_db)
    await gw.create_session("sess1", title="New Session")
    await gw.db.update_session_title("sess1", "My Name", title_source="manual")
    await _seed_turn(gw, "hi", "hello")
    _write_context(tmp_path, "10-scene.md", "Context Theme")
    llm = MockLLM(responses=[LLMResponse(content="LLM Title")])
    assert await SessionNamer(db=gw.db, gateway=gw, llm=llm).auto_rename_session("sess1", str(tmp_path)) is False
    sess = await gw.db.get_session("sess1")
    assert (sess.title, sess.title_source) == ("My Name", "manual")
    assert await _renames(gw) == []


@pytest.mark.asyncio
async def test_context_title_wins_and_initial_theme_is_kept(temp_db: str, tmp_path: Path) -> None:
    """The first context session_title overrides LLM titles and later context changes never rename it."""
    gw = await _gateway(temp_db)
    await gw.create_session("sess1", title="New Session")
    await gw.db.update_session_title("sess1", "LLM Guess", title_source="auto")
    await _seed_turn(gw, "hi", "hello")
    _write_context(tmp_path, "10-scene.md", "Chapter One Theme")
    namer = SessionNamer(db=gw.db, gateway=gw, llm=MockLLM(responses=[]))
    assert await namer.auto_rename_session("sess1", staging_dir=str(tmp_path)) is True
    sess = await gw.db.get_session("sess1")
    assert (sess.title, sess.title_source) == ("Chapter One Theme", "context")

    _write_context(tmp_path, "10-scene.md", "Chapter Two Theme")
    assert await namer.auto_rename_session("sess1", staging_dir=str(tmp_path)) is False
    assert (await gw.db.get_session("sess1")).title == "Chapter One Theme"
    assert await _renames(gw) == ["Chapter One Theme"]


@pytest.mark.asyncio
async def test_context_title_upgrades_provisional_without_llm(temp_db: str, tmp_path: Path) -> None:
    """A provisional title is replaced by a context title even after the LLM naming window."""
    gw = await _gateway(temp_db)
    await gw.create_session("sess1", title="New Session")
    await gw.db.update_session_title("sess1", "Menu", title_source="provisional")
    for i in range(8):
        await _seed_turn(gw, f"u{i}", f"a{i}")
    _write_context(tmp_path, "20-mode.md", "Ocean Voyage")
    namer = SessionNamer(db=gw.db, gateway=gw, llm=MockLLM(responses=[]))
    assert await namer.auto_rename_session("sess1", staging_dir=str(tmp_path)) is True
    assert (await gw.db.get_session("sess1")).title_source == "context"


@pytest.mark.asyncio
async def test_context_title_equal_to_current_locks_without_rename(temp_db: str, tmp_path: Path) -> None:
    """When the context title already matches, the source is locked but no rename is posted."""
    gw = await _gateway(temp_db)
    await gw.create_session("sess1", title="Same Theme")
    _write_context(tmp_path, "10-scene.md", "Same Theme")
    namer = SessionNamer(db=gw.db, gateway=gw, llm=MockLLM(responses=[]))
    assert await namer.auto_rename_session("sess1", staging_dir=str(tmp_path)) is False
    assert (await gw.db.get_session("sess1")).title_source == "context"
    assert await _renames(gw) == []


@pytest.mark.asyncio
async def test_context_rename_uses_channel_binding_without_messages(temp_db: str, tmp_path: Path) -> None:
    """Without messages the rename is routed through the session's channel binding."""
    gw = await _gateway(temp_db)
    await gw.create_session("sess1", title="New Session", chatbot_id="app", channel_id="app:sess1")
    _write_context(tmp_path, "10-scene.md", "Bound Theme")
    namer = SessionNamer(db=gw.db, gateway=gw, llm=MockLLM(responses=[]))
    assert await namer.auto_rename_session("sess1", staging_dir=str(tmp_path)) is True
    msgs = await gw.db.get_messages_by_filters(filters={"session_id": "sess1"})
    rename = [m for m in msgs if m.type == MessageType.SESSION_RENAME]
    assert rename and (rename[0].chatbot_id, rename[0].channel_id) == ("app", "app:sess1")


@pytest.mark.asyncio
async def test_llm_window_and_transcript_filtering(temp_db: str) -> None:
    """Naming stops after six turns; scaffold messages and spoilers never reach the naming prompt."""
    gw = await _gateway(temp_db)
    await gw.create_session("sess1", title="New Session")
    await gw.db.save_message(
        Message(
            session_id="sess1",
            chatbot_id="discord",
            channel_id="ch1",
            sender="system",
            role=MessageRole.USER,
            content="SCAFFOLD-MARKER",
            metadata={"is_scaffold": True},
            status=MessageStatus.PROCESSED,
        )
    )
    await _seed_turn(gw, "Plan a garden", "Sure <spoiler>SECRET-MARKER</spoiler> " + "x" * 1000)
    captured: list[str] = []

    class CapturingLLM(MockLLM):
        async def generate(self, prompt=None, system_prompt=None, history=None, tools=None, **kwargs):  # type: ignore[override]
            captured.append(history[-1].content)
            return LLMResponse(content="Garden Plan", tool_calls=[])

    namer = SessionNamer(db=gw.db, gateway=gw, llm=CapturingLLM())
    assert await namer.auto_rename_session("sess1") is True
    assert "SCAFFOLD-MARKER" not in captured[0] and "SECRET-MARKER" not in captured[0]
    assert "x" * 400 not in captured[0]

    await gw.db.update_session_title("sess1", "Garden Plan", title_source="")
    for i in range(6):
        await _seed_turn(gw, f"u{i}", f"a{i}")
    assert await namer.auto_rename_session("sess1") is False


def test_parse_title_response_variants() -> None:
    """JSON (fenced or bare) and plain-text replies are parsed; quotes are stripped."""
    assert parse_title_response('{"title": "A", "provisional": true}') == ("A", True)
    assert parse_title_response('```json\n{"title": "B"}\n```') == ("B", False)
    assert parse_title_response('Sure: {"title": "C", "provisional": false}') == ("C", False)
    assert parse_title_response('"Plain Title"') == ("Plain Title", False)
    assert parse_title_response("「标题」\nextra") == ("标题", False)
    assert parse_title_response("") == ("", False)


@pytest.mark.asyncio
async def test_translator_card_cleaning_and_two_stage_naming(temp_db: str) -> None:
    """Translator YAML cards strip ruby/tokens noise and upgrade provisional launch titles to topic titles."""
    gw = await _gateway(temp_db)
    await gw.create_session("sess1", title="New Session", chatbot_id="app", channel_id="app:sess1")

    await _seed_turn(
        gw,
        "进入通用极速翻译模式：母语 zh-CN ⇄ 常用外语 ja（常用外语：ja、en、ko），零寒暄。",
        "翻译台已就绪。",
    )
    captured: list[str] = []
    replies = [
        '{"title": "中日翻译", "provisional": true}',
        '{"title": "用户文档日译", "provisional": false}',
    ]

    class CapturingLLM(MockLLM):
        async def generate(self, prompt=None, system_prompt=None, history=None, tools=None, **kwargs):  # type: ignore[override]
            captured.append(history[-1].content)
            return LLMResponse(content=replies.pop(0), tool_calls=[])

    namer = SessionNamer(db=gw.db, gateway=gw, llm=CapturingLLM())
    assert await namer.auto_rename_session("sess1") is True
    sess = await gw.db.get_session("sess1")
    assert (sess.title, sess.title_source) == ("中日翻译", "provisional")

    card_reply = (
        "```card\n"
        "kind: translation\n"
        "src_lang: zh-CN\n"
        "dst_lang: ja\n"
        "source: 这份用户文档是AI生成的，请帮忙确认一下\n"
        "source_ruby: <ruby>漢字<rt>かんじ</rt></ruby>" + "x" * 400 + "\n"
        "target: このユーザー向けドキュメントはAIが作成したものです。\n"
        "target_ruby: このユーザー<ruby>向<rt>む</rt></ruby>け<ruby>文書<rt>ぶんしょ</rt></ruby>\n"
        "tokens:\n"
        "  - surface: ユーザー\n"
        "    reading: ユーザー\n"
        "    pos: 名词\n"
        "back_translation: 这份面向用户的文档是AI生成的。\n"
        "```\n"
        "注：<ruby>確認<rt>かくにん</rt></ruby>一下"
    )
    await _seed_turn(gw, "这份用户文档是AI生成的，请帮忙确认一下", card_reply)
    assert await namer.auto_rename_session("sess1") is True
    sess = await gw.db.get_session("sess1")
    assert (sess.title, sess.title_source) == ("用户文档日译", "auto")
    assert "source_ruby:" not in captured[1]
    assert "target_ruby:" not in captured[1]
    assert "surface: ユーザー" not in captured[1]
    assert "このユーザー向けドキュメントはAIが作成したものです。" in captured[1]
    assert "back_translation: 这份面向用户的文档是AI生成的。" in captured[1]
    assert "<rt>" not in captured[1] and "確認一下" in captured[1]

    # Turn 3: already `auto`, must not call LLM again.
    await _seed_turn(gw, "再翻译一句", "好的")
    assert await namer.auto_rename_session("sess1") is False


@pytest.mark.asyncio
async def test_session_namer_fallback_on_unsupported_thinking_level(temp_db: str) -> None:
    """When primary LLM fails with THINKING_LEVEL_MINIMAL, SessionNamer retries on fallback_llm and promotes it."""
    gw = await _gateway(temp_db)
    await gw.create_session("sess1", title="New Session", chatbot_id="app", channel_id="app:sess1")
    await _seed_turn(gw, "印度人口最多的城市是哪些", "孟买、德里、班加罗尔等。")

    primary_calls = 0

    class FailingLLM(MockLLM):
        async def generate(self, prompt=None, system_prompt=None, history=None, tools=None, **kwargs):  # type: ignore[override]
            nonlocal primary_calls
            primary_calls += 1
            raise RuntimeError(
                "ClientError: 400 INVALID_ARGUMENT. Thinking level is unsupported: THINKING_LEVEL_MINIMAL"
            )

    fallback = MockLLM(responses=[LLMResponse(content='{"title": "印度人口最多城市", "provisional": false}')])
    namer = SessionNamer(db=gw.db, gateway=gw, llm=FailingLLM(), fallback_llm=fallback)

    assert await namer.auto_rename_session("sess1") is True
    sess = await gw.db.get_session("sess1")
    assert (sess.title, sess.title_source) == ("印度人口最多城市", "auto")
    assert primary_calls == 1
    assert namer.llm is fallback
    assert namer.fallback_llm is None


def test_session_worker_namer_llm_uses_low_thinking_level(temp_db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """SessionWorker._namer_llm must request thinking_level='low' (never 'minimal') and return a fallback."""
    from kesoku.agent.agent import SessionWorker
    from kesoku.config import AgentConfig, GeminiConfig

    DatabaseManager(temp_db).init_tables()
    cfg = KesokuConfig(
        workspace=WorkspaceConfig(db_path=temp_db),
        agent=AgentConfig(llm="gemini"),
        gemini=GeminiConfig(model_name="gemini-4-hydrogen", context_model_name="gemini-3.8-flash"),
    )
    ctx = KesokuContext(config=cfg)
    gw = Gateway(context=ctx)

    calls: list[tuple[str | None, str | None]] = []
    session_llm = MockLLM()
    flash_llm = MockLLM()

    def fake_get_llm(model_name=None, thinking_level=None, **kwargs):
        calls.append((model_name, thinking_level))
        return flash_llm if model_name == "gemini-3.8-flash" else session_llm

    monkeypatch.setattr(ctx, "get_llm", fake_get_llm)
    worker = SessionWorker(session_id="sess1", gateway=gw, context=ctx)
    primary, fallback = worker._namer_llm()
    assert primary is flash_llm
    assert fallback is session_llm
    assert ("gemini-3.8-flash", "low") in calls
    assert all(tl != "minimal" for _, tl in calls)

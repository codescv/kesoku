"""Unit tests for `context_policy.py` and `context_dir.py` frontmatter parsing (`C1`, `R2-4`)."""

import pytest

from kesoku.agent.context_dir import collect_context_entries, parse_context_frontmatter
from kesoku.agent.context_policy import (
    infer_backfill_turn_policies,
    most_restrictive_policy,
    resolve_context_policy,
)
from kesoku.config import AgentConfig, BackfillContextMarker
from kesoku.constants import MessageRole, MessageType
from kesoku.db.models import Message


def test_resolve_context_policy_precedence_and_globs() -> None:
    """Verify `isolated > real_facts_only > full` and frontmatter overriding glob patterns (`C1`)."""
    cfg = AgentConfig(
        memory_isolated_contexts=["sandbox-*", "trpg-*"],
        memory_filtered_contexts=["companion-*", "lesson-*", "scene"],
    )

    # Default with no active blocks or unrecognized blocks -> full
    assert resolve_context_policy([], {}, cfg) == "full"
    assert resolve_context_policy(["notes", "running_jobs"], {}, cfg) == "full"

    # Filtered glob -> real_facts_only
    assert resolve_context_policy(["notes", "companion-daily"], {}, cfg) == "real_facts_only"
    assert resolve_context_policy(["scene"], {}, cfg) == "real_facts_only"

    # Isolated glob beats real_facts_only and full
    assert resolve_context_policy(["notes", "companion-daily", "trpg-quest"], {}, cfg) == "isolated"

    # Explicit frontmatter on a custom block overrides glob defaults
    assert (
        resolve_context_policy(
            ["custom-world"],
            {"custom-world": {"memory_policy": "isolated"}},
            cfg,
        )
        == "isolated"
    )
    # Explicit frontmatter `full` on a glob-matched block overrides that block's glob
    assert (
        resolve_context_policy(
            ["companion-daily"],
            {"companion-daily": "full"},
            cfg,
        )
        == "full"
    )
    assert most_restrictive_policy("full", "real_facts_only", "full") == "real_facts_only"
    assert most_restrictive_policy("real_facts_only", "isolated") == "isolated"


@pytest.mark.asyncio
async def test_context_dir_frontmatter_stripping_and_horizontal_rule_safety(tmp_path) -> None:
    """Verify `context/*.md` strips valid `key: val` YAML frontmatter without swallowing Markdown `---` (`R2-4`)."""
    staging = tmp_path / "sess_1"
    ctx_dir = staging / "context"
    ctx_dir.mkdir(parents=True)

    # 1. Valid YAML frontmatter in context/*.md -> stripped from injected content, memory_policy exposed
    (ctx_dir / "10-campaign.md").write_text(
        "---\n# comment allowed\nmemory_policy: isolated\nmode: tabletop\n---\n# Chapter 1\nParty enters cavern.\n",
        encoding="utf-8",
    )

    # 2. Markdown horizontal rules `---` with prose lines between them -> NEVER stripped!
    hr_doc = "---\nSection Heading Not Key Value\n---\nBody after horizontal rule."
    (ctx_dir / "20-divider.md").write_text(hr_doc, encoding="utf-8")

    # 3. notes.md starting with `---\nmemory_policy: isolated\n---` -> NEVER stripped (`notes.md` is exempt)
    notes_raw = "---\nmemory_policy: isolated\n---\nKeep this frontmatter literal in notes."
    (staging / "notes.md").write_text(notes_raw, encoding="utf-8")

    entries = await collect_context_entries(str(staging))
    by_name = {e.name: e for e in entries}

    assert by_name["campaign"].memory_policy == "isolated"
    assert "memory_policy" not in by_name["campaign"].content
    assert "# Chapter 1" in by_name["campaign"].content

    assert by_name["divider"].memory_policy is None
    assert "Section Heading Not Key Value" in by_name["divider"].content
    assert "Body after horizontal rule." in by_name["divider"].content

    assert by_name["notes"].memory_policy is None
    assert "memory_policy: isolated" in by_name["notes"].content

    # Unit check on parse_context_frontmatter directly
    fm, body = parse_context_frontmatter("---\n---\nEmpty fences")
    assert fm == {}
    assert "Empty fences" in body


def test_infer_backfill_turn_policies_markers_and_ambiguous_fallback() -> None:
    """Verify backfill marker carry-forward, `until` reset, explicit metadata, and `real_facts_only` fallback."""
    cfg = AgentConfig(
        memory_backfill_context_markers=[
            BackfillContextMarker(
                pattern=r"scenario_ctl\s+start\s+(trpg|stage)",
                policy="isolated",
                until=r"scenario_ctl\s+stop",
            )
        ]
    )

    def _user(mid: str, text: str, ts: float, meta: dict | None = None) -> Message:
        return Message(
            id=mid,
            session_id="s1",
            chatbot_id="discord",
            channel_id="c1",
            sender="u1",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=text,
            timestamp=ts,
            metadata=meta or {},
        )

    def _tool_call(mid: str, cmd: str, ts: float) -> Message:
        return Message(
            id=mid,
            session_id="s1",
            chatbot_id="discord",
            channel_id="c1",
            sender="run_shell_command",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content=cmd,
            timestamp=ts,
            metadata={"tool_name": "run_shell_command", "tool_arguments": {"command": cmd}},
        )

    turns = [
        # Turn 0: normal conversation -> full
        [_user("u0", "普通聊天", 10.0)],
        # Turn 1: starts isolated scenario via marker -> isolated
        [_user("u1", "开启跑团", 20.0), _tool_call("tc1", "scenario_ctl start trpg", 21.0)],
        # Turn 2: carried forward -> isolated
        [_user("u2", "投掷骰子", 30.0)],
        # Turn 3: stops scenario -> full
        [_user("u3", "结束跑团", 40.0), _tool_call("tc3", "scenario_ctl stop", 41.0)],
        # Turn 4: touches context/custom.md without matching any marker -> ambiguous defaults to real_facts_only
        [_user("u4", "写入自定义场景", 50.0), _tool_call("tc4", "tee context/custom.md", 51.0)],
        # Turn 5: explicit persisted metadata wins
        [_user("u5", "带显式策略", 60.0, meta={"context_policy": "isolated"})],
    ]

    policies = infer_backfill_turn_policies(turns, disk_entries=(), cfg=cfg)
    assert policies == [
        "full",
        "isolated",
        "isolated",
        "full",
        "real_facts_only",
        "isolated",
    ]


def test_infer_backfill_nested_stacked_markers() -> None:
    """Verify nested sub-markers (`companion` -> `duel` -> `stop duel#1`) return to outer marker policy."""
    cfg = AgentConfig(
        memory_backfill_context_markers=[
            BackfillContextMarker(
                pattern=r"ctl\s+start\s+companion\b",
                policy="real_facts_only",
                until=r"ctl\s+stop(?:\s+companion(?:#\d+)?\b|(?!\s+[A-Za-z0-9_]))",
            ),
            BackfillContextMarker(
                pattern=r"ctl\s+start\s+duel\b",
                policy="isolated",
                until=r"ctl\s+stop(?:\s+duel(?:#\d+)?\b|(?!\s+[A-Za-z0-9_]))",
            ),
        ]
    )

    def _turn(idx: int, cmd: str | None = None) -> list[Message]:
        msgs = [
            Message(
                id=f"u_{idx}",
                session_id="s_nest",
                chatbot_id="discord",
                channel_id="c1",
                sender="u1",
                role=MessageRole.USER,
                type=MessageType.TEXT,
                content=f"turn {idx}",
                timestamp=float(idx * 10),
            )
        ]
        if cmd:
            msgs.append(
                Message(
                    id=f"tc_{idx}",
                    session_id="s_nest",
                    chatbot_id="discord",
                    channel_id="c1",
                    sender="run_shell_command",
                    role=MessageRole.TOOL,
                    type=MessageType.TOOL_CALL,
                    content=cmd,
                    timestamp=float(idx * 10 + 1),
                    metadata={"tool_name": "run_shell_command", "tool_arguments": {"command": cmd}},
                )
            )
        return msgs

    turns = [
        _turn(1, "ctl start companion"),  # real_facts_only
        _turn(2, "ctl start duel"),  # isolated (nested)
        _turn(3),  # isolated (carried)
        _turn(4, "ctl stop duel#1"),  # returns to real_facts_only!
        _turn(5),  # real_facts_only (outer companion still active)
        _turn(6, "ctl stop"),  # bare stop clears all -> full
    ]
    assert infer_backfill_turn_policies(turns, disk_entries=(), cfg=cfg) == [
        "real_facts_only",
        "isolated",
        "isolated",
        "real_facts_only",
        "real_facts_only",
        "full",
    ]


def test_end_of_session_disk_entries_do_not_retroactively_poison_marker_managed_turns() -> None:
    """Verify end-of-session `disk_entries` never poisons earlier `full`/`real_facts_only` turns when markers match."""
    from kesoku.agent.context_dir import ContextEntry

    cfg = AgentConfig(
        memory_isolated_contexts=["stage-*"],
        memory_filtered_contexts=["companion-*"],
        memory_backfill_context_markers=[
            BackfillContextMarker(
                pattern=r"ctl\s+start\s+companion\b",
                policy="real_facts_only",
                until=r"ctl\s+stop\s+companion\b",
            ),
            BackfillContextMarker(
                pattern=r"ctl\s+start\s+stage\b",
                policy="isolated",
                until=r"ctl\s+stop\s+stage\b",
            ),
        ],
    )

    def _turn(idx: int, cmd: str | None = None) -> list[Message]:
        msgs = [
            Message(
                id=f"u_{idx}",
                session_id="s_agra",
                chatbot_id="discord",
                channel_id="c1",
                sender="u1",
                role=MessageRole.USER,
                type=MessageType.TEXT,
                content=f"turn {idx}",
                timestamp=float(idx * 10),
            )
        ]
        if cmd:
            msgs.append(
                Message(
                    id=f"tc_{idx}",
                    session_id="s_agra",
                    chatbot_id="discord",
                    channel_id="c1",
                    sender="run_shell_command",
                    role=MessageRole.TOOL,
                    type=MessageType.TOOL_CALL,
                    content=cmd,
                    timestamp=float(idx * 10 + 1),
                    metadata={"tool_name": "run_shell_command", "tool_arguments": {"command": cmd}},
                )
            )
        return msgs

    # Leftover isolated context file on disk from Turn 4's late `ctl start stage`
    leftover_disk = [ContextEntry(name="stage-2", content="Stage contest active")]

    # 1. Full session: Turn 1 plain (`full`), Turns 2-3 `companion` (`real_facts_only`), Turn 4 `stage` (`isolated`)
    all_turns = [
        _turn(1),
        _turn(2, "ctl start companion"),
        _turn(3),
        _turn(4, "ctl start stage"),
    ]
    assert infer_backfill_turn_policies(all_turns, disk_entries=leftover_disk, cfg=cfg) == [
        "full",
        "real_facts_only",
        "real_facts_only",
        "isolated",
    ]

    # 2. Multi-window split: Window 1 has plain chat only, while `later_tool_calls` shows marker usage in Window 2
    assert infer_backfill_turn_policies(
        [all_turns[0]],
        disk_entries=leftover_disk,
        cfg=cfg,
        later_tool_calls=[all_turns[3][1]],
    ) == ["full"]

    # 3. Window 2 with `prior_tool_calls` seeding `companion` (`real_facts_only`) despite `leftover_disk`
    assert infer_backfill_turn_policies(
        [all_turns[2]],
        disk_entries=leftover_disk,
        cfg=cfg,
        prior_tool_calls=[all_turns[1][1]],
        later_tool_calls=[all_turns[3][1]],
    ) == ["real_facts_only"]

    # 4. Out-of-band `disk_entries` WITH ZERO marker matches anywhere in the session -> `disk_entries` fallback applies!
    no_marker_turns = [_turn(1), _turn(2)]
    assert infer_backfill_turn_policies(no_marker_turns, disk_entries=leftover_disk, cfg=cfg) == [
        "isolated",
        "isolated",
    ]

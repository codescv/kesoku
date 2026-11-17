"""Unit tests for question `choice_items` enrichment in `kesoku.gateway.chatbot.app.segments`."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from kesoku.constants import MessageRole
from kesoku.db import Message
from kesoku.gateway.chatbot.app import segments as seg_mod
from kesoku.gateway.chatbot.app.segments import (
    load_character_registry,
    match_character_option,
    message_to_record,
    to_wire_segments,
)
from kesoku.gateway.chatbot.base import parse_message_content


def _write_registry(root: Path, folder: str, entries: list[dict[str, Any]], images: list[str] = ()) -> None:
    """Write `characters/<folder>/characters.json` plus empty image files (workspace-relative)."""
    d = root / "characters" / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "characters.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    for rel in images:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"img")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace with a generic character registry (a solo companion folder and a roster folder)."""
    _write_registry(
        tmp_path,
        "a_solo",
        [
            {
                "name": "Hoshino Rin",
                "draftable": False,
                "aliases": ["Rin", "星野凛", "星野 凛"],
                "image": "characters/a_solo/images/rin_solo.jpg",
            }
        ],
        ["characters/a_solo/images/rin_solo.jpg"],
    )
    _write_registry(
        tmp_path,
        "z_roster",
        [
            {"name": "Hoshino Rin", "aliases": ["星野凛"], "image": "characters/z_roster/images/rin.jpg"},
            {"name": "Aoki Mei", "aliases": ["青木芽衣", "Mei"], "image": "characters/z_roster/images/mei.jpg"},
            {"name": "Kudo Sora", "aliases": ["工藤空"], "image": "characters/z_roster/images/sora.jpg"},
            {"name": "Noimage Taro", "aliases": ["无图太郎"], "image": "characters/z_roster/images/missing.jpg"},
            {"name": "Mo", "aliases": ["墨"]},  # one-char CJK alias is ignored
        ],
        ["characters/z_roster/images/rin.jpg", "characters/z_roster/images/mei.jpg",
         "characters/z_roster/images/sora.jpg"],
    )
    _write_registry(
        tmp_path,
        "player",
        [{"name": "Player One", "draftable": False, "aliases": ["玩家一号"], "image": "characters/player/p.jpg"}],
        ["characters/player/p.jpg"],
    )
    seg_mod._registry_cache.clear()
    return tmp_path


def _question(choices: list[str], **extra: Any) -> dict[str, Any]:
    return {"type": "question", "question": "Pick", "choices": choices, "multi": False, "max": 1, **extra}


def _wire(seg: dict[str, Any], ws: Path) -> dict[str, Any]:
    wire, _ = to_wire_segments([seg], message_id="m1", live=False, workspace_dir=str(ws))
    return wire[0]


def test_registry_first_folder_wins_and_draftable_merges(workspace: Path) -> None:
    """Folders load alphabetically: the first folder's avatar wins while draftability is merged per key."""
    keys, refs = load_character_registry(str(workspace))
    rin = refs["星野凛"]
    assert rin.avatar == "characters/a_solo/images/rin_solo.jpg" and rin.draftable is True
    assert refs["hoshino rin"].avatar == "characters/a_solo/images/rin_solo.jpg"
    assert "rin" not in refs and "mei" not in refs  # Latin aliases are not detection keys
    assert "墨" not in refs
    assert refs["无图太郎"].avatar == ""
    assert refs["玩家一号"].draftable is False
    assert keys == sorted(keys, key=len, reverse=True)


def test_registry_cache_reloads_on_mtime_change(workspace: Path) -> None:
    """The registry is cached by file mtimes and rebuilt when a characters.json changes."""
    _, refs1 = load_character_registry(str(workspace))
    _, refs2 = load_character_registry(str(workspace))
    assert refs1 is refs2
    path = workspace / "characters" / "z_roster" / "characters.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.append({"name": "New Face", "aliases": ["新面孔"]})
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    _, refs3 = load_character_registry(str(workspace))
    assert "新面孔" in refs3 and refs3 is not refs1


def test_match_character_option_parsing(workspace: Path) -> None:
    """Leading index/emoji are stripped and `name（subtitle）` / delimiter forms are parsed."""
    keys, refs = load_character_registry(str(workspace))
    m = match_character_option("1. 星野凛（主唱·冷静）", keys, refs, draftable_only=True)
    assert m == {"name": "星野凛", "subtitle": "主唱·冷静", "avatar": "characters/a_solo/images/rin_solo.jpg"}
    assert match_character_option("🌸 青木芽衣 · 吉他手", keys, refs, draftable_only=True)["subtitle"] == "吉他手"
    assert match_character_option("Aoki Mei (guitar)", keys, refs, draftable_only=True)["name"] == "Aoki Mei"
    assert match_character_option("工藤空", keys, refs, draftable_only=True)["name"] == "工藤空"
    # Non-draftable entries are skipped for auto detection but usable for explicit cards.
    assert match_character_option("玩家一号", keys, refs, draftable_only=True) is None
    explicit = match_character_option("玩家一号", keys, refs, draftable_only=False)
    assert explicit is not None and explicit["avatar"] == "characters/player/p.jpg"
    # Action sentences never match: name not at the start, glued text, or a long pre-paren prefix.
    assert match_character_option("跟星野凛去二楼调查", keys, refs, draftable_only=True) is None
    assert match_character_option("星野凛去二楼调查", keys, refs, draftable_only=True) is None
    long_action = "星野凛 跟我一起去二楼仔细调查那个奇怪的房间"
    assert match_character_option(long_action, keys, refs, draftable_only=True) is None


def test_auto_cards_for_character_candidates(workspace: Path) -> None:
    """Mostly-character options render as cards; the reroll option stays as a plain item."""
    choices = ["星野凛（主唱）", "青木芽衣（吉他）", "工藤空（鼓手）", "🎲 再帮我想 8 个不同的候选"]
    item = _wire(_question(choices), workspace)
    assert item["layout"] == "cards"
    assert [c["value"] for c in item["choice_items"]] == item["choices"]
    assert item["choice_items"][0] == {
        "value": "星野凛（主唱）",
        "name": "星野凛",
        "subtitle": "主唱",
        "avatar": "characters/a_solo/images/rin_solo.jpg",
    }
    assert item["choice_items"][3] == {"value": "🎲 再帮我想 8 个不同的候选", "name": "🎲 再帮我想 8 个不同的候选"}


def test_auto_cards_thresholds(workspace: Path) -> None:
    """Cards need >= 2 avatar matches (1 for a single option) covering >= 50% of non-reroll options."""
    single = _wire(_question(["青木芽衣（吉他）"]), workspace)
    assert single["layout"] == "cards"
    one_of_two = _wire(_question(["青木芽衣（吉他）", "随便聊聊"]), workspace)
    assert "choice_items" not in one_of_two
    minority = _wire(_question(["星野凛", "青木芽衣", "看海", "散步", "吃饭"]), workspace)
    assert "choice_items" not in minority
    no_avatar = _wire(_question(["无图太郎", "青木芽衣", "看海"]), workspace)
    assert "choice_items" not in no_avatar
    players = _wire(_question(["玩家一号", "玩家一号（备用）"]), workspace)
    assert "choice_items" not in players


def test_no_cards_for_gameplay_actions(workspace: Path) -> None:
    """Normal gameplay action options that mention characters never become cards."""
    item = _wire(_question(["跟星野凛去二楼调查", "让青木芽衣先走", "独自留在大厅"]), workspace)
    assert "layout" not in item and "choice_items" not in item


def test_explicit_cards_layout_and_avatars(workspace: Path) -> None:
    """Explicit `layout: cards` / explicit avatars always emit cards, filling missing avatars from the registry."""
    content = (
        "```ui\nprompt: Who joins?\nlayout: cards\noptions:\n"
        "  - {name: 玩家一号, subtitle: 自己}\n"
        "  - {name: Custom Hero, avatar: '[file: art/hero.png]'}\n"
        "  - 随便\n```"
    )
    segs = parse_message_content(content, rich_blocks=True)
    q = next(s for s in segs if s["type"] == "question")
    assert q["layout"] == "cards" and len(q["option_items"]) == 3
    item = _wire(q, workspace)
    assert item["layout"] == "cards"
    assert item["choice_items"] == [
        {"value": "玩家一号（自己）", "name": "玩家一号", "subtitle": "自己", "avatar": "characters/player/p.jpg"},
        {"value": "Custom Hero", "name": "Custom Hero", "avatar": "art/hero.png"},
        {"value": "随便", "name": "随便"},
    ]
    # Plain (non-rich) parsing never carries the presentation metadata.
    plain_q = next(s for s in parse_message_content(content) if s["type"] == "question")
    assert "layout" not in plain_q and "option_items" not in plain_q


def test_explicit_list_layout_disables_auto_detection(workspace: Path) -> None:
    """`layout: list` keeps character options as a plain list."""
    item = _wire(_question(["星野凛", "青木芽衣"], layout="list"), workspace)
    assert item["layout"] == "list" and "choice_items" not in item


def test_missing_registry_and_default_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a registry nothing is enriched; `workspace_dir` defaults to the current directory."""
    seg_mod._registry_cache.clear()
    assert "choice_items" not in _wire(_question(["星野凛", "青木芽衣"]), tmp_path)
    monkeypatch.chdir(tmp_path)
    wire, _ = to_wire_segments([_question(["a", "b"])], message_id="m", live=True)
    assert wire[0]["choices"] == ["a", "b"] and "choice_items" not in wire[0]


def test_history_record_enrichment(workspace: Path) -> None:
    """`message_to_record` (session history) enriches question segments like live finals."""
    msg = Message(
        session_id="s",
        chatbot_id="app",
        channel_id="app:s",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        content=(
            "选一位：\n```ui\ntype: select\nprompt: 选谁？\noptions:\n"
            "  - 星野凛（主唱）\n  - 青木芽衣（吉他）\n```"
        ),
    )
    rec = message_to_record(msg, "app", workspace_dir=str(workspace))
    q = next(s for s in rec["segments"] if s["type"] == "question")
    assert q["layout"] == "cards" and q["choice_items"][1]["avatar"] == "characters/z_roster/images/mei.jpg"

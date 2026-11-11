"""Generic, configuration-driven context memory policy resolution for Kesoku.

Classifies active `<context>` blocks and historical backfill turns into one of three policies:
`full`, `real_facts_only`, or `isolated` (most restrictive wins: `isolated > real_facts_only > full`).
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from kesoku.agent.context_dir import ContextEntry
from kesoku.config import AgentConfig, KesokuConfig, get_config
from kesoku.constants import MessageRole, MessageType
from kesoku.db.models import Message

ContextPolicy = Literal["full", "real_facts_only", "isolated"]

POLICY_RANK: dict[str, int] = {
    "full": 0,
    "real_facts_only": 1,
    "isolated": 2,
}


def normalize_policy(value: Any) -> ContextPolicy | None:
    """Normalize a raw policy value into a valid `ContextPolicy`, or None if unrecognized."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if cleaned == "full":
        return "full"
    if cleaned == "real_facts_only":
        return "real_facts_only"
    if cleaned == "isolated":
        return "isolated"
    return None


def most_restrictive_policy(*policies: str | None, default: ContextPolicy = "full") -> ContextPolicy:
    """Return the most restrictive policy among `policies` (`isolated > real_facts_only > full`)."""
    best: ContextPolicy = default
    best_rank = POLICY_RANK[best]
    for raw in policies:
        norm = normalize_policy(raw)
        if norm is not None and POLICY_RANK[norm] > best_rank:
            best = norm
            best_rank = POLICY_RANK[norm]
            if best == "isolated":
                return "isolated"
    return best


def _extract_agent_config(cfg: AgentConfig | KesokuConfig | None) -> AgentConfig:
    """Unwrap `AgentConfig` from `KesokuConfig`, `AgentConfig`, or global default."""
    if isinstance(cfg, AgentConfig):
        return cfg
    if isinstance(cfg, KesokuConfig):
        return cfg.agent
    return get_config().agent


def _extract_frontmatter_policy(entry: Any) -> ContextPolicy | None:
    """Extract `memory_policy` from a string, mapping, or `ContextEntry`."""
    if entry is None:
        return None
    if isinstance(entry, str):
        return normalize_policy(entry)
    if isinstance(entry, ContextEntry):
        return normalize_policy(entry.memory_policy)
    if isinstance(entry, Mapping):
        return normalize_policy(entry.get("memory_policy"))
    return normalize_policy(getattr(entry, "memory_policy", None))


def _matches_any_glob(name: str, patterns: Sequence[str]) -> bool:
    """Check whether `name` matches any glob pattern in `patterns`."""
    for pat in patterns:
        cleaned = pat.strip()
        if cleaned and fnmatch.fnmatchcase(name, cleaned):
            return True
    return False


def resolve_context_policy(
    block_names: Sequence[str],
    frontmatters: Mapping[str, Any] | Sequence[Any] | None = None,
    cfg: AgentConfig | KesokuConfig | None = None,
) -> ContextPolicy:
    """Resolve the effective turn `ContextPolicy` from active context blocks and configuration.

    Resolution order per block:
      1. Explicit `memory_policy` in validated YAML frontmatter of the context file.
      2. `[agent] memory_isolated_contexts` / `memory_filtered_contexts` glob patterns on block name.
      3. `'full'`.
    Across all active blocks, the most restrictive policy (`isolated > real_facts_only > full`) wins.

    Args:
        block_names: Names of active `<context name="...">` blocks in the turn.
        frontmatters: Optional mapping `{block_name: policy_or_frontmatter_dict}` or sequence aligned
            with `block_names`.
        cfg: `AgentConfig` or `KesokuConfig` (defaults to `get_config().agent`).

    Returns:
        Resolved `ContextPolicy` (`'full'`, `'real_facts_only'`, or `'isolated'`).
    """
    agent_cfg = _extract_agent_config(cfg)
    isolated_globs = agent_cfg.memory_isolated_contexts
    filtered_globs = agent_cfg.memory_filtered_contexts

    fm_by_name: dict[str, ContextPolicy] = {}
    extra_policies: list[ContextPolicy] = []

    if isinstance(frontmatters, Mapping):
        for k, v in frontmatters.items():
            pol = _extract_frontmatter_policy(v)
            if pol is not None:
                fm_by_name[str(k)] = pol
    elif isinstance(frontmatters, Sequence) and not isinstance(frontmatters, (str, bytes)):
        for idx, item in enumerate(frontmatters):
            pol = _extract_frontmatter_policy(item)
            if pol is not None:
                if idx < len(block_names):
                    fm_by_name[block_names[idx]] = pol
                else:
                    extra_policies.append(pol)

    resolved: list[ContextPolicy] = list(extra_policies)
    for name in block_names:
        if name in fm_by_name:
            resolved.append(fm_by_name[name])
            continue
        if _matches_any_glob(name, isolated_globs):
            resolved.append("isolated")
        elif _matches_any_glob(name, filtered_globs):
            resolved.append("real_facts_only")
        else:
            resolved.append("full")

    for k, pol in fm_by_name.items():
        if k not in block_names:
            resolved.append(pol)

    return most_restrictive_policy(*resolved, default="full")


def _tool_call_text(msg: Message) -> str:
    """Serialize a TOOL_CALL message's tool name, arguments, and content for marker matching."""
    meta = msg.metadata or {}
    tool_name = str(meta.get("tool_name") or msg.sender or "")
    tool_args = meta.get("tool_arguments") or meta.get("tool_args") or {}
    try:
        args_str = json.dumps(tool_args, ensure_ascii=False) if isinstance(tool_args, (dict, list)) else str(tool_args)
    except Exception:
        args_str = str(tool_args)
    return f"{tool_name} {args_str} {msg.content or ''}"


def _touches_context_ambiguously(text: str) -> bool:
    """Detect whether a tool call touches a staging `context/` file without matching a known marker."""
    return bool(re.search(r"(?:^|[/\s\"'])context/[a-zA-Z0-9_.-]+\.md\b", text))


def infer_backfill_turn_policies(
    turns: Sequence[Sequence[Message]],
    disk_entries: Sequence[ContextEntry] = (),
    cfg: AgentConfig | KesokuConfig | None = None,
) -> list[ContextPolicy]:
    """Infer per-turn `ContextPolicy` across historical turns for `kesoku memory backfill`.

    Combines:
      (a) Persisted `metadata.context_policy` on turn messages when available.
      (b) Tool-call markers in `[agent] memory_backfill_context_markers`, carried forward until
          a matching `until` regex or session end.
      (c) Remaining `context/*.md` files on disk in the session staging directory.
      (d) Ambiguous context-modifying turns default to `'real_facts_only'` (never `'full'`).

    Args:
        turns: Chronological logical turns of the session.
        disk_entries: Context entries currently present on disk in `$STAGING_DIR`.
        cfg: `AgentConfig` or `KesokuConfig`.

    Returns:
        One `ContextPolicy` per input turn, in turn order.
    """
    agent_cfg = _extract_agent_config(cfg)
    disk_policy = resolve_context_policy(
        [e.name for e in disk_entries],
        {e.name: e.memory_policy for e in disk_entries},
        agent_cfg,
    )

    compiled_markers: list[tuple[re.Pattern[str], ContextPolicy, re.Pattern[str] | None]] = []
    for raw_marker in agent_cfg.memory_backfill_context_markers:
        pat = raw_marker["pattern"] if isinstance(raw_marker, Mapping) else raw_marker.pattern
        pol_raw = raw_marker["policy"] if isinstance(raw_marker, Mapping) else raw_marker.policy
        until_raw = (raw_marker.get("until", "") if isinstance(raw_marker, Mapping) else raw_marker.until) or ""
        pol = normalize_policy(pol_raw) or "real_facts_only"
        compiled_markers.append(
            (
                re.compile(pat),
                pol,
                re.compile(until_raw) if until_raw else None,
            )
        )

    active_markers: dict[int, tuple[ContextPolicy, re.Pattern[str] | None]] = {}
    out: list[ContextPolicy] = []

    for turn in turns:
        explicit_policy: ContextPolicy | None = None
        for msg in turn:
            if msg.role == MessageRole.USER and isinstance(msg.metadata, Mapping):
                pol = normalize_policy(msg.metadata.get("context_policy"))
                if pol is not None:
                    explicit_policy = most_restrictive_policy(explicit_policy, pol)

        turn_matched_policies: list[ContextPolicy] = []
        ambiguous = False

        for msg in turn:
            if msg.type != MessageType.TOOL_CALL:
                continue
            tc_text = _tool_call_text(msg)
            matched_any_marker = False
            for m_idx, (_, active_until) in list(active_markers.items()):
                if active_until is not None and active_until.search(tc_text):
                    matched_any_marker = True
                    active_markers.pop(m_idx, None)

            for m_idx, (start_re, marker_pol, until_re) in enumerate(compiled_markers):
                if start_re.search(tc_text):
                    matched_any_marker = True
                    turn_matched_policies.append(marker_pol)
                    active_markers[m_idx] = (marker_pol, until_re)
            if not matched_any_marker and _touches_context_ambiguously(tc_text):
                ambiguous = True

        if explicit_policy is not None:
            out.append(explicit_policy)
            continue

        candidates: list[str | None] = [
            *turn_matched_policies,
            *(pol for pol, _ in active_markers.values()),
        ]
        if disk_policy != "full":
            candidates.append(disk_policy)
        if ambiguous:
            candidates.append("real_facts_only")

        out.append(most_restrictive_policy(*candidates, default="full"))

    return out

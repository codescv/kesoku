"""CLI subcommands for Kesoku Cross-Session Memory (`kesoku memory ...`)."""

from __future__ import annotations

import asyncio
import datetime
import os
import sys
import time
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from kesoku.agent.context_dir import collect_context_entries
from kesoku.agent.context_policy import infer_backfill_turn_policies, normalize_policy
from kesoku.agent.history import segment_logical_turns
from kesoku.agent.memory_consolidator import (
    EpisodeExtract,
    MemoryConsolidator,
    _guarded_malloc_trim,
    consolidate_file_sync,
    consolidate_role_sync,
)
from kesoku.agent.memory_store import (
    default_pinned_cache,
    get_memory_index_version,
    sync_role_memories,
)
from kesoku.agent.session_digest import (
    MAX_LAST_USER_MSG_CHARS,
    MAX_LAST_USER_MSGS,
    MAX_RAW_EXCERPT_CHARS,
    is_excluded_from_memory,
)
from kesoku.agent.tools.memory import resolve_canonical_session_role_sync
from kesoku.config import KesokuConfig, load_config
from kesoku.constants import MessageRole, MessageType
from kesoku.context import KesokuContext
from kesoku.db.manager import DatabaseManager
from kesoku.db.models import Message
from kesoku.logger import setup_logger
from kesoku.utils.text import fts_tokenize, strip_injected_blocks

logger = setup_logger(__name__)
memory_app = typer.Typer(help="Inspect, backfill, consolidate, and reindex cross-session role memories.")


def _load_cfg_and_db(config_path: str) -> tuple[KesokuConfig, DatabaseManager]:
    cfg = load_config(config_path)
    db = DatabaseManager(cfg.workspace.db_path)
    db._config = cfg
    db.verify_db()
    return cfg, db


def _discover_roles(cfg: KesokuConfig, db: DatabaseManager, role_filter: str | None = None) -> list[str]:
    if role_filter:
        return [role_filter]
    roles: set[str] = set()
    roles_dir = cfg.workspace.roles_dir
    if os.path.isdir(roles_dir):
        for name in os.listdir(roles_dir):
            if not name.startswith(".") and os.path.isdir(os.path.join(roles_dir, name)):
                roles.add(name)
    with db.connection_provider.connection() as conn:
        for tbl, col in (("memory_files", "role"), ("session_digests", "role"), ("memory_extraction_cursors", "role")):
            try:
                for row in conn.execute(f"SELECT DISTINCT {col} FROM {tbl}").fetchall():
                    if row[0]:
                        roles.add(str(row[0]))
            except Exception:
                pass
    return sorted(roles) or ["default"]


def _parse_since_date(since_str: str) -> float:
    """Parse `YYYY-MM-DD` or ISO-8601 datetime string into a Unix timestamp."""
    cleaned = since_str.strip()
    try:
        if len(cleaned) == 10 and cleaned[4] == "-" and cleaned[7] == "-":
            dt = datetime.datetime.strptime(cleaned, "%Y-%m-%d")
            return dt.timestamp()
        dt = datetime.datetime.fromisoformat(cleaned)
        return dt.timestamp()
    except ValueError as e:
        raise typer.BadParameter(f"Invalid --since date '{since_str}' (expected YYYY-MM-DD or ISO-8601): {e}") from e


@memory_app.command("status")
def memory_status_cmd(
    role: Annotated[str | None, typer.Option("-r", "--role", help="Filter status to a single role")] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Show memory file sizes, pinned budget usage, FTS counts, pending cursors, failures, and token spend."""
    cfg, db = _load_cfg_and_db(config)
    console = Console()
    roles = _discover_roles(cfg, db, role)

    fts_state = db.get_fts_backfill_state("messages_fts")
    with db.connection_provider.connection() as conn:
        try:
            total_msg_fts = int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
            total_dig_fts = int(conn.execute("SELECT COUNT(*) FROM session_digests_fts").fetchone()[0])
        except Exception:
            total_msg_fts = -1
            total_dig_fts = -1

    table = Table(title="Kesoku Cross-Session Memory Status")
    table.add_column("Role", style="cyan", no_wrap=True)
    table.add_column("Files (Pinned)", justify="right")
    table.add_column("Total Bytes", justify="right")
    table.add_column("Pinned Chars", justify="right")
    table.add_column("Digests (Fresh)", justify="right")
    table.add_column("FTS Rows", justify="right")
    table.add_column("Pending Sess/Turns", justify="right")
    table.add_column("Failures", justify="right")
    table.add_column("Tokens Today / Budget", justify="right")
    table.add_column("Idx Ver", justify="right")

    for r_name in roles:
        mem_dir = os.path.join(cfg.workspace.roles_dir, r_name, "memories")
        sync_role_memories(db, r_name, mem_dir)
        files = db.get_memory_files_by_role(r_name)
        pinned_count = sum(1 for f in files if str(f.get("visibility") or "DEFAULT") == "PINNED")
        total_bytes = sum(int(f.get("size_bytes") or 0) for f in files)
        pinned_block = default_pinned_cache.get_block(
            r_name,
            memories_dir=mem_dir,
            max_chars=cfg.agent.memory_pinned_max_chars,
            pinned_files=cfg.agent.memory_pinned_files,
            db=db,
        )

        with db.connection_provider.connection() as conn:
            d_row = conn.execute(
                "SELECT COUNT(*), MAX(last_active_at) FROM session_digests WHERE role = ?",
                (r_name,),
            ).fetchone()
            dig_count = int(d_row[0] or 0) if d_row else 0
            latest_ts = float(d_row[1] or 0.0) if d_row else 0.0
            try:
                safe_r = r_name.replace('"', '""')
                role_fts = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
                        (f'persona_role:"{safe_r}"',),
                    ).fetchone()[0]
                )
            except Exception:
                role_fts = 0

        fresh_str = (
            datetime.datetime.fromtimestamp(latest_ts).astimezone().strftime("%m-%d %H:%M")
            if latest_ts > 0
            else "-"
        )
        cursors = db.list_extraction_cursors(role=r_name, min_pending_turns=0)
        pending_sess = sum(1 for c in cursors if int(c.get("pending_user_turns") or 0) > 0)
        pending_turns = sum(int(c.get("pending_user_turns") or 0) for c in cursors)
        failed_sess = sum(1 for c in cursors if int(c.get("failures") or 0) > 0)
        tokens_today = db.get_daily_extraction_tokens(r_name)
        budget = cfg.agent.memory_extraction_daily_token_budget

        table.add_row(
            r_name,
            f"{len(files)} ({pinned_count})",
            f"{total_bytes:,}",
            f"{len(pinned_block)}/{cfg.agent.memory_pinned_max_chars}",
            f"{dig_count} ({fresh_str})",
            f"{role_fts:,}",
            f"{pending_sess}/{pending_turns}",
            str(failed_sess) if failed_sess == 0 else f"[red]{failed_sess}[/red]",
            f"{tokens_today:,} / {budget:,}",
            str(get_memory_index_version(r_name)),
        )

    console.print(table)
    fts_done = bool(fts_state and fts_state.get("done"))
    console.print(
        f"Global FTS5: messages_fts={total_msg_fts:,}, session_digests_fts={total_dig_fts:,}, "
        f"backfill_done={fts_done}"
    )


def run_backfill_fts_sync(
    cfg: KesokuConfig,
    db: DatabaseManager,
    *,
    batch_size: int = 500,
) -> tuple[int, int]:
    """Deterministically backfill `session_digests` and `messages_fts` in bounded batches."""
    sessions = sorted(db.list_sessions(), key=lambda s: (s.created_at, s.id))
    digests_written = 0
    fts_rows_inserted = 0
    eff_batch = max(50, int(batch_size))
    pending_fts_batch: list[dict[str, Any]] = []
    last_ts = 0.0
    last_id = ""

    for sess in sessions:
        canonical_role, is_cron = resolve_canonical_session_role_sync(db, cfg, sess)
        if is_cron:
            continue

        # Stream explicit columns only (never selecting `embedding` BLOB)
        with db.connection_provider.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT {db._MSG_EXPLICIT_COLS}
                FROM messages
                WHERE session_id = ?
                  AND type != ?
                ORDER BY timestamp ASC, id ASC
                """,
                (sess.id, MessageType.SESSION_RENAME),
            ).fetchall()

        if not rows:
            continue

        msgs: list[Message] = [db._row_to_message(r) for r in rows]
        if all(is_excluded_from_memory(m) for m in msgs if m.role == MessageRole.USER):
            continue

        turns = segment_logical_turns(msgs)
        if not turns:
            continue

        staging_dir = os.path.join(cfg.workspace.sessions_dir, sess.workspace_name)
        disk_entries = asyncio.run(collect_context_entries(staging_dir, include_policy_only=True))
        inferred_policies = infer_backfill_turn_policies(turns, disk_entries=disk_entries, cfg=cfg)

        user_snippets: list[str] = []
        human_turn_count = 0
        first_msg = msgs[0]
        last_active = float(msgs[-1].timestamp)
        last_policy = "full"

        for turn, inf_pol in zip(turns, inferred_policies, strict=True):
            explicit = next(
                (
                    normalize_policy(m.metadata.get("context_policy"))
                    for m in turn
                    if m.role == MessageRole.USER and isinstance(m.metadata, dict)
                ),
                None,
            )
            turn_pol = explicit if explicit is not None else inf_pol
            last_policy = turn_pol

            for m in turn:
                if is_excluded_from_memory(m) or m.sender == "Notification":
                    continue
                if m.role == MessageRole.USER and m.type == MessageType.TEXT:
                    human_turn_count += 1
                    cleaned_u = " ".join(strip_injected_blocks(m.content or "").split()).strip()
                    if cleaned_u:
                        snip = (
                            cleaned_u[: MAX_LAST_USER_MSG_CHARS - 1].rstrip() + "…"
                            if len(cleaned_u) > MAX_LAST_USER_MSG_CHARS
                            else cleaned_u
                        )
                        if not user_snippets or user_snippets[-1] != snip:
                            user_snippets.append(snip)

                if m.role in (MessageRole.USER, MessageRole.ASSISTANT) and m.type == MessageType.TEXT:
                    if isinstance(m.metadata, dict) and m.metadata.get("prompt_blocked"):
                        continue
                    cleaned = strip_injected_blocks(m.content or "")
                    if not cleaned:
                        continue
                    tok = fts_tokenize(cleaned, for_query=False)
                    if not tok:
                        continue
                    pending_fts_batch.append(
                        {
                            "message_id": m.id,
                            "session_id": sess.id,
                            "persona_role": canonical_role,
                            "context_policy": turn_pol,
                            "speaker_role": str(m.role),
                            "ts": float(m.timestamp),
                            "raw_excerpt": " ".join(cleaned.split())[:MAX_RAW_EXCERPT_CHARS],
                            "text": tok,
                        }
                    )
                    last_ts = float(m.timestamp)
                    last_id = m.id
                    if len(pending_fts_batch) >= eff_batch:
                        fts_rows_inserted += db.fts_insert_messages(pending_fts_batch)
                        db.upsert_fts_backfill_state("messages_fts", last_ts, last_id, done=False)
                        pending_fts_batch.clear()
                        _guarded_malloc_trim()

        if human_turn_count > 0:
            db.upsert_session_digest(
                session_id=sess.id,
                role=canonical_role,
                chatbot_id=first_msg.chatbot_id,
                channel_id=first_msg.channel_id,
                title=sess.title,
                started_at=float(sess.created_at or first_msg.timestamp),
                last_active_at=last_active,
                turn_count=human_turn_count,
                last_user_msgs=user_snippets[-MAX_LAST_USER_MSGS:],
                context_policy=last_policy,
                updated_at=time.time(),
            )
            digests_written += 1

    if pending_fts_batch:
        fts_rows_inserted += db.fts_insert_messages(pending_fts_batch)
        pending_fts_batch.clear()
        _guarded_malloc_trim()

    db.upsert_fts_backfill_state("messages_fts", last_ts, last_id, done=True)
    return digests_written, fts_rows_inserted


@memory_app.command("backfill-fts")
def memory_backfill_fts_cmd(
    batch_size: Annotated[int, typer.Option("--batch-size", help="Rows per FTS5 insert batch")] = 500,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Deterministically backfill `session_digests`, `session_digests_fts`, and `messages_fts`."""
    cfg, db = _load_cfg_and_db(config)
    console = Console()
    t0 = time.perf_counter()
    digests_written, fts_rows = run_backfill_fts_sync(cfg, db, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    console.print(
        f"[bold green]FTS5 & digest backfill complete[/bold green] in {elapsed:.2f}s: "
        f"session_digests upserted={digests_written}, new messages_fts rows={fts_rows}"
    )


async def _run_backfill_async(
    cfg: KesokuConfig,
    db: DatabaseManager,
    *,
    role: str,
    since_ts: float,
    dry_run: bool,
    budget_tokens: int,
    console: Console,
) -> dict[str, int]:
    ctx = KesokuContext(config=cfg, db=db)
    consolidator = MemoryConsolidator(ctx, prewarm_embeddings=False, enable_timers=False)
    stats = {
        "sessions_scanned": 0,
        "windows_extracted": 0,
        "isolated_skipped": 0,
        "ops_total": 0,
        "tokens_spent": 0,
    }
    try:
        # Order oldest -> newest so newer facts supersede older ones cleanly (M1 / §3.5)
        all_sessions = sorted(db.list_sessions(), key=lambda s: (s.created_at, s.id))
        for sess in all_sessions:
            if float(sess.updated_at or sess.created_at or 0.0) < since_ts:
                continue
            canonical_role, is_cron = resolve_canonical_session_role_sync(db, cfg, sess)
            if is_cron or canonical_role != role:
                continue

            stats["sessions_scanned"] += 1

            # Ensure deterministic session_digest base row exists before LLM summary enrichment
            existing_dig = db.get_session_digest(sess.id)
            if existing_dig is None and not dry_run:
                db.upsert_session_digest(
                    session_id=sess.id,
                    role=role,
                    title=sess.title,
                    started_at=sess.created_at,
                    last_active_at=sess.updated_at or sess.created_at,
                )

            while True:
                if budget_tokens > 0 and stats["tokens_spent"] >= budget_tokens:
                    console.print(
                        f"[yellow]Stopped backfill: token budget reached "
                        f"({stats['tokens_spent']:,} >= {budget_tokens:,}).[/yellow]"
                    )
                    return stats

                cur = db.get_extraction_cursor(sess.id)
                before_id = str(cur["last_extracted_msg_id"]) if cur else ""
                outcome = await consolidator.run_episode_extract(
                    EpisodeExtract(
                        role=role,
                        session_id=sess.id,
                        reason="backfill",
                        bypass_budget=True,
                        force=True,
                    ),
                    dry_run=dry_run,
                )
                if outcome.status == "empty_window":
                    break
                if outcome.status == "skipped_isolated":
                    stats["isolated_skipped"] += 1
                    console.print(f"  [dim]#session:{sess.id[:8]} '{sess.title}' -> isolated (0 tokens)[/dim]")
                    if dry_run:
                        break
                    continue
                if outcome.status == "error":
                    console.print(f"  [red]#session:{sess.id[:8]} '{sess.title}' -> extraction error[/red]")
                    break

                tok_used = outcome.input_tokens + outcome.output_tokens
                stats["tokens_spent"] += tok_used
                stats["windows_extracted"] += 1
                ops_list = outcome.ops or []
                stats["ops_total"] += len(ops_list) if dry_run else outcome.ops_applied

                if dry_run:
                    console.print(
                        f"  [cyan]#session:{sess.id[:8]}[/cyan] \"{sess.title}\" "
                        f"(msgs={outcome.messages_processed}, tokens={tok_used:,}) -> "
                        f"summary={outcome.episode_summary!r}, ops={len(ops_list)}"
                    )
                    for op in ops_list:
                        console.print(f"    • {op}")
                    break

                console.print(
                    f"  [green]#session:{sess.id[:8]}[/green] \"{sess.title}\" "
                    f"(msgs={outcome.messages_processed}, ops_applied={outcome.ops_applied}, tokens={tok_used:,})"
                )
                after_cur = db.get_extraction_cursor(sess.id)
                after_id = str(after_cur["last_extracted_msg_id"]) if after_cur else ""
                if not after_id or after_id == before_id or outcome.messages_processed < 200:
                    break
    finally:
        await consolidator.stop(wait_seconds=5.0)
    return stats


@memory_app.command("backfill")
def memory_backfill_cmd(
    role: Annotated[str, typer.Option("-r", "--role", help="Target persona role to backfill")],
    since: Annotated[str, typer.Option("--since", help="Start date (YYYY-MM-DD or ISO-8601)")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview extracted ops without writing files")] = False,
    budget_tokens: Annotated[
        int | None, typer.Option("--budget-tokens", help="Token cap for this backfill run")
    ] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Backfill durable `$MEMORY_DIR` files and episode summaries from historical sessions (oldest -> newest)."""
    cfg, db = _load_cfg_and_db(config)
    since_ts = _parse_since_date(since)
    eff_budget = (
        int(budget_tokens)
        if budget_tokens is not None
        else int(cfg.agent.memory_extraction_daily_token_budget)
    )
    console = Console()
    mode_str = "DRY-RUN" if dry_run else "LIVE"
    console.print(
        f"[bold]Starting {mode_str} memory backfill[/bold] for role='{role}' since={since} "
        f"(budget={eff_budget:,} tokens)..."
    )
    stats = asyncio.run(
        _run_backfill_async(
            cfg,
            db,
            role=role,
            since_ts=since_ts,
            dry_run=dry_run,
            budget_tokens=eff_budget,
            console=console,
        )
    )
    console.print(
        f"[bold green]Backfill ({mode_str}) finished[/bold green]: "
        f"sessions={stats['sessions_scanned']}, windows_extracted={stats['windows_extracted']}, "
        f"isolated_skipped={stats['isolated_skipped']}, ops={stats['ops_total']}, "
        f"tokens_spent={stats['tokens_spent']:,}"
    )


@memory_app.command("consolidate")
def memory_consolidate_cmd(
    role: Annotated[str, typer.Option("-r", "--role", help="Target persona role to consolidate")],
    file: Annotated[
        str | None, typer.Option("-f", "--file", help="Optional single relative file under $MEMORY_DIR")
    ] = None,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Consolidate a role's memory files (deduplicate bullets, demote over-budget PINNED, split >12 KiB)."""
    cfg, db = _load_cfg_and_db(config)
    console = Console()
    mem_dir = os.path.join(cfg.workspace.roles_dir, role, "memories")
    if file:
        touched = consolidate_file_sync(
            db,
            role,
            file,
            mem_dir,
            max_file_bytes=cfg.agent.memory_file_max_bytes,
            pinned_max_chars=cfg.agent.memory_pinned_max_chars,
        )
    else:
        touched = consolidate_role_sync(
            db,
            role,
            mem_dir,
            max_file_bytes=cfg.agent.memory_file_max_bytes,
            pinned_max_chars=cfg.agent.memory_pinned_max_chars,
        )
    console.print(
        f"[bold green]Consolidation complete[/bold green] for role='{role}': "
        f"{len(touched)} file(s) updated {touched if touched else ''}"
    )


@memory_app.command("reindex")
def memory_reindex_cmd(
    role: Annotated[str | None, typer.Option("-r", "--role", help="Role to reindex (omit for all roles)")] = None,
    force: Annotated[bool, typer.Option("--force", help="Force full re-embedding of all files")] = False,
    config: Annotated[str, typer.Option("-c", "--config", help="Path to config.toml")] = "config.toml",
) -> None:
    """Synchronize on-disk `$MEMORY_DIR` Markdown files with SQLite chunk embeddings."""
    cfg, db = _load_cfg_and_db(config)
    console = Console()
    roles = _discover_roles(cfg, db, role)
    total_updated = 0
    for r_name in roles:
        mem_dir = os.path.join(cfg.workspace.roles_dir, r_name, "memories")
        try:
            cnt = sync_role_memories(db, r_name, mem_dir, force=force)
            total_updated += cnt
            console.print(f"  • role [cyan]{r_name}[/cyan]: {cnt} file(s) indexed/updated")
        except Exception as e:
            logger.error(f"Failed reindexing role '{r_name}': {e}")
            sys.exit(1)
    console.print(f"[bold green]Reindex complete[/bold green]: {total_updated} file(s) updated.")

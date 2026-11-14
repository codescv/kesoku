"""`confirm_only` memory proposals: stage extracted facts for explicit user confirmation.

Under the `confirm_only` policy the background extractor never writes `$MEMORY_DIR`. Candidate ops are staged
in `memory_proposals` (status `pending`) instead, and only `ProposalStore.confirm` calls
`MemoryWriter.apply_ops`. Dismissed facts are fingerprinted into `memory_proposal_blocklist` so the same fact
is never proposed again.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from kesoku.agent.memory_writer import MemoryWriter, normalize_rel_memory_path
from kesoku.config import KesokuConfig
from kesoku.db.manager import DatabaseManager
from kesoku.logger import setup_logger

logger = setup_logger(__name__)

PROPOSAL_TTL_SECONDS = 7 * 86400
PROPOSAL_TEXT_MAX_CHARS = 500
PROPOSAL_MAX_OPS = 10
PROPOSAL_SECTION_MAX_CHARS = 200
PROPOSAL_MAX_TAGS = 8
PROPOSABLE_OP_KINDS = frozenset({"upsert_bullet"})

ProposalListener = Callable[[dict[str, Any]], Awaitable[None]]


class ProposalError(Exception):
    """Base error for proposal resolution failures."""

    code = "BAD_REQUEST"


class ProposalNotFoundError(ProposalError):
    """Raised when a proposal id does not exist."""

    code = "NOT_FOUND"


class ProposalNotPendingError(ProposalError):
    """Raised when resolving a proposal that is no longer pending."""

    code = "PROPOSAL_NOT_PENDING"


class ProposalValidationError(ProposalError):
    """Raised when edited ops are invalid."""

    code = "BAD_REQUEST"


def resolve_role_memories_dir(cfg: KesokuConfig, role: str) -> str:
    """Return the absolute `$MEMORY_DIR` (`<roles_dir>/<role>/memories`) for `role`."""
    roles_dir = cfg.workspace.roles_dir
    if not os.path.isabs(roles_dir) and cfg.agent_working_dir:
        roles_dir = os.path.join(cfg.agent_working_dir, roles_dir)
    return os.path.abspath(os.path.join(roles_dir, role or "default", "memories"))


def _norm_text(value: Any) -> str:
    """Collapse whitespace and lowercase a fingerprint component."""
    return " ".join(str(value or "").split()).lower()


def op_fingerprint(op: Mapping[str, Any]) -> str:
    """Return `sha256(normalize(file + section + text))` for one memory op."""
    raw = "\n".join((_norm_text(op.get("file")), _norm_text(op.get("section")), _norm_text(op.get("text"))))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def proposal_fingerprint(ops: Sequence[Mapping[str, Any]]) -> str:
    """Return a stable fingerprint over all op fingerprints of a proposal."""
    joined = "|".join(sorted(op_fingerprint(o) for o in ops))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def sanitize_proposal_op(op: Mapping[str, Any], memories_dir: str) -> dict[str, Any] | None:
    """Validate one candidate op and project it to the proposal shape, or return None when unsafe.

    Only `upsert_bullet` ops targeting a normalized `.md` path inside `memories_dir` are kept; text is
    limited to 500 characters.
    """
    kind = str(op.get("op") or "upsert_bullet").strip().lower()
    if kind not in PROPOSABLE_OP_KINDS:
        return None
    rel = normalize_rel_memory_path(str(op.get("file") or ""), memories_dir)
    if not rel:
        return None
    text = " ".join(str(op.get("text") or "").split()).strip()
    if not text:
        return None
    text = text[:PROPOSAL_TEXT_MAX_CHARS]
    section = " ".join(str(op.get("section") or "").split()).strip()[:PROPOSAL_SECTION_MAX_CHARS]
    raw_tags = op.get("tags") or []
    tags = [str(t).strip()[:40] for t in raw_tags if str(t).strip()][:PROPOSAL_MAX_TAGS] if isinstance(
        raw_tags, (list, tuple)
    ) else []
    out: dict[str, Any] = {"op": "upsert_bullet", "file": rel, "section": section, "text": text, "tags": tags}
    if bool(op.get("pinned")):
        out["pinned"] = True
    replaces = op.get("replaces")
    if isinstance(replaces, str) and replaces.strip():
        out["replaces"] = replaces.strip()[:PROPOSAL_TEXT_MAX_CHARS]
    return out


class ProposalStore:
    """Stages, lists, confirms, dismisses and expires `confirm_only` memory proposals."""

    def __init__(
        self,
        db: DatabaseManager,
        cfg: KesokuConfig,
        listeners: list[ProposalListener] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the store.

        Args:
            db: Synchronous database manager (methods run on worker threads).
            cfg: Kesoku configuration (roles dir, pinned files, writer limits).
            listeners: Async callbacks notified with each newly staged proposal.
            clock: Injectable time source.
        """
        self.db = db
        self.cfg = cfg
        self.listeners: list[ProposalListener] = listeners if listeners is not None else []
        self._clock = clock

    # ------------------------------------------------------------------ staging
    def stage_sync(
        self,
        *,
        session_id: str,
        role: str,
        ops: Sequence[Mapping[str, Any]],
        message_id: str | None = None,
        source_mode: str | None = None,
    ) -> dict[str, Any] | None:
        """Validate, de-duplicate and persist candidate ops as one pending proposal (no `$MEMORY_DIR` writes).

        Returns:
            The stored proposal dict, or None when nothing remains after filtering.
        """
        safe_role = role or "default"
        mem_dir = resolve_role_memories_dir(self.cfg, safe_role)
        cleaned: list[dict[str, Any]] = []
        seen: set[str] = set()
        for op in ops:
            if not isinstance(op, Mapping):
                continue
            proj = sanitize_proposal_op(op, mem_dir)
            if proj is None:
                continue
            fp = op_fingerprint(proj)
            if fp in seen:
                continue
            seen.add(fp)
            cleaned.append(proj)
        if not cleaned:
            return None

        blocked = self.db.get_blocklisted_fingerprints(seen)
        pending_fps: set[str] = set()
        for existing in self.db.list_memory_proposals(status="pending", limit=500):
            if existing.get("role") != safe_role:
                continue
            for eop in existing.get("ops") or []:
                if isinstance(eop, Mapping):
                    pending_fps.add(op_fingerprint(eop))
        remaining = [o for o in cleaned if op_fingerprint(o) not in blocked and op_fingerprint(o) not in pending_fps]
        remaining = remaining[:PROPOSAL_MAX_OPS]
        if not remaining:
            return None

        now = float(self._clock())
        proposal_id = f"mp_{uuid.uuid4().hex[:16]}"
        self.db.insert_memory_proposal(
            proposal_id=proposal_id,
            session_id=session_id,
            message_id=message_id,
            role=safe_role,
            source_mode=source_mode,
            ops=remaining,
            fingerprint=proposal_fingerprint(remaining),
            created_at=now,
            expires_at=now + PROPOSAL_TTL_SECONDS,
        )
        return self.db.get_memory_proposal(proposal_id)

    async def notify(self, proposal: dict[str, Any]) -> None:
        """Notify every registered listener of a newly staged proposal (errors are logged, never raised)."""
        for listener in list(self.listeners):
            try:
                await listener(dict(proposal))
            except Exception as e:
                logger.warning(f"Memory proposal listener failed for {proposal.get('proposal_id')}: {e}")

    async def stage(self, **kwargs: Any) -> dict[str, Any] | None:
        """Async wrapper for `stage_sync` that also notifies listeners."""
        proposal = await asyncio.to_thread(lambda: self.stage_sync(**kwargs))
        if proposal is not None:
            await self.notify(proposal)
        return proposal

    # ------------------------------------------------------------------ resolution
    def _validate_edited_ops(
        self, original: Sequence[Mapping[str, Any]], edited: Sequence[Any], mem_dir: str
    ) -> list[dict[str, Any]]:
        """Validate user-edited ops: a non-empty subset (≤ original count) targeting only original files."""
        if not isinstance(edited, (list, tuple)) or not edited:
            raise ProposalValidationError("edited_ops must be a non-empty list")
        if len(edited) > len(original):
            raise ProposalValidationError("edited_ops may not add operations")
        allowed_files = {str(o.get("file")) for o in original if isinstance(o, Mapping)}
        out: list[dict[str, Any]] = []
        for raw in edited:
            if not isinstance(raw, Mapping):
                raise ProposalValidationError("edited op must be an object")
            if len(str(raw.get("text") or "")) > PROPOSAL_TEXT_MAX_CHARS:
                raise ProposalValidationError("edited op text exceeds 500 characters")
            proj = sanitize_proposal_op({**raw, "op": "upsert_bullet"}, mem_dir)
            if proj is None or proj["file"] not in allowed_files:
                raise ProposalValidationError("edited op targets an invalid or foreign memory file")
            out.append(proj)
        return out

    def confirm_sync(
        self,
        proposal_id: str,
        edited_ops: Sequence[Any] | None = None,
        device_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply a pending proposal through `MemoryWriter.apply_ops` (the ONLY write path for `confirm_only`).

        Writes are confined to the proposal role's `$MEMORY_DIR`; `apply_ops` itself takes the memory lock.

        Returns:
            `{proposal_id, status:"applied", applied_files, ops_applied}`.
        """
        proposal = self.db.get_memory_proposal(proposal_id)
        if proposal is None:
            raise ProposalNotFoundError(f"Proposal '{proposal_id}' not found")
        if proposal["status"] != "pending":
            raise ProposalNotPendingError(f"Proposal '{proposal_id}' is {proposal['status']}")
        role = proposal["role"] or "default"
        mem_dir = resolve_role_memories_dir(self.cfg, role)
        ops = (
            self._validate_edited_ops(proposal["ops"], edited_ops, mem_dir)
            if edited_ops is not None
            else [o for o in (sanitize_proposal_op(op, mem_dir) for op in proposal["ops"]) if o is not None]
        )
        if not self.db.claim_memory_proposal(proposal_id):
            raise ProposalNotPendingError(f"Proposal '{proposal_id}' is no longer pending")
        try:
            agent_cfg = self.cfg.agent
            result = MemoryWriter.apply_ops(
                ops,
                role=role,
                memories_dir=mem_dir,
                db=self.db,
                session_id=proposal["session_id"],
                pinned_files=agent_cfg.memory_pinned_files,
                rejected_tags=agent_cfg.memory_rejected_tags,
                max_file_bytes=agent_cfg.memory_file_max_bytes,
            )
        except Exception:
            self.db.finish_memory_proposal(proposal_id, "pending")
            raise
        self.db.finish_memory_proposal(
            proposal_id, "applied", ts=float(self._clock()), device_id=device_id, applied_ops=ops
        )
        return {
            "proposal_id": proposal_id,
            "status": "applied",
            "applied_files": list(result.touched_files),
            "ops_applied": int(result.ops_applied),
            "session_id": proposal["session_id"],
            "role": role,
        }

    def dismiss_sync(self, proposal_id: str, device_id: str | None = None) -> dict[str, Any]:
        """Dismiss a pending proposal and blocklist the fingerprints of all its facts."""
        proposal = self.db.get_memory_proposal(proposal_id)
        if proposal is None:
            raise ProposalNotFoundError(f"Proposal '{proposal_id}' not found")
        if not self.db.resolve_memory_proposal(proposal_id, "dismissed", ts=float(self._clock()), device_id=device_id):
            raise ProposalNotPendingError(f"Proposal '{proposal_id}' is no longer pending")
        fps = [op_fingerprint(o) for o in proposal["ops"] if isinstance(o, Mapping)]
        self.db.add_memory_proposal_blocklist(fps, proposal["role"] or "default", ts=float(self._clock()))
        return {
            "proposal_id": proposal_id,
            "status": "dismissed",
            "session_id": proposal["session_id"],
            "role": proposal["role"],
        }

    def expire_sync(self) -> list[dict[str, Any]]:
        """Expire overdue pending proposals; returns the expired proposal dicts."""
        ids = self.db.expire_memory_proposals(now=float(self._clock()))
        out: list[dict[str, Any]] = []
        for pid in ids:
            p = self.db.get_memory_proposal(pid)
            if p is not None:
                out.append(p)
        return out

    def list_sync(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List proposals (newest first)."""
        return self.db.list_memory_proposals(status=status, limit=limit)

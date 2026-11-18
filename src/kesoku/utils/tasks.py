"""Helpers for fire-and-forget asyncio background tasks.

`asyncio.create_task` only keeps a *weak* reference to the task in the event loop, so a task whose result is
never stored can be garbage-collected mid-flight, and an exception it raises is only reported (if at all) as
"Task exception was never retrieved" at GC time. `spawn_background` keeps a strong reference until the task
finishes and logs any failure immediately.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)

# Strong references to in-flight background tasks (discarded by the done-callback).
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _on_task_done(task: asyncio.Task[Any], registry: set[asyncio.Task[Any]]) -> None:
    """Drop the strong reference and log a failure (cancellation is not an error)."""
    registry.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            f"Background task {task.get_name()!r} failed: {type(exc).__name__}: {exc}",
            exc_info=(type(exc), exc, exc.__traceback__),
        )


def track_task[T](task: asyncio.Task[T], registry: set[asyncio.Task[Any]] | None = None) -> asyncio.Task[T]:
    """Keep a strong reference to an existing task and log its failure when it completes.

    Args:
        task: Task to track.
        registry: Optional owner-specific set (e.g. so an owner can cancel its own tasks on shutdown);
            defaults to the module-level registry.

    Returns:
        The same task, for chaining.
    """
    reg = _BACKGROUND_TASKS if registry is None else registry
    reg.add(task)
    task.add_done_callback(lambda t: _on_task_done(t, reg))
    return task


def spawn_background[T](
    coro: Coroutine[Any, Any, T],
    *,
    name: str | None = None,
    registry: set[asyncio.Task[Any]] | None = None,
) -> asyncio.Task[T]:
    """Schedule `coro` as a tracked fire-and-forget task on the running loop.

    Args:
        coro: Coroutine to run.
        name: Optional task name (shown in logs on failure).
        registry: Optional owner-specific set holding the strong reference; defaults to a module-level set.

    Returns:
        The created task (callers may still await or cancel it).
    """
    return track_task(asyncio.create_task(coro, name=name), registry)


def background_task_count() -> int:
    """Number of in-flight tasks tracked in the module-level registry (for tests/diagnostics)."""
    return len(_BACKGROUND_TASKS)

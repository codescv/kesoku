"""Timestamped event log written by the scheduler.

Event kinds emitted by :class:`dagrun.scheduler.Scheduler`:

``run_start`` / ``run_end``
    Once per run (node name ``"*"``).
``attempt``
    The node's function is about to be called (``attempt=<n>``).
``retry``
    An attempt failed and will be retried after ``delay`` seconds.
``success`` / ``failed``
    Terminal outcome of a node that ran.
``skipped``
    The node never ran because an upstream node did not succeed (``cause=<node>``).
``cancelled``
    The run was cancelled before the node finished.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    """One entry of the event log.

    Attributes:
        seq: Monotonically increasing sequence number (0-based).
        ts: Clock reading when the event was emitted.
        kind: Event kind (see module docstring).
        node: Node name, or ``"*"`` for run-level events.
        data: Extra key/value details.
    """

    seq: int
    ts: float
    kind: str
    node: str
    data: dict[str, Any] = field(default_factory=dict)


Subscriber = Callable[[Event], None]


class EventLog:
    """Append-only, ordered list of :class:`Event` objects."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """Create an empty log.

        Args:
            clock: Time source used for event timestamps.
        """
        self._clock = clock
        self._events: list[Event] = []
        self._subscribers: list[Subscriber] = []

    def emit(self, kind: str, node: str, **data: Any) -> Event:
        """Append an event and notify subscribers.

        Args:
            kind: Event kind.
            node: Node name (``"*"`` for run-level events).
            **data: Extra details stored on the event.

        Returns:
            The appended event.
        """
        event = Event(seq=len(self._events), ts=self._clock(), kind=kind, node=node, data=dict(data))
        self._events.append(event)
        for sub in list(self._subscribers):
            sub(event)
        return event

    def subscribe(self, callback: Subscriber) -> None:
        """Call ``callback(event)`` for every future event."""
        self._subscribers.append(callback)

    def __iter__(self) -> Iterator[Event]:
        """Iterate over events in emission order."""
        return iter(list(self._events))

    def __len__(self) -> int:
        """Return the number of events."""
        return len(self._events)

    def for_node(self, node: str) -> list[Event]:
        """Return the events of one node, in order."""
        return [e for e in self._events if e.node == node]

    def kinds(self, node: str) -> list[str]:
        """Return the event kinds of one node, in order."""
        return [e.kind for e in self.for_node(node)]

    def count(self, kind: str, node: str | None = None) -> int:
        """Count events of ``kind`` (optionally restricted to ``node``)."""
        return sum(1 for e in self._events if e.kind == kind and (node is None or e.node == node))

    def first(self, kind: str, node: str | None = None) -> Event | None:
        """Return the first event of ``kind`` (optionally for ``node``), or None."""
        for e in self._events:
            if e.kind == kind and (node is None or e.node == node):
                return e
        return None

    def format(self, start: float | None = None) -> str:
        """Render the log as aligned text lines.

        Args:
            start: Timestamp used as time zero (defaults to the first event).
        """
        return "\n".join(format_event(e, start if start is not None else self._origin()) for e in self._events)

    def _origin(self) -> float:
        """Return the timestamp of the first event (0.0 for an empty log)."""
        return self._events[0].ts if self._events else 0.0


def format_event(event: Event, origin: float = 0.0) -> str:
    """Render one event as ``+0.123s  kind  node  k=v ...``."""
    details = " ".join(f"{k}={v}" for k, v in event.data.items())
    return f"+{event.ts - origin:7.3f}s  {event.kind:<9} {event.node:<10} {details}".rstrip()

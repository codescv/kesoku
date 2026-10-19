"""dagrun: run a DAG of async tasks with bounded concurrency, retries and an event log."""

from .errors import CycleError, DagError, DuplicateNodeError, SchedulerStateError, UnknownNodeError, UpstreamFailed
from .events import Event, EventLog
from .graph import DAG, Node
from .retry import NO_RETRY, RetryPolicy
from .scheduler import NodeResult, NodeState, RunResult, Scheduler

__all__ = [
    "DAG",
    "NO_RETRY",
    "CycleError",
    "DagError",
    "DuplicateNodeError",
    "Event",
    "EventLog",
    "Node",
    "NodeResult",
    "NodeState",
    "RetryPolicy",
    "RunResult",
    "Scheduler",
    "SchedulerStateError",
    "UnknownNodeError",
    "UpstreamFailed",
]

__version__ = "0.4.1"

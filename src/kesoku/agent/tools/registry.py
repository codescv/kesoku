"""Base tool registry and execution context structures for Kesoku AI Agent."""

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolContext(BaseModel):
    """Contextual session metadata injected into executing tools."""

    model_config = {"arbitrary_types_allowed": True}

    session_id: str = Field(..., description="Unique session identifier")
    session_workspace: str = Field(..., description="Relative folder name for the session workspace")
    original_msg_id: str | None = Field(None, description="ID of the message initiating the turn")
    active_jobs: Any = Field(None, exclude=True)
    transitioned_to_session: str | None = Field(None, description="New session ID if history was compacted")
    gateway: Any = Field(None, exclude=True)

    @property
    def lcm_engine(self) -> Any:
        """Dynamically retrieve the session-bound LCMEngine instance."""
        return self.gateway.context.get_lcm_engine(self.session_id)


def _create_schema_func(func: Callable) -> Callable:
    """Create a wrapper function with context parameter removed from its signature for LLM schema generation."""
    sig = inspect.signature(func)
    if "context" not in sig.parameters:
        return func
    new_params = [p for p in sig.parameters.values() if p.name != "context"]
    new_sig = sig.replace(parameters=new_params)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore
    return wrapper


class ToolRegistry:
    """Maintains a registry of callable Python functions exposed as LLM tools."""

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, Callable] = {}
        self._schema_tools: dict[str, Callable] = {}

    def register(self, func: Callable) -> Callable:
        """Register a function as a tool. Can be used as a decorator.

        Args:
            func: The Python function to register. Must have type hints and docstrings.

        Returns:
            The registered function unchanged.
        """
        self._tools[func.__name__] = func
        self._schema_tools[func.__name__] = _create_schema_func(func)
        logger.info(f"Registered tool: {func.__name__}")
        return func

    def get_tools_list(self) -> list[Callable]:
        """Retrieve the list of registered tool callables formatted for LLM schema generation.

        Returns:
            A list of callable functions with context arguments stripped.
        """
        return list(self._schema_tools.values())

    def get_tool(self, name: str) -> Callable:
        """Retrieve a specific tool function by name for execution.

        Args:
            name: Name of the tool function.

        Returns:
            The callable function.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]


# Default global registry instance
default_registry = ToolRegistry()

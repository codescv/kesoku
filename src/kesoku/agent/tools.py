"""Tool registry and MVP skills for Kesoku AI Agent."""

import ast
import operator
import time
from collections.abc import Callable

from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class ToolRegistry:
    """Maintains a registry of callable Python functions exposed as LLM tools."""

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, Callable] = {}

    def register(self, func: Callable) -> Callable:
        """Register a function as a tool. Can be used as a decorator.

        Args:
            func: The Python function to register. Must have type hints and docstrings.

        Returns:
            The registered function unchanged.
        """
        self._tools[func.__name__] = func
        logger.info(f"Registered tool: {func.__name__}")
        return func

    def get_tools_list(self) -> list[Callable]:
        """Retrieve the list of registered tool callables.

        Returns:
            A list of callable functions.
        """
        return list(self._tools.values())

    def get_tool(self, name: str) -> Callable:
        """Retrieve a specific tool function by name.

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


@default_registry.register
def calculator(expression: str) -> str:
    """Perform simple mathematical calculations on numerical expressions.

    Supports addition (+), subtraction (-), multiplication (*), and division (/).

    Args:
        expression: A simple mathematical string expression (e.g., '25 * 4 + 10').

    Returns:
        The calculated numerical result as a string.
    """
    # Safe arithmetic evaluation using AST
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op = type(node.op)
            if op in operators:
                return operators[op](left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        raise ValueError(f"Unsupported mathematical construct in expression: {expression}")

    try:
        parsed = ast.parse(expression, mode="eval").body
        result = _eval(parsed)
        return f"Result of {expression} = {result}"
    except Exception as e:
        logger.error(f"Calculator tool failed to evaluate '{expression}': {e}")
        return f"Error evaluating expression '{expression}': {e}"


@default_registry.register
def get_current_time() -> str:
    """Retrieve the current local time and date formatted as a readable string.

    Returns:
        The current local time string.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())


@default_registry.register
def web_search(query: str) -> str:
    """Search the web for current information on a given topic.

    Args:
        query: The search query string.

    Returns:
        Simulated search results summary.
    """
    logger.info(f"Executing simulated web search for query: '{query}'")
    return f"Search results for '{query}': Current information indicates standard operations."

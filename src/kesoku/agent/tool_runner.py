"""Handles safe parameter validation and asynchronous execution of tools."""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from kesoku.agent.tools import ToolContext, ToolRegistry
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class ToolRunner:
    """Handles safe parameter validation and asynchronous execution of tools."""

    def __init__(self, tool_registry: ToolRegistry, tool_context: ToolContext) -> None:
        """Initialize ToolRunner.

        Args:
            tool_registry: Registry of available tools.
            tool_context: Execution context passed to tools.
        """
        self.tool_registry = tool_registry
        self.tool_context = tool_context

    async def execute_tool(
        self,
        call: Any,
        tc_msg: Message,
        is_interrupted: Callable[[], bool] | None = None,
    ) -> Message:
        """Execute a single tool call asynchronously and return a ToolResult Message.

        Args:
            call: ToolCallRequest representing the request.
            tc_msg: Message corresponding to the ToolCall.
            is_interrupted: Optional callback returning True if execution is interrupted.

        Returns:
            A Message representing either the tool execution result or error.
        """
        # is_interrupted is a Callable (e.g., lambda) to support late binding/dynamic evaluation.
        # When multiple tool coroutines are gathered via asyncio.gather, one tool might yield control via
        # await. If a new user message arrives during this time, other concurrent tool tasks must
        # dynamically evaluate the queue state at their execution time to detect and abort in real time.
        if is_interrupted and is_interrupted():
            logger.info(f"Interruption prior to tool '{call.name}'. Aborting tool execution.")
            return Message(
                session_id=tc_msg.session_id,
                chatbot_id=tc_msg.chatbot_id,
                channel_id=tc_msg.channel_id,
                sender=call.name,
                role=MessageRole.TOOL,
                type=MessageType.TOOL_RESULT,
                content=f"Tool `{call.name}` execution was aborted due to thought interruption.",
                status=MessageStatus.RESPONDED,
                parent_id=tc_msg.id,
                metadata={
                    "tool_name": call.name,
                    "tool_error": "Tool execution was aborted due to thought interruption.",
                },
            )

        try:
            tool_func = self.tool_registry.get_tool(call.name)
            call_kwargs = dict(call.arguments)
            sig = inspect.signature(tool_func)

            # Validate required parameters to handle LLM truncation gracefully
            missing_args = []
            for param in sig.parameters.values():
                if param.name == "context":
                    continue
                if param.default is inspect.Parameter.empty and param.name not in call_kwargs:
                    missing_args.append(param.name)
            if missing_args:
                raise ValueError(
                    "Command too long! Split your command into smaller chunks!\n"
                    "If you are writing a file, write at most 4000 characters per command!\n"
                    "Note: only emit 1 tool call in your response because it's too long!"
                )

            if "context" in sig.parameters:
                call_kwargs["context"] = self.tool_context

            # Atomic tool execution
            if inspect.iscoroutinefunction(tool_func):
                # a function defined with async
                result = await tool_func(**call_kwargs)
            else:
                # a normal function
                result = await asyncio.to_thread(tool_func, **call_kwargs)
            return Message(
                session_id=tc_msg.session_id,
                chatbot_id=tc_msg.chatbot_id,
                channel_id=tc_msg.channel_id,
                sender=call.name,
                role=MessageRole.TOOL,
                type=MessageType.TOOL_RESULT,
                content=f"Tool `{call.name}` returned:\n```\n{result}\n```",
                status=MessageStatus.RESPONDED,
                parent_id=tc_msg.id,
                metadata={"tool_name": call.name, "tool_result": str(result)},
            )
        except Exception as te:
            logger.error(f"Error executing tool '{call.name}': {te}")
            return Message(
                session_id=tc_msg.session_id,
                chatbot_id=tc_msg.chatbot_id,
                channel_id=tc_msg.channel_id,
                sender=call.name,
                role=MessageRole.TOOL,
                type=MessageType.TOOL_RESULT,
                content=f"Tool `{call.name}` error:\n```\n{te}\n```",
                status=MessageStatus.RESPONDED,
                parent_id=tc_msg.id,
                metadata={"tool_name": call.name, "tool_error": str(te)},
            )

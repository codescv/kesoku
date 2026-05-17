"""LLM backend interface and implementations for Kesoku AI Agent.

Provides an abstract base class BaseLLM and a concrete GeminiLLM implementation.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from kesoku.config import GeminiConfig
from kesoku.db import Message
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class ToolCallRequest(BaseModel):
    """Represents a requested tool execution from the LLM."""

    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""

    content: str
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)


class BaseLLM(ABC):
    """Abstract base class defining the interface for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[Message] | None = None,
        tools: list[Callable] | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM given prompt and context.

        Args:
            prompt: The latest user prompt string.
            system_prompt: Optional system instructions.
            history: Optional list of prior messages in the session.
            tools: Optional list of callable Python tool functions.

        Returns:
            An LLMResponse containing text and/or requested tool calls.
        """
        pass


class GeminiLLM(BaseLLM):
    """Google GenAI (Gemini) implementation of BaseLLM supporting API Key and Vertex AI."""

    def __init__(self, config: GeminiConfig | None = None) -> None:
        """Initialize the Gemini LLM client.

        Args:
            config: Configuration structure for Gemini backend.
        """
        if config is None:
            from kesoku.config import get_config

            config = get_config().gemini
        self.config = config
        self.model_name = config.model_name
        logger.info(f"Using GeminiLLM backend ({self.model_name}).")

        try:
            from google import genai

            if config.auth_mode == "vertex":
                logger.info(
                    f"Initializing Gemini client in Vertex AI mode "
                    f"(Project: {config.project_id}, Region: {config.location})"
                )
                self.client = genai.Client(
                    vertexai=True,
                    project=config.project_id,
                    location=config.location,
                )
            else:
                key = config.api_key or os.getenv("GEMINI_API_KEY")
                if not key:
                    logger.warning("GEMINI_API_KEY is not set. GeminiLLM calls may fail if not authenticated.")
                self.client = genai.Client(api_key=key)
        except ImportError:
            logger.error("google-genai package is not installed.")
            raise

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[Message] | None = None,
        tools: list[Callable] | None = None,
    ) -> LLMResponse:
        """Generate content using Google GenAI client."""
        import asyncio

        from google.genai import types

        contents = []
        if history:
            for msg in history:
                role = "model" if msg.role == "assistant" else "user"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

        # Append the current prompt
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

        config = types.GenerateContentConfig()
        if system_prompt:
            config.system_instruction = system_prompt
        if tools:
            config.tools = tools  # type: ignore
            config.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)

        def _call() -> Any:
            return self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

        try:
            res = await asyncio.to_thread(_call)
            text_content = res.text if res.text else ""
            tool_calls = []

            if res.function_calls:
                for call in res.function_calls:
                    args_dict = dict(call.args) if call.args else {}
                    tool_calls.append(ToolCallRequest(name=call.name, arguments=args_dict))

            return LLMResponse(content=text_content, tool_calls=tool_calls)
        except Exception as e:
            logger.error(f"GeminiLLM generation failed: {e}")
            raise


class MockLLM(BaseLLM):
    """Mock LLM implementation for unit testing and local verification."""

    def __init__(
        self, mock_response: str = "Hello! I am Kesoku Agent.", mock_tools: list[ToolCallRequest] | None = None
    ) -> None:
        """Initialize MockLLM with canned responses."""
        self.mock_response = mock_response
        self.mock_tools = mock_tools or []

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[Message] | None = None,
        tools: list[Callable] | None = None,
    ) -> LLMResponse:
        """Return canned mock response."""
        logger.debug(f"MockLLM received prompt: {prompt}")
        if "tool execution results" in prompt.lower():
            return LLMResponse(content="The calculation result is 35.", tool_calls=[])
        # If user asks to calculate something, mock tool call
        if "calculate" in prompt.lower() or "+" in prompt:
            return LLMResponse(
                content="Let me calculate that.",
                tool_calls=[ToolCallRequest(name="calculator", arguments={"expression": "25 + 10"})],
            )
        return LLMResponse(content=self.mock_response, tool_calls=self.mock_tools)

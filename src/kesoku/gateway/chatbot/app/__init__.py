"""App channel adapter: an outbound relay link serving native / web app clients."""

from typing import Any

__all__ = ["AppChatbot"]


def __getattr__(name: str) -> Any:
    """Lazily import `AppChatbot` so lightweight submodules import without the full adapter."""
    if name == "AppChatbot":
        from kesoku.gateway.chatbot.app.adapter import AppChatbot

        return AppChatbot
    raise AttributeError(name)

"""Unit tests for Kesoku CLI chat runner module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from rich.console import Console

from kesoku.cli_chat import _list_chat_sessions, _show_session_history, run_cli_chat_async
from kesoku.db import Session


@pytest.mark.asyncio
async def test_list_chat_sessions_empty() -> None:
    """Test listing chat sessions when none exist."""
    gw = MagicMock()
    gw.list_sessions = AsyncMock(return_value=[])
    console = MagicMock(spec=Console)

    await _list_chat_sessions(gw, console)
    console.print.assert_called_once_with("[yellow]No chat sessions found.[/yellow]")


@pytest.mark.asyncio
async def test_list_chat_sessions_with_data() -> None:
    """Test listing chat sessions with existing sessions."""
    gw = MagicMock()
    gw.list_sessions = AsyncMock(
        return_value=[
            Session(id="s1", title="Test Session", created_at=1000.0, updated_at=1000.0)
        ]
    )
    console = MagicMock(spec=Console)

    await _list_chat_sessions(gw, console)
    assert console.print.call_count == 1


@pytest.mark.asyncio
@patch("kesoku.cli_chat._list_chat_sessions", new_callable=AsyncMock)
async def test_run_cli_chat_async_list(mock_list: AsyncMock) -> None:
    """Test run_cli_chat_async delegates to list sessions."""
    await run_cli_chat_async(message=None, list_sessions=True, resume=None, resume_latest=False, show_history=None)
    mock_list.assert_called_once()

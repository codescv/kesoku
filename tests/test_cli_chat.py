"""Unit tests for Kesoku CLI chat runner module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from kesoku.cli_chat import _list_chat_sessions, run_cli_chat_async
from kesoku.db import Session


@pytest.mark.asyncio
@patch("kesoku.cli_chat.logger.info")
async def test_list_chat_sessions_empty(mock_info: MagicMock) -> None:
    """Test listing chat sessions when none exist."""
    gw = MagicMock()
    gw.list_sessions = AsyncMock(return_value=[])
    console = MagicMock(spec=Console)

    await _list_chat_sessions(gw, console)
    mock_info.assert_called_once_with("No chat sessions found.")


@pytest.mark.asyncio
async def test_list_chat_sessions_with_data() -> None:
    """Test listing chat sessions with existing sessions."""
    gw = MagicMock()
    gw.list_sessions = AsyncMock(
        return_value=[Session(id="s1", title="Test Session", created_at=1000.0, updated_at=1000.0)]
    )
    console = MagicMock(spec=Console)

    await _list_chat_sessions(gw, console)
    assert console.print.call_count == 1


@pytest.mark.asyncio
@patch("kesoku.cli_chat.Gateway")
@patch("kesoku.cli_chat._list_chat_sessions", new_callable=AsyncMock)
async def test_run_cli_chat_async_list(mock_list: AsyncMock, mock_gateway: MagicMock) -> None:
    """Test run_cli_chat_async delegates to list sessions."""
    await run_cli_chat_async(message=None, list_sessions=True, resume=None, resume_latest=False, show_history=None)
    mock_list.assert_called_once()


@pytest.mark.asyncio
@patch("kesoku.cli_chat.Gateway")
@patch("kesoku.cli_chat.build_clean_history", new_callable=AsyncMock)
async def test_run_cli_chat_async_show_history_phased(mock_build: AsyncMock, mock_gateway: MagicMock) -> None:
    """Test run_cli_chat_async show_history defaults to phased order."""
    gw_instance = mock_gateway.return_value
    gw_instance.get_session = AsyncMock(return_value=Session(id="s1", title="T", created_at=1, updated_at=1))
    mock_build.return_value = []

    await run_cli_chat_async(
        message=None,
        list_sessions=False,
        resume=None,
        resume_latest=False,
        show_history="s1",
        grouped=False,
    )
    mock_build.assert_called_once_with(gateway=gw_instance, session_id="s1", order="phased", heal_orphans=False)


@pytest.mark.asyncio
@patch("kesoku.cli_chat.Gateway")
@patch("kesoku.cli_chat.build_clean_history", new_callable=AsyncMock)
async def test_run_cli_chat_async_show_history_grouped(mock_build: AsyncMock, mock_gateway: MagicMock) -> None:
    """Test run_cli_chat_async show_history uses grouped order when grouped=True."""
    gw_instance = mock_gateway.return_value
    gw_instance.get_session = AsyncMock(return_value=Session(id="s1", title="T", created_at=1, updated_at=1))
    mock_build.return_value = []

    await run_cli_chat_async(
        message=None,
        list_sessions=False,
        resume=None,
        resume_latest=False,
        show_history="s1",
        grouped=True,
    )
    mock_build.assert_called_once_with(gateway=gw_instance, session_id="s1", order="grouped", heal_orphans=False)

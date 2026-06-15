"""Unit tests for Kesoku CLI chat runner module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from kesoku.cli.chat import _list_chat_sessions, run_cli_chat_async
from kesoku.db import Session


@pytest.mark.asyncio
@patch("kesoku.cli.chat.logger.info")
async def test_list_chat_sessions_empty(mock_info: MagicMock) -> None:
    """Test listing chat sessions when none exist."""
    gw = MagicMock()
    mock_db = AsyncMock()
    gw.db = mock_db
    mock_db.list_sessions = AsyncMock(return_value=[])
    console = MagicMock(spec=Console)

    await _list_chat_sessions(gw, console)
    mock_info.assert_called_once_with("No chat sessions found.")


@pytest.mark.asyncio
async def test_list_chat_sessions_with_data() -> None:
    """Test listing chat sessions with existing sessions."""
    gw = MagicMock()
    mock_db = AsyncMock()
    gw.db = mock_db
    mock_db.list_sessions = AsyncMock(
        return_value=[Session(id="s1", title="Test Session", created_at=1000.0, updated_at=1000.0)]
    )
    console = MagicMock(spec=Console)

    await _list_chat_sessions(gw, console)
    assert console.print.call_count == 1


@pytest.mark.asyncio
@patch("kesoku.cli.chat.Gateway")
@patch("kesoku.cli.chat._list_chat_sessions", new_callable=AsyncMock)
async def test_run_cli_chat_async_list(mock_list: AsyncMock, mock_gateway: MagicMock) -> None:
    """Test run_cli_chat_async delegates to list sessions."""
    await run_cli_chat_async(message=None, list_sessions=True, resume=None, resume_latest=False, show_history=None)
    mock_list.assert_called_once()


@pytest.mark.asyncio
@patch("kesoku.cli.chat.Gateway")
@patch("kesoku.cli.chat.build_history", new_callable=AsyncMock)
async def test_run_cli_chat_async_show_history_phased(mock_build: AsyncMock, mock_gateway: MagicMock) -> None:
    """Test run_cli_chat_async show_history defaults to phased order."""
    gw_instance = mock_gateway.return_value
    mock_db = AsyncMock()
    gw_instance.db = mock_db
    mock_db.get_session = AsyncMock(return_value=Session(id="s1", title="T", created_at=1, updated_at=1))
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
@patch("kesoku.cli.chat.Gateway")
@patch("kesoku.cli.chat.build_history", new_callable=AsyncMock)
async def test_run_cli_chat_async_show_history_grouped(mock_build: AsyncMock, mock_gateway: MagicMock) -> None:
    """Test run_cli_chat_async show_history uses grouped order when grouped=True."""
    gw_instance = mock_gateway.return_value
    mock_db = AsyncMock()
    gw_instance.db = mock_db
    mock_db.get_session = AsyncMock(return_value=Session(id="s1", title="T", created_at=1, updated_at=1))
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


@pytest.mark.asyncio
@patch("kesoku.cli.chat.Gateway")
@patch("kesoku.cli.chat.Agent")
@patch("kesoku.cli.chat.CLIChatbot")
@patch("kesoku.cli.chat.build_history", new_callable=AsyncMock)
@patch("kesoku.cli.chat.build_sys_prompt")
async def test_run_cli_chat_async_resume_updates_prompt(
    mock_build_sys_prompt: MagicMock,
    mock_build_history: AsyncMock,
    mock_cli_bot: MagicMock,
    mock_agent: MagicMock,
    mock_gateway_class: MagicMock,
) -> None:
    """Verify resuming a session in CLI chat updates the system prompt."""
    gw_instance = mock_gateway_class.return_value
    mock_db = AsyncMock()
    gw_instance.db = mock_db
    gw_instance.post = AsyncMock()

    session_id = "s_resume"
    existing_session = Session(id=session_id, title="Test", created_at=1, updated_at=1, system_prompt="old prompt")

    mock_db.get_session = AsyncMock(return_value=existing_session)
    mock_db.update_session_updated_at = AsyncMock()
    mock_db.update_session_system_prompt = AsyncMock()

    mock_build_history.return_value = []
    mock_build_sys_prompt.return_value = "new updated prompt"

    # Mock CLIChatbot and Agent tasks to complete quickly
    cli_bot_instance = mock_cli_bot.return_value
    cli_bot_instance.start = AsyncMock()
    cli_bot_instance.stop = MagicMock()
    cli_bot_instance.final_response_event = MagicMock()
    cli_bot_instance.final_response_event.wait = AsyncMock()

    agent_instance = mock_agent.return_value
    agent_instance.start = AsyncMock()
    agent_instance.stop = MagicMock()

    await run_cli_chat_async(
        message="hello again",
        list_sessions=False,
        resume=session_id,
        resume_latest=False,
        show_history=None,
    )

    # Verify build_sys_prompt was called with the session
    mock_build_sys_prompt.assert_called_once_with(session=existing_session)
    # Verify DB update was called
    mock_db.update_session_system_prompt.assert_called_once_with(session_id, "new updated prompt")

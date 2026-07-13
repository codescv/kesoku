"""Unit tests for Kesoku CLI chat runner module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kesoku.cli.chat import run_cli_chat_async
from kesoku.db import Session


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
        resume=session_id,
        resume_latest=False,
    )

    # Verify build_sys_prompt was called with the session
    mock_build_sys_prompt.assert_called_once_with(session=existing_session)
    # Verify DB update was called
    mock_db.update_session_system_prompt.assert_called_once_with(session_id, "new updated prompt")

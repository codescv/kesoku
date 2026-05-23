import asyncio
import datetime
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
import tomli_w

from kesoku.cron import CronManager, cron_matches, load_cronjobs


def test_cron_matches_exact():
    # Match exactly at 12:00 on 15th of April (Wednesday)
    dt = datetime.datetime(2026, 4, 15, 12, 0)
    # Wednesday matches weekday 3 (dt.isoweekday() is 3, 3 % 7 = 3)
    assert cron_matches("0 12 15 4 3", dt) is True
    assert cron_matches("0 12 15 4 4", dt) is False  # Wrong weekday
    assert cron_matches("5 12 15 4 3", dt) is False  # Wrong minute


def test_cron_matches_wildcard_and_steps():
    # Test every 13 minutes between hours 10 and 22
    dt = datetime.datetime(2026, 5, 19, 10, 13)
    assert cron_matches("*/13 10-22 * * *", dt) is True

    dt_non_matching = datetime.datetime(2026, 5, 19, 10, 14)
    assert cron_matches("*/13 10-22 * * *", dt_non_matching) is False


def test_cron_matches_ranges_and_lists():
    # Test range list and specific day of week list
    dt = datetime.datetime(2026, 5, 19, 15, 30)  # Tuesday (isoweekday=2 % 7 = 2)
    # matches: minute 30, hour 15 (in range 12-18), day 19, month 5, day of week 2 or 4
    assert cron_matches("30 12-18 * * 2,4", dt) is True

    # Non-matching day of week
    dt_wrong_dow = datetime.datetime(2026, 5, 20, 15, 30)  # Wednesday (isoweekday=3 % 7 = 3)
    assert cron_matches("30 12-18 * * 2,4", dt_wrong_dow) is False


def test_load_cronjobs_single():
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        toml_data = {
            "job": {
                "schedule": "*/13 10-22 * * *",
                "prompt": "prompts/test.md",
                "channel_id": "12345",
                "chatbot_id": "discord",
            }
        }
        with open(tmp.name, "wb") as f:
            tomli_w.dump(toml_data, f)
        tmp_path = tmp.name

    try:
        jobs = load_cronjobs(tmp_path)
        assert len(jobs) == 1
        assert jobs[0]["schedule"] == "*/13 10-22 * * *"
        assert jobs[0]["prompt"] == "prompts/test.md"
    finally:
        os.unlink(tmp_path)


def test_load_cronjobs_multiple():
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        toml_data = {
            "job": [
                {"schedule": "0 10 * * *", "prompt": "1.md", "channel_id": "1"},
                {"schedule": "0 11 * * *", "prompt": "2.md", "channel_id": "2"},
            ]
        }
        with open(tmp.name, "wb") as f:
            tomli_w.dump(toml_data, f)
        tmp_path = tmp.name

    try:
        jobs = load_cronjobs(tmp_path)
        assert len(jobs) == 2
        assert jobs[0]["prompt"] == "1.md"
        assert jobs[1]["prompt"] == "2.md"
    finally:
        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_cron_manager_duplicates_and_trigger():
    mock_bot = MagicMock()
    mock_bot.chatbot_id = "discord"
    mock_bot.trigger_cronjob = AsyncMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)  # noqa: ASYNC240
        prompt_file_path = os.path.join(tmpdir_real, "test_prompt.md")
        with open(prompt_file_path, "w") as f:  # noqa: ASYNC230
            f.write("Hello from cron!")

        job = {
            "schedule": "* * * * *",  # always matches
            "prompt": "test_prompt.md",
            "channel_id": "999",
            "chatbot_id": "discord",
            "mention_user_id": "111",
        }

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)

        # First execution: matches and triggers
        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)  # yield to background tasks
        mock_bot.trigger_cronjob.assert_called_once_with(
            channel_id="999",
            prompt_content="Hello from cron!",
            mention_user_id="111",
        )

        # Second execution within same minute: matches but skipped (prevent duplicates)
        mock_bot.trigger_cronjob.reset_mock()
        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)
        mock_bot.trigger_cronjob.assert_not_called()


@pytest.mark.asyncio
async def test_cron_manager_path_traversal():
    mock_bot = MagicMock()
    mock_bot.chatbot_id = "discord"
    mock_bot.trigger_cronjob = AsyncMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)  # noqa: ASYNC240
        job = {
            "schedule": "* * * * *",
            "prompt": "../outside_prompt.md",  # path traversal attempt
            "channel_id": "999",
            "chatbot_id": "discord",
        }

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)
        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)
        mock_bot.trigger_cronjob.assert_not_called()


@pytest.mark.asyncio
async def test_cron_manager_wechat_optional_channel():
    mock_bot = MagicMock()
    mock_bot.chatbot_id = "wechat"
    mock_bot.trigger_cronjob = AsyncMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)  # noqa: ASYNC240
        prompt_file_path = os.path.join(tmpdir_real, "test_prompt.md")
        with open(prompt_file_path, "w") as f:  # noqa: ASYNC230
            f.write("Hello WeChat cron!")

        job = {
            "schedule": "* * * * *",
            "prompt": "test_prompt.md",
            "chatbot_id": "wechat",
            # channel_id is omitted
        }

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)
        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)
        mock_bot.trigger_cronjob.assert_called_once_with(
            channel_id=None,
            prompt_content="Hello WeChat cron!",
            mention_user_id=None,
        )

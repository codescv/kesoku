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


def create_mock_bot(chatbot_id="discord"):
    bot = MagicMock()
    bot.chatbot_id = chatbot_id
    bot.trigger_cronjob = AsyncMock()

    gateway = MagicMock()
    db = AsyncMock()
    db.get_cronjob_sent_stats_today.return_value = (0, None)
    db.get_last_message_timestamp.return_value = None
    gateway.db = db
    bot.gateway = gateway
    return bot, db


@pytest.mark.asyncio
async def test_cron_manager_duplicates_and_trigger():
    mock_bot, _ = create_mock_bot()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
        prompt_file_path = os.path.join(tmpdir_real, "test_prompt.md")
        with open(prompt_file_path, "w") as f:
            f.write("Hello from cron!")

        job = {
            "schedule": "* * * * *",  # always matches
            "prompt": "test_prompt.md",
            "channel_id": "999",
            "chatbot_id": "discord",
            "mention_user_id": "111",
        }

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)

        fixed_now = datetime.datetime(2026, 6, 14, 12, 0, 0)
        from unittest.mock import patch

        with patch("kesoku.cron.datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            # First execution: matches and triggers
            await manager._check_and_trigger_jobs([job])
            await asyncio.sleep(0.1)  # yield to background tasks
            mock_bot.trigger_cronjob.assert_called_once_with(
                channel_id="999",
                prompt_content="Hello from cron!",
                mention_user_id="111",
                tag=None,
            )

            # Second execution within same minute: matches but skipped (prevent duplicates)
            mock_bot.trigger_cronjob.reset_mock()
            await manager._check_and_trigger_jobs([job])
            await asyncio.sleep(0.1)
            mock_bot.trigger_cronjob.assert_not_called()


@pytest.mark.asyncio
async def test_cron_manager_path_traversal():
    mock_bot, _ = create_mock_bot()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
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
async def test_cron_manager_wechat_missing_channel():
    mock_bot, _ = create_mock_bot(chatbot_id="wechat")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
        prompt_file_path = os.path.join(tmpdir_real, "test_prompt.md")
        with open(prompt_file_path, "w") as f:
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
        mock_bot.trigger_cronjob.assert_not_called()


@pytest.mark.asyncio
async def test_cron_manager_min_idle_time():
    mock_bot, mock_db = create_mock_bot()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
        prompt_file_path = os.path.join(tmpdir_real, "test_prompt.md")
        with open(prompt_file_path, "w") as f:
            f.write("Hello!")

        job = {
            "schedule": "* * * * *",
            "prompt": "test_prompt.md",
            "channel_id": "999",
            "chatbot_id": "discord",
            "min_idle_time": 60,
        }

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)

        # Case 1: Last message was 30 seconds ago (not idle enough)
        mock_db.get_last_message_timestamp.return_value = datetime.datetime.now().timestamp() - 30

        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)
        mock_bot.trigger_cronjob.assert_not_called()

        # Reset minutes cache in manager to allow checking again
        manager.last_executed_minute.clear()

        # Case 2: Last message was 90 seconds ago (idle enough)
        mock_db.get_last_message_timestamp.return_value = datetime.datetime.now().timestamp() - 90

        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)
        mock_bot.trigger_cronjob.assert_called_once_with(
            channel_id="999",
            prompt_content="Hello!",
            mention_user_id=None,
            tag=None,
        )


@pytest.mark.asyncio
async def test_cron_manager_max_messages():
    mock_bot, mock_db = create_mock_bot()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
        prompt_file_path = os.path.join(tmpdir_real, "test_prompt.md")
        with open(prompt_file_path, "w") as f:
            f.write("Hello!")

        job = {
            "schedule": "* * * * *",
            "prompt": "test_prompt.md",
            "channel_id": "999",
            "chatbot_id": "discord",
            "max_messages": 3,
        }

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)

        # Case 1: Already sent 3 messages today -> should skip
        mock_db.get_cronjob_sent_stats_today.return_value = (3, None)

        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)
        mock_bot.trigger_cronjob.assert_not_called()

        # Reset minutes cache in manager to allow checking again
        manager.last_executed_minute.clear()

        # Case 2: Only sent 2 messages today -> should trigger
        mock_db.get_cronjob_sent_stats_today.return_value = (2, None)

        await manager._check_and_trigger_jobs([job])
        await asyncio.sleep(0.1)
        mock_bot.trigger_cronjob.assert_called_once_with(
            channel_id="999",
            prompt_content="Hello!",
            mention_user_id=None,
            tag=None,
        )


@pytest.mark.asyncio
async def test_cron_manager_accumulating_probability():
    mock_bot, _ = create_mock_bot()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
        prompt_file_path = os.path.join(tmpdir_real, "test_prompt.md")
        with open(prompt_file_path, "w") as f:
            f.write("Hello!")

        job = {
            "schedule": "* * * * *",
            "prompt": "test_prompt.md",
            "channel_id": "999",
            "chatbot_id": "discord",
            "probability": 0.1,
        }

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)

        # Tick 1: random.random() is 0.2 (> 0.1) -> should skip.
        # p should increase to 0.1 * 1.1 = 0.11
        from unittest.mock import patch

        with patch("random.random", return_value=0.2):
            await manager._check_and_trigger_jobs([job])
            await asyncio.sleep(0.1)
            mock_bot.trigger_cronjob.assert_not_called()

        job_key = ("discord", "999", "test_prompt.md")
        assert manager.job_states[job_key]["current_p"] == pytest.approx(0.11)

        # Reset minutes cache
        manager.last_executed_minute.clear()

        # Tick 2: random.random() is 0.12 (> 0.11) -> should skip.
        # p should increase to 0.11 * 1.1 = 0.121
        with patch("random.random", return_value=0.12):
            await manager._check_and_trigger_jobs([job])
            await asyncio.sleep(0.1)
            mock_bot.trigger_cronjob.assert_not_called()

        assert manager.job_states[job_key]["current_p"] == pytest.approx(0.121)

        # Reset minutes cache
        manager.last_executed_minute.clear()

        # Tick 3: random.random() is 0.10 (<= 0.121) -> should trigger!
        # p should reset to 0.1
        with patch("random.random", return_value=0.10):
            await manager._check_and_trigger_jobs([job])
            await asyncio.sleep(0.1)
            mock_bot.trigger_cronjob.assert_called_once()

        assert manager.job_states[job_key]["current_p"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_cron_manager_probability_daily_reset():
    mock_bot, _ = create_mock_bot()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
        prompt_file_path = os.path.join(tmpdir_real, "test_prompt.md")
        with open(prompt_file_path, "w") as f:
            f.write("Hello!")

        job = {
            "schedule": "* * * * *",
            "prompt": "test_prompt.md",
            "channel_id": "999",
            "chatbot_id": "discord",
            "probability": 0.1,
        }

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)
        job_key = ("discord", "999", "test_prompt.md")

        # Manually set state with increased p and yesterday's date
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        manager.job_states[job_key] = {"current_p": 0.5, "last_reset_date": yesterday}

        # Tick: should reset p to 0.1 because of new day
        # random.random() is 0.2 (> 0.1) -> should skip
        from unittest.mock import patch

        with patch("random.random", return_value=0.2):
            await manager._check_and_trigger_jobs([job])
            await asyncio.sleep(0.1)
            mock_bot.trigger_cronjob.assert_not_called()

        assert manager.job_states[job_key]["last_reset_date"] == datetime.date.today()
        # Since it skipped, it should have increased the RESET p (0.1 * 1.1 = 0.11)
        assert manager.job_states[job_key]["current_p"] == pytest.approx(0.11)


@pytest.mark.asyncio
async def test_cron_manager_get_all_jobs():
    mock_bot, _ = create_mock_bot()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
        cron_toml_path = os.path.join(tmpdir_real, "cronjob.toml")
        toml_data = {
            "job": [
                {
                    "schedule": "0 10 * * *",
                    "prompt": "1.md",
                    "channel_id": "1",
                    "chatbot_id": "discord",
                    "tag": "tag1",
                },
                {
                    "schedule": "0 11 * * *",
                    "prompt": "2.md",
                    "channel_id": "2",
                    "chatbot_id": "discord",
                },
            ]
        }
        with open(cron_toml_path, "wb") as f:
            tomli_w.dump(toml_data, f)

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)
        jobs = manager.get_all_jobs()
        assert len(jobs) == 2
        assert jobs[0]["tag"] == "tag1"
        assert "tag" not in jobs[1]


@pytest.mark.asyncio
async def test_cron_manager_trigger_jobs_by_tag():
    mock_bot, _ = create_mock_bot()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_real = os.path.realpath(tmpdir)
        cron_toml_path = os.path.join(tmpdir_real, "cronjob.toml")

        # Create prompt files
        prompt1_path = os.path.join(tmpdir_real, "1.md")
        with open(prompt1_path, "w") as f:
            f.write("Prompt 1")
        prompt2_path = os.path.join(tmpdir_real, "2.md")
        with open(prompt2_path, "w") as f:
            f.write("Prompt 2")

        toml_data = {
            "job": [
                {
                    "schedule": "0 10 * * *",
                    "prompt": "1.md",
                    "channel_id": "1",
                    "chatbot_id": "discord",
                    "tag": "trigger_me",
                },
                {
                    "schedule": "0 11 * * *",
                    "prompt": "2.md",
                    "channel_id": "2",
                    "chatbot_id": "discord",
                    "tag": "dont_trigger_me",
                },
                {
                    "schedule": "0 12 * * *",
                    "prompt": "1.md",
                    "channel_id": "3",
                    "chatbot_id": "discord",
                    "tag": "trigger_me",
                },
            ]
        }
        with open(cron_toml_path, "wb") as f:
            tomli_w.dump(toml_data, f)

        manager = CronManager(chatbots=[mock_bot], config_dir=tmpdir_real)

        # Trigger "trigger_me"
        count = await manager.trigger_jobs_by_tag("trigger_me")
        assert count == 2
        await asyncio.sleep(0.1)  # yield to background tasks

        assert mock_bot.trigger_cronjob.call_count == 2
        # Verify calls
        calls = sorted(mock_bot.trigger_cronjob.call_args_list, key=lambda c: c[1]["channel_id"])
        assert calls[0][1]["channel_id"] == "1"
        assert calls[0][1]["prompt_content"] == "Prompt 1"
        assert calls[1][1]["channel_id"] == "3"
        assert calls[1][1]["prompt_content"] == "Prompt 1"

        # Trigger non-existent tag
        mock_bot.trigger_cronjob.reset_mock()
        count = await manager.trigger_jobs_by_tag("non_existent")
        assert count == 0
        await asyncio.sleep(0.1)
        mock_bot.trigger_cronjob.assert_not_called()

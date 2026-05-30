"""Cron scheduling and management module for Kesoku AI Agent framework.

Provides a pure-Python cron expression parser/matcher and background CronManager.
"""

import asyncio
import datetime
import os
import random
import tomllib
from typing import Any

from kesoku.async_utils import (
    async_exists,
    async_read_text_file,
    async_realpath,
)
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


def _field_matches(field_str: str, value: int, min_val: int, max_val: int) -> bool:
    """Determine if a datetime component matches a specific cron field string.

    Args:
        field_str: The cron field pattern (e.g. '*', '*/13', '10-22', '1,3,5').
        value: The actual datetime component value.
        min_val: The minimum allowable value for this field.
        max_val: The maximum allowable value for this field.

    Returns:
        True if the value matches the cron field expression, False otherwise.
    """
    if field_str == "*":
        return True

    if "," in field_str:
        return any(_field_matches(part, value, min_val, max_val) for part in field_str.split(","))

    step = 1
    if "/" in field_str:
        base_str, step_str = field_str.split("/", 1)
        try:
            step = int(step_str)
        except ValueError:
            return False
    else:
        base_str = field_str

    if base_str == "*":
        return (value - min_val) % step == 0
    elif "-" in base_str:
        if base_str.count("-") != 1:
            return False
        start_str, end_str = base_str.split("-", 1)
        try:
            start = int(start_str)
            end = int(end_str)
        except ValueError:
            return False
        return start <= value <= end and (value - start) % step == 0
    else:
        try:
            single_val = int(base_str)
        except ValueError:
            return False
        return value == single_val and (value - single_val) % step == 0


def cron_matches(schedule: str, dt: datetime.datetime) -> bool:
    """Check if a datetime matches a standard 5-field cron schedule.

    Fields:
        0: Minute (0-59)
        1: Hour (0-23)
        2: Day of month (1-31)
        3: Month (1-12)
        4: Day of week (0-6, where Sunday is 0)

    Args:
        schedule: Standard cron schedule string (e.g. '*/13 10-22 * * *').
        dt: The datetime object to check.

    Returns:
        True if the datetime matches the schedule, False otherwise.
    """
    fields = schedule.split()
    if len(fields) != 5:
        logger.warning(f"Invalid cron schedule pattern (expected 5 fields): '{schedule}'")
        return False

    # dt.isoweekday() % 7 yields: 0 for Sunday, 1 for Monday, ..., 6 for Saturday.
    minute = dt.minute
    hour = dt.hour
    day = dt.day
    month = dt.month
    day_of_week = dt.isoweekday() % 7

    return (
        _field_matches(fields[0], minute, 0, 59)
        and _field_matches(fields[1], hour, 0, 23)
        and _field_matches(fields[2], day, 1, 31)
        and _field_matches(fields[3], month, 1, 12)
        and _field_matches(fields[4], day_of_week, 0, 6)
    )


def load_cronjobs(toml_path: str) -> list[dict[str, Any]]:
    """Load cronjobs configuration from a TOML file.

    Supports single table [job] or list of tables [[job]].

    Args:
        toml_path: Absolute path to the cronjob.toml file.

    Returns:
        A list of job configuration dictionaries.
    """
    if not os.path.exists(toml_path):
        return []
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        jobs = data.get("job", [])
        if isinstance(jobs, dict):
            return [jobs]
        elif isinstance(jobs, list):
            return [j for j in jobs if isinstance(j, dict)]
        return []
    except Exception as e:
        logger.error(f"Failed to load cronjobs from {toml_path}: {e}")
        return []


class CronManager:
    """Manager that polls and runs scheduled background cron tasks."""

    def __init__(self, chatbots: list[Any], config_dir: str) -> None:
        """Initialize the CronManager.

        Args:
            chatbots: Active chatbot adapters list.
            config_dir: Directory containing the config and cron files.
        """
        self.chatbots = {bot.chatbot_id: bot for bot in chatbots}
        self.config_dir = config_dir
        self.last_executed_minute: dict[int, int] = {}

    async def start(self) -> None:
        """Start the background cron task checking loop."""
        logger.info("CronManager background loop started.")
        cron_toml_path = os.path.join(self.config_dir, "cronjob.toml")

        while True:
            try:
                jobs = load_cronjobs(cron_toml_path)
                if jobs:
                    await self._check_and_trigger_jobs(jobs)
            except asyncio.CancelledError:
                logger.info("CronManager background loop stopped.")
                break
            except Exception as e:
                logger.error(f"Error in CronManager background loop: {e}", exc_info=True)

            await asyncio.sleep(10)

    async def _check_and_trigger_jobs(self, jobs: list[dict[str, Any]]) -> None:
        """Check all jobs and trigger them if their schedule matches the current time.

        Args:
            jobs: Loaded job configurations list.
        """
        now = datetime.datetime.now()
        current_minute_epoch = int(now.timestamp() // 60)

        for idx, job in enumerate(jobs):
            schedule = job.get("schedule")
            if not schedule:
                continue

            if cron_matches(schedule, now):
                # Ensure the job is run at most once per minute
                if self.last_executed_minute.get(idx) == current_minute_epoch:
                    continue

                self.last_executed_minute[idx] = current_minute_epoch

                probability = job.get("probability")
                if probability is not None:
                    try:
                        prob = float(probability)
                        if random.random() > prob:
                            logger.info(f"Job {idx} matched schedule but was skipped by probability filter.")
                            continue
                    except ValueError:
                        logger.warning(f"Invalid probability format for job {idx}: {probability}")

                chatbot_id = job.get("chatbot_id")
                channel_id = job.get("channel_id")

                # Check min_idle_time_seconds if channel_id is explicitly provided
                min_idle_time_seconds = job.get("min_idle_time_seconds")
                if min_idle_time_seconds is not None:
                    try:
                        min_idle = float(min_idle_time_seconds)
                        if channel_id and chatbot_id:
                            bot = self.chatbots.get(chatbot_id)
                            if bot:
                                last_msg_ts = await bot.gateway.get_last_message_timestamp(chatbot_id, channel_id)
                                if last_msg_ts is not None:
                                    idle_time = now.timestamp() - last_msg_ts
                                    if idle_time < min_idle:
                                        logger.info(
                                            f"Job {idx} matched schedule but was skipped because channel {channel_id} "
                                            f"has been idle for only {idle_time:.1f}s (required {min_idle}s)."
                                        )
                                        continue
                    except ValueError:
                        logger.warning(f"Invalid min_idle_time_seconds format for job {idx}: {min_idle_time_seconds}")

                # Check daily_target and min_interval_seconds if channel_id is explicitly provided
                daily_target = job.get("daily_target")
                min_interval_seconds = job.get("min_interval_seconds")
                if (daily_target is not None or min_interval_seconds is not None) and channel_id and chatbot_id:
                    bot = self.chatbots.get(chatbot_id)
                    if bot:
                        count, last_ts = await bot.gateway.get_cronjob_sent_stats_today(chatbot_id, channel_id)

                        if daily_target is not None:
                            try:
                                target = int(daily_target)
                                if count >= target:
                                    logger.info(
                                        f"Job {idx} matched schedule but was skipped because the daily target "
                                        f"of {target} messages has already been reached today ({count} sent)."
                                    )
                                    continue
                            except ValueError:
                                logger.warning(f"Invalid daily_target format for job {idx}: {daily_target}")

                        if min_interval_seconds is not None and last_ts is not None:
                            try:
                                min_interval = float(min_interval_seconds)
                                elapsed = now.timestamp() - last_ts
                                if elapsed < min_interval:
                                    logger.info(
                                        f"Job {idx} matched schedule but was skipped because minimum interval "
                                        f"of {min_interval}s has not elapsed since last trigger "
                                        f"({elapsed:.1f}s elapsed)."
                                    )
                                    continue
                            except ValueError:
                                logger.warning(
                                    f"Invalid min_interval_seconds format for job {idx}: {min_interval_seconds}"
                                )

                # Run the job asynchronously in the background
                asyncio.create_task(self._execute_job(job, idx))

    async def _execute_job(self, job: dict[str, Any], job_idx: int) -> None:
        """Execute a single cron job trigger.

        Args:
            job: The job configuration dictionary.
            job_idx: The index of the job in the config.
        """
        chatbot_id = job.get("chatbot_id")
        channel_id = job.get("channel_id")
        prompt_path = job.get("prompt")
        mention_user_id = job.get("mention_user_id")

        if not chatbot_id:
            logger.error(f"Cronjob {job_idx} is missing chatbot_id field.")
            return

        if chatbot_id == "cronjob" and not channel_id:
            channel_id = f"silent_{job_idx}"

        if not prompt_path:
            logger.warning(f"Cronjob {job_idx} is missing prompt field.")
            return

        if not channel_id and chatbot_id != "wechat":
            logger.warning(f"Cronjob {job_idx} is missing channel_id field.")
            return

        bot = self.chatbots.get(chatbot_id)
        if not bot:
            logger.warning(f"Chatbot '{chatbot_id}' for cronjob {job_idx} is not available.")
            return

        # Safe path resolution to prevent path traversal
        full_prompt_path = prompt_path
        if not os.path.isabs(prompt_path):
            full_prompt_path = os.path.join(self.config_dir, prompt_path)

        abs_prompt_path = await async_realpath(full_prompt_path)
        abs_config_dir = await async_realpath(self.config_dir)
        if not abs_prompt_path.startswith(abs_config_dir):
            logger.error(f"Security Warning: Path traversal attempt blocked in cronjob {job_idx}: {prompt_path}")
            return

        if not await async_exists(abs_prompt_path):
            logger.error(f"Prompt file not found for cronjob {job_idx}: {abs_prompt_path}")
            return

        try:
            prompt_content = await async_read_text_file(abs_prompt_path)
        except Exception as e:
            logger.error(f"Failed to read prompt file for cronjob {job_idx}: {e}")
            return

        logger.info(
            f"Triggering scheduled cronjob {job_idx} on chatbot '{chatbot_id}' "
            f"in channel {channel_id or 'auto-resolved'}..."
        )

        try:
            if hasattr(bot, "trigger_cronjob"):
                kwargs = {}
                min_idle_time_seconds = job.get("min_idle_time_seconds")
                if min_idle_time_seconds is not None:
                    try:
                        kwargs["min_idle_time_seconds"] = float(min_idle_time_seconds)
                    except ValueError:
                        pass

                daily_target = job.get("daily_target")
                if daily_target is not None:
                    try:
                        kwargs["daily_target"] = int(daily_target)
                    except ValueError:
                        pass

                min_interval_seconds = job.get("min_interval_seconds")
                if min_interval_seconds is not None:
                    try:
                        kwargs["min_interval_seconds"] = float(min_interval_seconds)
                    except ValueError:
                        pass

                await bot.trigger_cronjob(
                    channel_id=str(channel_id) if channel_id else None,
                    prompt_content=prompt_content,
                    mention_user_id=str(mention_user_id) if mention_user_id else None,
                    **kwargs,
                )
            else:
                logger.error(f"Chatbot '{chatbot_id}' does not support trigger_cronjob.")
        except Exception as e:
            logger.error(f"Failed executing scheduled cronjob {job_idx}: {e}", exc_info=True)

"""Time utility functions for Kesoku AI Agent."""

import datetime
import logging

import tzlocal

logger = logging.getLogger(__name__)


def parse_time_to_timestamp(time_str: str | None) -> float | None:
    """Parse an ISO 8601 date/time string or float epoch string into a Unix epoch timestamp in seconds.

    If the date/time string does not contain a timezone offset, the local system timezone
    is attached by default. Date-only strings (YYYY-MM-DD) are parsed as start-of-day (T00:00:00).

    Args:
        time_str: ISO string (e.g. '2026-06-15', '2026-06-15T12:00:00+08:00') or float timestamp.

    Returns:
        Float Unix epoch timestamp, or None if parsing fails.
    """
    if not time_str:
        return None
    time_str = time_str.strip()
    try:
        # Check if just YYYY-MM-DD date
        if len(time_str) == 10 and "-" in time_str:
            time_str = f"{time_str}T00:00:00"

        dt = datetime.datetime.fromisoformat(time_str)
        # Check if timezone is attached
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            local_tz = tzlocal.get_localzone()
            dt = dt.replace(tzinfo=local_tz)
        return dt.timestamp()
    except Exception as e:
        logger.debug(f"Failed to parse time string '{time_str}' as ISO format: {e}")
        try:
            return float(time_str)
        except ValueError:
            logger.warning(f"Invalid time format: '{time_str}'. Must be ISO 8601 or float epoch seconds.")
            return None

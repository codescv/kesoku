"""Time utility functions for Kesoku AI Agent."""

import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import tzlocal

logger = logging.getLogger(__name__)

DEFAULT_HOME_TZ = "Asia/Shanghai"
DEFAULT_HOME_PLACE = "家"


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


def _valid_zone(name: str) -> ZoneInfo | None:
    """Return `ZoneInfo(name)` when `name` is a valid non-empty IANA zone, else `None`."""
    cleaned = str(name or "").strip()
    if not cleaned:
        return None
    try:
        return ZoneInfo(cleaned)
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        return None


def load_user_clock(
    path: str | os.PathLike[str] | None,
    *,
    now_epoch: float | None = None,
) -> dict[str, Any] | None:
    """Load the user's physical timezone contract JSON from `path` (read fresh each call).

    Missing files, invalid JSON, or invalid IANA timezone identifiers return `None`.
    When `expires_at` is present and has passed relative to `now_epoch` (or `time.time()`),
    returns `(home_tz, home_place)` instead of the expired trip zone.

    Args:
        path: Filesystem path to `user_clock.json`, or None/empty.
        now_epoch: Optional reference Unix timestamp in seconds (defaults to `time.time()`).

    Returns:
        Normalized clock dictionary (`tz`, `zone`, `place`, `home_tz`, `home_place`, `expired`),
        or `None` when absent or invalid.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    tz_name = str(data.get("tz") or "").strip()
    zone = _valid_zone(tz_name)
    if zone is None:
        return None

    raw_home_tz = str(data.get("home_tz") or DEFAULT_HOME_TZ).strip() or DEFAULT_HOME_TZ
    home_zone = _valid_zone(raw_home_tz)
    if home_zone is None:
        raw_home_tz = DEFAULT_HOME_TZ
        home_zone = ZoneInfo(DEFAULT_HOME_TZ)

    home_place = str(data.get("home_place") or DEFAULT_HOME_PLACE).strip() or DEFAULT_HOME_PLACE
    place = str(data.get("place") or "").strip()

    expires_raw = str(data.get("expires_at") or "").strip()
    if expires_raw:
        try:
            exp_dt = datetime.datetime.fromisoformat(expires_raw)
        except ValueError:
            return None
        if exp_dt.tzinfo is None or exp_dt.tzinfo.utcoffset(exp_dt) is None:
            exp_dt = exp_dt.replace(tzinfo=home_zone)
        ref_now = time.time() if now_epoch is None else float(now_epoch)
        if ref_now >= exp_dt.timestamp():
            return {
                "tz": raw_home_tz,
                "zone": home_zone,
                "place": home_place,
                "home_tz": raw_home_tz,
                "home_place": home_place,
                "expired": True,
            }

    return {
        "tz": tz_name,
        "zone": zone,
        "place": place,
        "home_tz": raw_home_tz,
        "home_place": home_place,
        "expired": False,
    }


def _format_utc_offset(td: datetime.timedelta | None) -> str:
    """Format a `timedelta` UTC offset as `+HH:MM` / `-HH:MM` (supports 30m/45m offsets)."""
    total_min = round((td or datetime.timedelta(0)).total_seconds() / 60)
    sign = "+" if total_min >= 0 else "-"
    hh, mm = divmod(abs(total_min), 60)
    return f"{sign}{hh:02d}:{mm:02d}"


def user_local_stamp(ts: float, uc: dict[str, Any]) -> str:
    """Format Unix timestamp `ts` in the user's `ZoneInfo` (e.g. `2026-11-09 18:59:58 (Sunday) IST UTC+05:30`).

    Args:
        ts: Unix timestamp in seconds.
        uc: Loaded user clock dictionary from `load_user_clock`.

    Returns:
        Formatted local timestamp string with weekday, abbreviation (when alphabetic), and UTC offset.
    """
    zone = uc.get("zone") if isinstance(uc.get("zone"), ZoneInfo) else ZoneInfo(str(uc["tz"]))
    dt = datetime.datetime.fromtimestamp(float(ts), tz=zone)
    off = _format_utc_offset(dt.utcoffset())
    abbr = (dt.tzname() or "").strip()
    if abbr and abbr[0] not in "+-":
        return f"{dt.strftime('%Y-%m-%d %H:%M:%S (%A)')} {abbr} UTC{off}"
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S (%A)')} UTC{off}"


def humanize_gap(seconds: float) -> str:
    """Format elapsed seconds since the previous user message (`<1m` | `31m` | `2h05m` | `1d3h`).

    Args:
        seconds: Non-negative elapsed duration in seconds.

    Returns:
        Compact human-readable gap string.
    """
    sec = max(0.0, float(seconds))
    if sec < 60.0:
        return "<1m"
    if sec < 3600.0:
        return f"{int(sec // 60)}m"
    if sec < 86400.0:
        hours, mins = divmod(int(sec // 60), 60)
        return f"{hours}h" if mins == 0 else f"{hours}h{mins:02d}m"
    days, rem_hours = divmod(int(sec // 3600), 24)
    return f"{days}d" if rem_hours == 0 else f"{days}d{rem_hours}h"

"""Unit tests for `kesoku.utils.time_utils` user clock loader, local stamp, and gap humanizer."""

import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kesoku.utils.time_utils import (
    DEFAULT_HOME_PLACE,
    DEFAULT_HOME_TZ,
    humanize_gap,
    load_user_clock,
    user_local_stamp,
)


def test_load_user_clock_missing_and_invalid(tmp_path: Path) -> None:
    assert load_user_clock(None) is None
    assert load_user_clock("") is None
    assert load_user_clock(tmp_path / "missing.json") is None

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    assert load_user_clock(bad_json) is None

    non_dict = tmp_path / "list.json"
    non_dict.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_user_clock(non_dict) is None

    bad_tz = tmp_path / "bad_tz.json"
    bad_tz.write_text(json.dumps({"tz": "Mars/Olympus"}), encoding="utf-8")
    assert load_user_clock(bad_tz) is None

    bad_exp = tmp_path / "bad_exp.json"
    bad_exp.write_text(json.dumps({"tz": "Asia/Kolkata", "expires_at": "not-a-timestamp"}), encoding="utf-8")
    assert load_user_clock(bad_exp) is None


def test_load_user_clock_kolkata_kathmandu_dst_and_no_stale_cache(tmp_path: Path) -> None:
    clock_file = tmp_path / "user_clock.json"
    # 2026-11-09 21:29:58+08:00 == 2026-11-09 18:59:58+05:30 (Monday)
    ts_nov = datetime.datetime(2026, 11, 9, 21, 29, 58, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()

    clock_file.write_text(
        json.dumps(
            {
                "tz": "Asia/Kolkata",
                "place": "新德里",
                "home_tz": "Asia/Shanghai",
                "home_place": "上海",
                "expires_at": "2026-11-10T06:00:00+08:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    uc_ist = load_user_clock(clock_file, now_epoch=ts_nov)
    assert uc_ist is not None
    assert uc_ist["tz"] == "Asia/Kolkata"
    assert uc_ist["place"] == "新德里"
    assert uc_ist["expired"] is False
    assert user_local_stamp(ts_nov, uc_ist) == "2026-11-09 18:59:58 (Monday) IST UTC+05:30"

    # Immediate overwrite within same 1-second mtime window must be observed fresh (no mtime cache).
    clock_file.write_text(
        json.dumps({"tz": "Asia/Kathmandu", "place": "加德满都"}, ensure_ascii=False), encoding="utf-8"
    )
    uc_npt = load_user_clock(clock_file, now_epoch=ts_nov)
    assert uc_npt is not None and uc_npt["tz"] == "Asia/Kathmandu"
    assert user_local_stamp(ts_nov, uc_npt) == "2026-11-09 19:14:58 (Monday) UTC+05:45"

    # DST zone: America/Los_Angeles in July (-07:00 PDT) vs December (-08:00 PST)
    clock_file.write_text(
        json.dumps({"tz": "America/Los_Angeles", "place": "洛杉矶"}, ensure_ascii=False), encoding="utf-8"
    )
    uc_la = load_user_clock(clock_file)
    assert uc_la is not None
    ts_jul = datetime.datetime(2026, 7, 1, 12, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()
    ts_dec = datetime.datetime(2026, 12, 1, 12, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()
    assert user_local_stamp(ts_jul, uc_la).endswith("PDT UTC-07:00")
    assert user_local_stamp(ts_dec, uc_la).endswith("PST UTC-08:00")


def test_load_user_clock_expiry_returns_home_tz_and_home_place(tmp_path: Path) -> None:
    clock_file = tmp_path / "user_clock.json"
    clock_file.write_text(
        json.dumps(
            {
                "tz": "Asia/Kolkata",
                "place": "新德里",
                "home_tz": "Asia/Shanghai",
                "home_place": "北京",
                "expires_at": "2026-11-10T06:00:00+08:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    after_exp = datetime.datetime(2026, 11, 10, 6, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    uc_exp = load_user_clock(clock_file, now_epoch=after_exp)
    assert uc_exp is not None
    assert uc_exp["expired"] is True
    assert uc_exp["tz"] == "Asia/Shanghai"
    assert uc_exp["place"] == "北京"

    # Default home_tz / home_place when omitted
    clock_file.write_text(
        json.dumps({"tz": "Europe/London", "place": "伦敦", "expires_at": "2026-11-10 06:00"}, ensure_ascii=False),
        encoding="utf-8",
    )
    uc_def = load_user_clock(clock_file, now_epoch=after_exp)
    assert uc_def is not None
    assert uc_def["tz"] == DEFAULT_HOME_TZ
    assert uc_def["place"] == DEFAULT_HOME_PLACE


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (-5, "<1m"),
        (0, "<1m"),
        (59.9, "<1m"),
        (60, "1m"),
        (31 * 60, "31m"),
        (3600, "1h"),
        (2 * 3600 + 5 * 60, "2h05m"),
        (86400, "1d"),
        (27 * 3600, "1d3h"),
    ],
)
def test_humanize_gap(seconds: float, expected: str) -> None:
    assert humanize_gap(seconds) == expected

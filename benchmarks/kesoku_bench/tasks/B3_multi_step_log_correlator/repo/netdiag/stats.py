"""Summary statistics of a saved packet dump (offline, no network)."""

from __future__ import annotations

from pathlib import Path

from .dump import Packet, read_dump

STAT_KEYS = ("packets", "delivered", "dropped", "retransmits", "bytes", "loss_pct", "mean_len")


def summarise(packets: list[Packet]) -> dict[str, int | float]:
    """Compute summary statistics for a list of packets.

    Args:
        packets: Parsed packet records.

    Returns:
        A dict with the keys of :data:`STAT_KEYS`: packet counts, delivered
        payload bytes, loss percentage (2 decimals) and mean delivered length
        (1 decimal).
    """
    delivered = [p for p in packets if p.status != "drop"]
    total_bytes = sum(p.length for p in delivered)
    return {
        "packets": len(packets),
        "delivered": len(delivered),
        "dropped": len(packets) - len(delivered),
        "retransmits": sum(p.status == "rtx" for p in packets),
        "bytes": total_bytes,
        "loss_pct": round(100.0 * (len(packets) - len(delivered)) / len(packets), 2) if packets else 0.0,
        "mean_len": round(total_bytes / len(delivered), 1) if delivered else 0.0,
    }


def compute_stats(path: str | Path) -> dict[str, int | float]:
    """Read a dump file and return its summary statistics."""
    return summarise(read_dump(path))


def format_stats(stats: dict[str, int | float]) -> str:
    """Render statistics as ``key: value`` lines."""
    return "\n".join(f"{key}: {stats[key]}" for key in STAT_KEYS)

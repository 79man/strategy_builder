from datetime import timedelta
from typing import Dict

_INTERVAL_MAP: Dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}


def is_valid_interval(interval: str) -> bool:
    """Validate that the interval is supported"""
    if interval in _INTERVAL_MAP:
        return True

    return False


def validate_interval(interval: str) -> None:
    """Validate that the interval is supported"""
    if interval not in _INTERVAL_MAP:
        raise ValueError(f"Unsupported interval: {interval}")


def get_interval_timedelta(interval: str) -> timedelta:
    """Get timedelta for a given interval string"""
    validate_interval(interval)
    return _INTERVAL_MAP[interval]

def get_interval_freq(interval_str: str):
    validate_interval(interval_str)
    return (_INTERVAL_MAP[interval_str]).total_seconds()


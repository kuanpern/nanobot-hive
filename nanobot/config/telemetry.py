"""Telemetry configuration helpers — reads env vars that control observability."""
from __future__ import annotations

import os


def is_telemetry_enabled() -> bool:
    return os.getenv("NANOBOT_ENABLE_TELEMETRY", "0") in {"1", "true", "yes"}


def get_otel_exporter() -> str | None:
    return os.getenv("NANOBOT_OTEL_EXPORTER") or None


def get_otel_endpoint() -> str:
    return os.getenv("NANOBOT_OTEL_ENDPOINT", "http://localhost:4317")


def get_metrics_port() -> int:
    try:
        return int(os.getenv("NANOBOT_METRICS_PORT", "8000"))
    except ValueError:
        return 8000

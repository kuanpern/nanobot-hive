"""OpenTelemetry tracing and Prometheus metrics for nanobot.

Enabled with NANOBOT_ENABLE_TELEMETRY=1 (or "true"/"yes").
All public symbols are no-ops when telemetry is disabled or required packages
are not installed — callers never need to guard with if-checks.

Environment variables:
    NANOBOT_ENABLE_TELEMETRY    Master switch ("1"/"true"/"yes")
    NANOBOT_OTEL_EXPORTER       Exporter type: "otlp" (default) or "" (disable)
    NANOBOT_OTEL_ENDPOINT       OTLP collector URL (default http://localhost:4317)
    NANOBOT_METRICS_PORT        Prometheus /metrics HTTP port (default 8000)
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Feature flag — evaluated once at import time so the hot path is a bool check
# ---------------------------------------------------------------------------
_ENABLED: bool = os.getenv("NANOBOT_ENABLE_TELEMETRY", "0") in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# Lazy-initialized singletons
# ---------------------------------------------------------------------------
_tracer_provider: Any = None
_tracer: Any = None
_prom_registry: Any = None
_metrics: dict[str, Any] = {}


def _init_otel() -> bool:
    """Lazy-initialize OTel TracerProvider. Returns True on success."""
    global _tracer_provider, _tracer
    if _tracer is not None:
        return True
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({
            "service.name": "nanobot",
            "deployment.environment": os.getenv("ENV", "development"),
        })
        _tracer_provider = TracerProvider(resource=resource)

        exporter_type = os.getenv("NANOBOT_OTEL_EXPORTER", "otlp")
        if exporter_type:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                endpoint = os.getenv("NANOBOT_OTEL_ENDPOINT", "http://localhost:4317")
                span_exporter = OTLPSpanExporter(endpoint=endpoint)
                _tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
                logger.info("OTel tracing enabled", endpoint=endpoint)
            except Exception as exc:
                logger.warning("OTel exporter init failed; spans will not be exported", error=str(exc))

        _tracer = _tracer_provider.get_tracer("nanobot")
        return True
    except ImportError:
        logger.warning("opentelemetry-sdk not installed; OTel tracing disabled. "
                       "Install with: pip install 'nanobot-ai[telemetry]'")
        return False
    except Exception as exc:
        logger.warning("OTel initialization failed", error=str(exc))
        return False


def _init_prometheus() -> bool:
    """Lazy-initialize Prometheus registry and metrics. Returns True on success."""
    global _prom_registry
    if _prom_registry is not None:
        return True
    try:
        from prometheus_client import CollectorRegistry, Counter, Histogram

        _prom_registry = CollectorRegistry()
        _metrics["nanobot_requests_total"] = Counter(
            name="nanobot_requests_total",
            documentation="Total inbound messages received by the agent",
            registry=_prom_registry,
            labelnames=["channel"],
        )
        _metrics["nanobot_agent_iterations_total"] = Counter(
            name="nanobot_agent_iterations_total",
            documentation="Agent loop iterations completed, labelled by outcome",
            registry=_prom_registry,
            labelnames=["outcome"],
        )
        _metrics["nanobot_token_usage"] = Histogram(
            name="nanobot_token_usage",
            documentation="Token usage (prompt + completion) per LLM call",
            buckets=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
            registry=_prom_registry,
        )
        _metrics["nanobot_web_fetch_bytes"] = Histogram(
            name="nanobot_web_fetch_bytes",
            documentation="Response size in bytes per web_fetch call",
            buckets=[1024, 4096, 16384, 65536, 262144, 1048576],
            registry=_prom_registry,
        )
        _metrics["nanobot_heartbeat_ticks_total"] = Counter(
            name="nanobot_heartbeat_ticks_total",
            documentation="Heartbeat tick outcomes",
            registry=_prom_registry,
            labelnames=["outcome"],
        )
        return True
    except ImportError:
        logger.warning("prometheus-client not installed; metrics disabled. "
                       "Install with: pip install 'nanobot-ai[telemetry]'")
        return False
    except Exception as exc:
        logger.warning("Prometheus initialization failed", error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@asynccontextmanager
async def trace(span_name: str) -> AsyncGenerator[Any, None]:
    """Async context manager that wraps a block in an OTel span.

    Usage::

        async with trace("process_message"):
            await do_work()

    Is a no-op when telemetry is disabled or OTel packages are absent.
    Never raises — exporter failures are caught and logged.
    """
    if not _ENABLED:
        yield None
        return

    if _tracer is None and not _init_otel():
        yield None
        return

    try:
        with _tracer.start_as_current_span(span_name) as span:
            span.set_attribute("nanobot.operation", span_name)
            yield span
    except Exception as exc:
        logger.debug("OTel span error (non-fatal)", span=span_name, error=str(exc))
        yield None


def record_metric(name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
    """Increment a Prometheus counter or observe a histogram.

    ``name`` must be one of the registered metric names (see module-level
    ``_metrics`` dict).  Unknown names are silently ignored.
    Is a no-op when telemetry is disabled or prometheus-client is absent.
    """
    if not _ENABLED:
        return

    if _prom_registry is None and not _init_prometheus():
        return

    metric = _metrics.get(name)
    if metric is None:
        return

    try:
        from prometheus_client import Counter, Histogram

        if isinstance(metric, Counter):
            metric.labels(**(labels or {})).inc(value)
        elif isinstance(metric, Histogram):
            metric.observe(value)
    except Exception as exc:
        logger.debug("record_metric failed (non-fatal)", metric=name, error=str(exc))


async def start_telemetry() -> list[asyncio.Task]:
    """Initialize telemetry subsystems and start the Prometheus HTTP server.

    Call once at agent startup.  Returns a list of asyncio.Task objects that
    should be passed to :func:`stop_telemetry` during shutdown.
    """
    if not _ENABLED:
        return []

    _init_otel()

    if not _init_prometheus():
        return []

    port = int(os.getenv("NANOBOT_METRICS_PORT", "8000"))

    def _serve_metrics() -> None:
        from prometheus_client import start_http_server
        start_http_server(port, registry=_prom_registry)
        logger.info("Prometheus /metrics endpoint started", port=port)

    loop = asyncio.get_event_loop()
    # run_in_executor returns a Future; wrap it so callers get a consistent type
    fut = loop.run_in_executor(None, _serve_metrics)
    task = asyncio.ensure_future(fut)
    return [task]


async def stop_telemetry(tasks: list[asyncio.Task]) -> None:
    """Cancel background telemetry tasks and flush the OTel exporter."""
    for task in tasks:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
        except Exception as exc:
            logger.debug("OTel provider shutdown error (non-fatal)", error=str(exc))


__all__ = ["trace", "record_metric", "start_telemetry", "stop_telemetry"]

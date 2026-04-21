"""Redis-backed async message bus using LIST queues (RPUSH / BLPOP)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from datetime import datetime
from typing import Any

import structlog
logger = structlog.get_logger()

try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore[assignment]

if aioredis is None:  # pragma: no cover
    raise RuntimeError(
        "Redis support is not installed. "
        "Install it with: pip install 'nanobot-ai[redis]'"
    )

from nanobot.agent.events import InboundMessage, OutboundMessage

_INBOUND_KEY = "nanobot:inbound"
_OUTBOUND_KEY = "nanobot:outbound"
_BLPOP_TIMEOUT = 1.0  # seconds; enables periodic closed-check


def _encode(msg: Any) -> str:
    d = dataclasses.asdict(msg)
    d["_type"] = type(msg).__name__
    # datetime → ISO string
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return json.dumps(d, ensure_ascii=False)


def _decode_inbound(raw: str) -> InboundMessage:
    d = json.loads(raw)
    d.pop("_type", None)
    ts = d.get("timestamp")
    if isinstance(ts, str):
        d["timestamp"] = datetime.fromisoformat(ts)
    return InboundMessage(**d)


def _decode_outbound(raw: str) -> OutboundMessage:
    d = json.loads(raw)
    d.pop("_type", None)
    return OutboundMessage(**d)


class RedisBus:
    """
    Async message bus backed by Redis LIST queues.

    Two keys are used:
        ``nanobot:inbound``   – inbound messages from channels to the agent.
        ``nanobot:outbound``  – outbound messages from the agent to channels.

    Implements the same four coroutines as the in-process ``MessageBus``.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        inbound_key: str = _INBOUND_KEY,
        outbound_key: str = _OUTBOUND_KEY,
        reconnect_interval: float = 5.0,
        max_backoff: float = 30.0,
    ) -> None:
        self._redis_url = redis_url
        self._inbound_key = inbound_key
        self._outbound_key = outbound_key
        self._reconnect_interval = reconnect_interval
        self._max_backoff = max_backoff
        self._redis: aioredis.Redis | None = None
        self._closed = False
        self._reconnect_task: asyncio.Task | None = None

    async def _ensure_connected(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_keepalive=True,
            )
            await self._redis.ping()
            logger.debug("Connected to Redis at {}", self._redis_url)
        return self._redis

    async def _reconnect_loop(self) -> None:
        backoff = self._reconnect_interval
        while not self._closed:
            await asyncio.sleep(backoff)
            try:
                if self._redis is not None:
                    await self._redis.aclose()
                    self._redis = None
                await self._ensure_connected()
                backoff = self._reconnect_interval
                logger.info("Reconnected to Redis")
            except Exception as exc:  # pragma: no cover
                logger.warning("Redis reconnection failed: {}", exc)
                backoff = min(backoff * 2, self._max_backoff)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self._ensure_connected()
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def stop(self) -> None:
        self._closed = True
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        await self.close()

    async def close(self) -> None:
        self._closed = True
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.debug("Closed Redis connection")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def publish_inbound(self, msg: InboundMessage) -> None:
        if self._closed:
            raise RuntimeError("RedisBus is closed")
        r = await self._ensure_connected()
        await r.rpush(self._inbound_key, _encode(msg))

    async def consume_inbound(self) -> InboundMessage:
        if self._closed:
            raise RuntimeError("RedisBus is closed")
        while not self._closed:
            try:
                r = await self._ensure_connected()
                result = await r.blpop(self._inbound_key, timeout=_BLPOP_TIMEOUT)
            except Exception as exc:  # pragma: no cover
                logger.warning("Redis BLPOP error (inbound): {}", exc)
                self._redis = None
                await asyncio.sleep(self._reconnect_interval)
                continue
            if result is None:
                continue
            _key, payload = result
            return _decode_inbound(payload)
        raise RuntimeError("RedisBus is closed")

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        if self._closed:
            raise RuntimeError("RedisBus is closed")
        r = await self._ensure_connected()
        await r.rpush(self._outbound_key, _encode(msg))

    async def consume_outbound(self) -> OutboundMessage:
        if self._closed:
            raise RuntimeError("RedisBus is closed")
        while not self._closed:
            try:
                r = await self._ensure_connected()
                result = await r.blpop(self._outbound_key, timeout=_BLPOP_TIMEOUT)
            except Exception as exc:  # pragma: no cover
                logger.warning("Redis BLPOP error (outbound): {}", exc)
                self._redis = None
                await asyncio.sleep(self._reconnect_interval)
                continue
            if result is None:
                continue
            _key, payload = result
            return _decode_outbound(payload)
        raise RuntimeError("RedisBus is closed")

    @property
    def inbound_size(self) -> int:
        raise NotImplementedError("Use async llen() directly for Redis queue sizes")

    @property
    def outbound_size(self) -> int:
        raise NotImplementedError("Use async llen() directly for Redis queue sizes")

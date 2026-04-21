"""Message bus module for decoupled channel-agent communication."""

from __future__ import annotations

import os

from nanobot.agent.events import InboundMessage, OutboundMessage
from nanobot.optional.bus.asyncio_queue import MessageBus

try:
    from nanobot.optional.bus.redis_pubsub import RedisBus
except Exception:  # pragma: no cover
    RedisBus = None  # type: ignore[assignment,misc]


def _get_redis_url() -> str:
    return os.getenv("NANOBOT_REDIS_URL", "redis://localhost:6379/0")


def get_bus() -> MessageBus:
    """Return a bus instance selected by the ``NANOBOT_BUS`` env var.

    Default backend is ``redis``.  Set ``NANOBOT_BUS=asyncio`` to use the
    in-process asyncio queue instead.
    """
    backend = os.getenv("NANOBOT_BUS", "redis").strip().lower()
    if backend == "redis":
        if RedisBus is None:  # pragma: no cover
            raise RuntimeError(
                "Redis backend requested but 'redis' package is not installed. "
                "Install it with: pip install 'nanobot-ai[redis]'"
            )
        return RedisBus(redis_url=_get_redis_url())  # type: ignore[return-value]
    return MessageBus()


__all__ = ["MessageBus", "RedisBus", "InboundMessage", "OutboundMessage", "get_bus"]

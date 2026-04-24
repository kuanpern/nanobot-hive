"""Message bus module for decoupled channel-agent communication."""

from __future__ import annotations

import os

from nanobot.agent.events import InboundMessage, OutboundMessage
from .redis_pubsub import RedisBus as MessageBus

try:
    from .asyncio_queue import MessageBus as AsyncioMessageBus
except Exception:  # pragma: no cover
    AsyncioMessageBus = None  # type: ignore[assignment,misc]

RedisBus = MessageBus


def _get_redis_url() -> str:
    return os.getenv("NANOBOT_REDIS_URL", "redis://localhost:6379/0")


def get_bus() -> MessageBus:
    """Return a bus instance selected by the ``NANOBOT_BUS`` env var.

    Default backend is ``redis``.  Set ``NANOBOT_BUS=asyncio`` to use the
    in-process asyncio queue instead.
    """
    backend = os.getenv("NANOBOT_BUS", "redis").strip().lower()
    if backend == "asyncio":
        if AsyncioMessageBus is None:  # pragma: no cover
            raise RuntimeError("asyncio bus backend not available")
        return AsyncioMessageBus()  # type: ignore[return-value]
    return MessageBus(redis_url=_get_redis_url())


__all__ = ["MessageBus", "RedisBus", "AsyncioMessageBus", "InboundMessage", "OutboundMessage", "get_bus"]

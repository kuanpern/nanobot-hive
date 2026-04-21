"""Message bus module for decoupled channel-agent communication."""

from nanobot_hive.core.events import InboundMessage, OutboundMessage
from nanobot_hive.optional.bus.asyncio_queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]

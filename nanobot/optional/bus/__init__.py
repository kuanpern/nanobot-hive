"""Message bus module for decoupled channel-agent communication."""

from nanobot.core.events import InboundMessage, OutboundMessage
from nanobot.optional.bus.asyncio_queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]

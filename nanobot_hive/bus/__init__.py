"""Message bus module for decoupled channel-agent communication."""

from nanobot_hive.bus.events import InboundMessage, OutboundMessage
from nanobot_hive.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]

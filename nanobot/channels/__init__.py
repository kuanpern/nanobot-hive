"""
Chat channels module.
Use ChannelManager to orchestrate communication.
"""

from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
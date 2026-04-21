"""Chat channels module with plugin architecture."""

from nanobot_hive.channels.base import BaseChannel
from nanobot_hive.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]

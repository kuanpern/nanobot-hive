"""Slash command routing and built-in handlers."""

from nanobot_hive.command.builtin import register_builtin_commands
from nanobot_hive.command.router import CommandContext, CommandRouter

__all__ = ["CommandContext", "CommandRouter", "register_builtin_commands"]

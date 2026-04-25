"""Spawn tool for creating background subagents."""

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnToolSchema(BaseModel):
    """Input schema for SpawnTool."""
    task: str = Field(description="The task for the subagent to complete")
    label: str | None = Field(default=None, description="Optional short label for the task (for display)")


class SpawnTool(BaseTool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel: ContextVar[str] = ContextVar("spawn_origin_channel", default="cli")
        self._origin_chat_id: ContextVar[str] = ContextVar("spawn_origin_chat_id", default="direct")
        self._session_key: ContextVar[str] = ContextVar("spawn_session_key", default="cli:direct")
        super().__init__()

    def set_context(self, channel: str, chat_id: str, effective_key: str | None = None) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel.set(channel)
        self._origin_chat_id.set(chat_id)
        self._session_key.set(effective_key or f"{channel}:{chat_id}")

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done. "
            "For deliverables or existing projects, inspect the workspace first "
            "and use a dedicated subdirectory when helpful."
        )
    args_schema: type[BaseModel] = SpawnToolSchema

    async def _arun(self, task: str, label: str | None = None) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel.get(),
            origin_chat_id=self._origin_chat_id.get(),
            session_key=self._session_key.get(),
        )

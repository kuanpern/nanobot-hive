"""Spawn tool for creating background subagents."""
import asyncio
from typing import TYPE_CHECKING, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Import the global context vars defined in your tools/base.py or engine
from nanobot.tools.base import channel_ctx, chat_id_ctx, session_key_ctx 

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager

class SpawnSchema(BaseModel):
    task: str = Field(description="The task for the subagent to complete")
    label: str | None = Field(default=None, description="Optional short label for the task")

class SpawnTool(BaseTool):
    """Tool to spawn a subagent for background task execution."""
    
    name: str = "spawn"
    description: str = (
        "Spawn a subagent to handle a task in the background. "
        "Use this for complex or time-consuming tasks."
    )
    args_schema: type[BaseModel] = SpawnSchema
    
    manager: "SubagentManager"

    def _run(self, task: str, label: str | None = None) -> str:
        return asyncio.run(self._arun(task, label))

    async def _arun(self, task: str, label: str | None = None) -> str:
        """Spawn a subagent using the current task context."""
        
        # Pull context directly from the global contextvars
        origin_channel = channel_ctx.get()
        origin_chat_id = chat_id_ctx.get()
        effective_key = session_key_ctx.get()
        
        return await self.manager.spawn(
            task=task,
            label=label,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            session_key=effective_key,
        )
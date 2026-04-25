# nanobot/tools/base.py
from contextvars import ContextVar
from langchain_core.tools import BaseTool

# Create global context variables for tools to consume
channel_ctx: ContextVar[str] = ContextVar("channel", default="cli")
chat_id_ctx: ContextVar[str] = ContextVar("chat_id", default="direct")
session_key_ctx: ContextVar[str] = ContextVar("session_key", default="cli:direct")

class BaseTool(BaseTool):
    # This automatically captures the context if you set it at the start of _dispatch
    def get_context(self):
        return channel_ctx.get(), chat_id_ctx.get()
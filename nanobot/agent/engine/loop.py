"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import structlog
logger = structlog.get_logger()

from nanobot.telemetry import record_metric, trace
from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory.consolidator import Consolidator
from nanobot.agent.memory.dream import Dream
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.subagent import SubagentManager
from nanobot.tools.base import channel_ctx, chat_id_ctx, session_key_ctx
from nanobot.tools.cron import CronTool
from nanobot.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.tools.message import MessageTool
from nanobot.tools.notebook import NotebookEditTool
from nanobot.tools.registry import ToolRegistry
from nanobot.tools.search import GlobTool, GrepTool
from nanobot.tools.shell import ExecTool
from nanobot.tools.self import MyTool
from nanobot.tools.spawn import SpawnTool
from nanobot.tools.web import WebFetchTool, WebSearchTool
from nanobot.agent.events import InboundMessage, OutboundMessage
from nanobot.core.bus import MessageBus
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.core.config.schema import AgentDefaults
from nanobot.providers.base import LLMProvider
from nanobot.utils.document import extract_documents
from nanobot.utils.helpers import image_placeholder_text
from nanobot.utils.helpers import truncate_text as truncate_text_fn
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE
from nanobot.agent.engine.checkpoint import get_checkpointer
from nanobot.agent.engine.graph import create_agent_graph
from nanobot.agent.engine.preprocessor import MessagePreProcessor


if TYPE_CHECKING:
    from nanobot.core.config.schema import ChannelsConfig, ExecToolConfig, ToolsConfig, WebToolsConfig
    from nanobot.cron import CronService


UNIFIED_SESSION_KEY = "unified:default"
LAST_N_MESSAGES = 100

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage


async def graph_observer(event: dict):
    """Central listener for every node execution."""
    kind = event["event"]
    if kind == "on_node_end":
        node = event["node"]
        record_metric("nanobot_agent_iterations_total", labels={"outcome": node})
        logger.debug(f"Node finished: {node}")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: (x + y)[-LAST_N_MESSAGES:]]


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_USER_TURN_KEY = "pending_user_turn"

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        max_tool_result_chars: int | None = None,
        provider_retry_mode: str = "standard",
        web_config: WebToolsConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        session_ttl_minutes: int = 0,
        unified_session: bool = False,
        disabled_skills: list[str] | None = None,
        tools_config: ToolsConfig | None = None,
    ):
        from nanobot.core.config.schema import ExecToolConfig, ToolsConfig, WebToolsConfig

        _tc = tools_config or ToolsConfig()
        defaults = AgentDefaults()
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.preprocessor = MessagePreProcessor(self.workspace, timezone=timezone)
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        self.context_window_tokens = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        self.context_block_limit = context_block_limit
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        self.web_config = web_config or WebToolsConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()

        self.context = ContextBuilder(workspace, timezone=timezone, disabled_skills=disabled_skills)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_config=self.web_config,
            max_tool_result_chars=self.max_tool_result_chars,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            disabled_skills=disabled_skills,
        )
        self._unified_session = unified_session
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stacks: dict[str, AsyncExitStack] = {}
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Per-session pending queues for mid-turn message injection.
        # When a session has an active task, new messages for that session
        # are routed here instead of creating a new task.
        self._pending_queues: dict[str, asyncio.Queue] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.consolidator = Consolidator(
            store=self.context.memory,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=self.context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
        )
        self.dream = Dream(
            store=self.context.memory,
            provider=provider,
            model=self.model,
        )
        self._register_default_tools()

        if _tc.my.enable:
            self.tools.register(MyTool(loop=self, modify_allowed=_tc.my.allow_set))
        self._current_iteration: int = 0
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

        self.checkpointer = get_checkpointer(self.workspace)

        # Build and compile the graph
        builder = create_agent_graph(self.provider, self.tools)
        self.app = builder.compile(checkpointer=self.checkpointer)


    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = (
            self.workspace if (self.restrict_to_workspace or self.exec_config.sandbox) else None
        )
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(
            ReadFileTool(
                workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read
            )
        )
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        for cls in (GlobTool, GrepTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(NotebookEditTool(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    sandbox=self.exec_config.sandbox,
                    path_append=self.exec_config.path_append,
                    allowed_env_keys=self.exec_config.allowed_env_keys,
                )
            )
        if self.web_config.enable:
            self.tools.register(
                WebSearchTool(config=self.web_config.search, proxy=self.web_config.proxy)
            )
            self.tools.register(WebFetchTool(proxy=self.web_config.proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.tools.mcp import connect_mcp_servers

        try:
            self._mcp_stacks = await connect_mcp_servers(self._mcp_servers, self.tools)
            if self._mcp_stacks:
                self._mcp_connected = True
            else:
                logger.warning("No MCP servers connected successfully (will retry next message)")
        except asyncio.CancelledError:
            logger.warning("MCP connection cancelled (will retry next message)")
            self._mcp_stacks.clear()
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            self._mcp_stacks.clear()
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        # Compute the effective session key (accounts for unified sessions)
        # so that subagent results route to the correct pending queue.
        effective_key = UNIFIED_SESSION_KEY if self._unified_session else f"{channel}:{chat_id}"
        for name in ("message", "spawn", "cron", "my"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    if name == "spawn":
                        tool.set_context(channel, chat_id, effective_key=effective_key)
                    else:
                        tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think

        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hints with smart abbreviation."""
        from nanobot.utils.tool_hints import format_tool_hints

        return format_tool_hints(tool_calls)

    def _effective_session_key(self, msg: InboundMessage) -> str:
        """Return the session key used for task routing and mid-turn injections."""
        if self._unified_session and not msg.session_key_override:
            return UNIFIED_SESSION_KEY
        return msg.session_key

    async def run(self) -> None:
        """Run the agent, listening to the bus and dispatching to the graph."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started — listening for inbound messages")

        while self._running:
            try:
                # Consume messages from the bus
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                
                # Check for priority commands (stop, restart) before dispatching to graph
                if self.commands.is_priority(msg.content.strip()):
                    ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=msg.content, loop=self)
                    result = await self.commands.dispatch_priority(ctx)
                    if result:
                        await self.bus.publish_outbound(result)
                    continue

                # Dispatch message to LangGraph
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Bus consumption error: {e}")


    async def _dispatch(self, msg: InboundMessage):
        effective_key = self._effective_session_key(msg)
        
        # 1. Scope Tool Context
        token_channel = channel_ctx.set(msg.channel)
        token_chat = chat_id_ctx.set(msg.chat_id)
        token_key = session_key_ctx.set(effective_key)
        
        try:
            # 2. Prepare Inputs
            # The preprocessor now handles context tagging and document extraction
            inputs = {"messages": [await self.preprocessor.process(msg)]}
            config = {"configurable": {"thread_id": effective_key}}
            
            # 3. Invoke Graph (No re-compilation!)
            # LangGraph handles the message injection from the state 
            # and manages persistence via the checkpointer defined at compile time
            final_state = await self.app.ainvoke(inputs, config=config)
            
            # 4. Response
            final_content = final_state["messages"][-1].content
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, 
                chat_id=msg.chat_id, 
                content=str(final_content)
            ))
            
        finally:
            # 5. ContextVar cleanup
            channel_ctx.reset(token_channel)
            chat_id_ctx.reset(token_chat)
            session_key_ctx.reset(token_key)


    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        for name, stack in self._mcp_stacks.items():
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                logger.debug("MCP server '{}' cleanup error (can be ignored)", name)
        self._mcp_stacks.clear()

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        **kwargs: Any
    ) -> OutboundMessage:
        """Programmatically trigger the agent graph and return the final result."""
        
        # 1. Initialize State
        initial_state: AgentState = {
            "messages": [HumanMessage(content=content)],
            "iteration": 0,
            "subagent_status": {},
            "scratchpad": self._runtime_vars, # Reference to current loop variables
            "is_error": False,
            "last_error": None
        }
        
        # 2. Invoke Graph via checkpoint
        # thread_id persists the session in your sqlite DB
        config = {"configurable": {"thread_id": session_key}}
        
        # 3. Execution
        final_state = await self.app.ainvoke(initial_state, config=config)
        
        # 4. Extract Result safely
        # We access the last message, ensuring it's a content-bearing message
        final_msg = final_state["messages"][-1]
        final_content = getattr(final_msg, "content", "No response generated.")
        
        # Update our runtime vars from the final state to keep self._loop in sync
        self._runtime_vars.update(final_state.get("scratchpad", {}))
        
        return OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=str(final_content)
        )
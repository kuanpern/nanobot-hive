"""LangChain-backed LLM provider base class and shared utilities."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import string
from collections.abc import Awaitable, Callable
from typing import Any, TYPE_CHECKING

import json_repair
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_ALNUM = string.ascii_letters + string.digits


def _short_id() -> str:
    """9-char alphanumeric ID compatible with all providers."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def to_lc_messages(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """Convert OpenAI-format message dicts to LangChain BaseMessage objects."""
    result: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            result.append(SystemMessage(content=content or ""))

        elif role == "user":
            result.append(HumanMessage(content=content or ""))

        elif role == "assistant":
            # Build content list: thinking blocks first, then text
            thinking_blocks: list[dict[str, Any]] = msg.get("thinking_blocks") or []
            content_blocks: list[Any] = []
            for tb in thinking_blocks:
                if isinstance(tb, dict) and tb.get("type") == "thinking":
                    content_blocks.append(tb)
            if isinstance(content, str) and content:
                if content_blocks:
                    content_blocks.append({"type": "text", "text": content})
                else:
                    content_blocks = content
            elif isinstance(content, list):
                content_blocks.extend(content)

            # Normalize tool calls
            lc_tool_calls: list[dict[str, Any]] = []
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = json_repair.loads(args) if args else {}
                lc_tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "args": args if isinstance(args, dict) else {},
                    "type": "tool_call",
                })

            result.append(AIMessage(
                content=content_blocks or (content or ""),
                tool_calls=lc_tool_calls,
            ))

        elif role == "tool":
            # ToolMessage content can be str or list
            result.append(ToolMessage(
                content=content or "",
                tool_call_id=msg.get("tool_call_id", ""),
            ))

    return result


def from_lc_response(response: AIMessage | AIMessageChunk) -> LLMResponse:
    """Convert a LangChain AIMessage to an LLMResponse."""
    raw_content = response.content
    thinking_blocks: list[dict[str, Any]] | None = None
    text_parts: list[str] = []

    if isinstance(raw_content, list):
        tb_list: list[dict[str, Any]] = []
        for block in raw_content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "thinking":
                    tb_list.append(block)
                elif btype == "text":
                    text_parts.append(block.get("text", ""))
                # skip tool_use and other block types
        if tb_list:
            thinking_blocks = tb_list
        content: str | None = "".join(text_parts) or None
    elif isinstance(raw_content, str):
        content = raw_content or None
    else:
        content = None

    # Tool calls
    tool_calls: list[ToolCallRequest] = []
    for tc in response.tool_calls or []:
        args = tc.get("args", {})
        if isinstance(args, str):
            args = json_repair.loads(args) if args else {}
        tool_calls.append(ToolCallRequest(
            id=tc.get("id") or _short_id(),
            name=tc.get("name", ""),
            arguments=args if isinstance(args, dict) else {},
        ))

    # Usage
    usage: dict[str, int] = {}
    if response.usage_metadata:
        usage = {
            "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
            "completion_tokens": response.usage_metadata.get("output_tokens", 0),
            "total_tokens": response.usage_metadata.get("total_tokens", 0),
        }
        token_details = response.usage_metadata.get("input_token_details") or {}
        cache_read = token_details.get("cache_read") or 0
        if cache_read:
            usage["cached_tokens"] = int(cache_read)

    # Finish reason
    finish_reason = (response.response_metadata or {}).get("finish_reason") or "stop"
    if tool_calls and finish_reason not in ("error",):
        finish_reason = "tool_calls"

    # Provider-specific fields
    additional = response.additional_kwargs or {}
    reasoning_content = additional.get("reasoning_content")

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
        thinking_blocks=thinking_blocks,
    )


def handle_lc_error(e: Exception) -> LLMResponse:
    """Convert an exception from a LangChain provider call to an LLMResponse."""
    response = getattr(e, "response", None)
    headers = getattr(response, "headers", None)
    status_code = getattr(e, "status_code", None) or getattr(response, "status_code", None)
    payload = (
        getattr(e, "body", None)
        or getattr(e, "doc", None)
        or getattr(response, "text", None)
    )
    payload_text = payload if isinstance(payload, str) else (str(payload) if payload else "")
    msg = (
        f"Error: {payload_text.strip()[:500]}"
        if payload_text.strip()
        else f"Error calling LLM: {e}"
    )
    retry_after = LLMProvider._extract_retry_after_from_headers(headers)
    if retry_after is None:
        retry_after = LLMProvider._extract_retry_after(msg)

    error_type, error_code = LLMProvider._extract_error_type_code(payload)

    error_kind: str | None = None
    err_name = e.__class__.__name__.lower()
    if "timeout" in err_name:
        error_kind = "timeout"
    elif "connection" in err_name:
        error_kind = "connection"

    should_retry: bool | None = None
    if headers is not None:
        raw = headers.get("x-should-retry")
        if isinstance(raw, str):
            lower = raw.strip().lower()
            if lower == "true":
                should_retry = True
            elif lower == "false":
                should_retry = False

    return LLMResponse(
        content=msg,
        finish_reason="error",
        retry_after=retry_after,
        error_status_code=int(status_code) if status_code is not None else None,
        error_kind=error_kind,
        error_type=error_type,
        error_code=error_code,
        error_retry_after_s=retry_after,
        error_should_retry=should_retry,
    )


class LangChainProvider(LLMProvider):
    """LLM provider backed by a LangChain BaseChatModel.

    Subclasses override ``_get_model_for_call()`` to return a model configured
    for the specific call parameters (temperature, max_tokens, etc.).
    """

    def __init__(
        self,
        model: BaseChatModel,
        default_model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        super().__init__(api_key, api_base)
        self._model = model
        self._default_model = default_model

    def get_default_model(self) -> str:
        return self._default_model

    def _get_model_for_call(
        self,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
    ) -> BaseChatModel:
        """Return a configured model for this call. Override in subclasses."""
        return self._model

    def _preprocess_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Hook for subclasses to transform messages/tools before the LLM call.

        Called after ``_sanitize_empty_content`` and before ``to_lc_messages``.
        """
        return messages, tools

    def _bind_tools(
        self,
        model: BaseChatModel,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None,
    ) -> BaseChatModel:
        """Bind tools with normalized tool_choice for LangChain."""
        # Normalize OpenAI-style tool_choice to LangChain format
        lc_choice: str | None = None
        if isinstance(tool_choice, str) and tool_choice not in ("auto", "none", None):
            lc_choice = tool_choice  # "required" or custom string
        elif isinstance(tool_choice, dict):
            name = tool_choice.get("function", {}).get("name")
            if name:
                lc_choice = name

        bind_kwargs: dict[str, Any] = {}
        if lc_choice:
            bind_kwargs["tool_choice"] = lc_choice
        return model.bind_tools(tools, **bind_kwargs)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        try:
            messages, tools = self._preprocess_messages(
                self._sanitize_empty_content(messages), tools,
            )
            lc_messages = to_lc_messages(messages)
            m = self._get_model_for_call(model, max_tokens, temperature, reasoning_effort)
            if tools and tool_choice != "none":
                m = self._bind_tools(m, tools, tool_choice)
            response = await m.ainvoke(lc_messages)
            return from_lc_response(response)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return handle_lc_error(e)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        idle_timeout_s = int(os.environ.get("NANOBOT_STREAM_IDLE_TIMEOUT_S", "90"))
        try:
            messages, tools = self._preprocess_messages(
                self._sanitize_empty_content(messages), tools,
            )
            lc_messages = to_lc_messages(messages)
            m = self._get_model_for_call(model, max_tokens, temperature, reasoning_effort)
            if tools and tool_choice != "none":
                m = self._bind_tools(m, tools, tool_choice)

            accumulated: AIMessageChunk | None = None
            stream_iter = m.astream(lc_messages).__aiter__()
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        stream_iter.__anext__(), timeout=idle_timeout_s,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    return LLMResponse(
                        content=(
                            f"Error calling LLM: stream stalled for more than "
                            f"{idle_timeout_s} seconds"
                        ),
                        finish_reason="error",
                        error_kind="timeout",
                    )
                if not isinstance(chunk, AIMessageChunk):
                    continue
                accumulated = chunk if accumulated is None else accumulated + chunk
                if on_content_delta:
                    delta: str = ""
                    if isinstance(chunk.content, str):
                        delta = chunk.content
                    elif isinstance(chunk.content, list):
                        for block in chunk.content:
                            if isinstance(block, str):
                                delta += block
                            elif isinstance(block, dict) and block.get("type") == "text":
                                delta += block.get("text", "")
                    if delta:
                        await on_content_delta(delta)

            if accumulated is None:
                return LLMResponse(content=None, finish_reason="stop")
            return from_lc_response(accumulated)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return handle_lc_error(e)

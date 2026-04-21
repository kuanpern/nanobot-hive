"""Anthropic provider backed by LangChain's ChatAnthropic."""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic

from nanobot.optional.llm.langchain_provider import LangChainProvider


class AnthropicProvider(LangChainProvider):
    """LLM provider for Claude models, using LangChain's ChatAnthropic.

    Supports extended thinking (reasoning_effort), prompt caching, and
    tool use — all via LangChain's unified interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        client_kwargs: dict[str, Any] = {"max_retries": 0}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["base_url"] = api_base
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers

        model = ChatAnthropic(
            model=default_model,
            **client_kwargs,
        )
        super().__init__(model, default_model, api_key, api_base)

    @staticmethod
    def _strip_prefix(model_name: str) -> str:
        if model_name.startswith("anthropic/"):
            return model_name[len("anthropic/"):]
        return model_name

    def _get_model_for_call(
        self,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
    ) -> ChatAnthropic:
        model_name = self._strip_prefix(model or self._default_model)
        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max(1, max_tokens),
        }

        if reasoning_effort == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["temperature"] = 1.0
        elif reasoning_effort:
            budget_map = {"low": 1024, "medium": 4096, "high": max(8192, max_tokens)}
            budget = budget_map.get(reasoning_effort.lower(), 4096)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["max_tokens"] = max(max_tokens, budget + 4096)
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = temperature

        return self._model.bind(**kwargs)

    def _preprocess_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Inject prompt-caching markers into messages and tools."""
        return self._apply_cache_control(messages, tools)

    @classmethod
    def _apply_cache_control(
        cls,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Inject cache_control markers for Anthropic prompt caching."""
        marker: dict[str, Any] = {"type": "ephemeral"}
        new_messages = list(messages)

        def _mark(msg: dict[str, Any]) -> dict[str, Any]:
            content = msg.get("content")
            if isinstance(content, str):
                return {**msg, "content": [
                    {"type": "text", "text": content, "cache_control": marker},
                ]}
            if isinstance(content, list) and content:
                nc = list(content)
                nc[-1] = {**nc[-1], "cache_control": marker}
                return {**msg, "content": nc}
            return msg

        # Cache the system prompt
        if new_messages and new_messages[0].get("role") == "system":
            new_messages[0] = _mark(new_messages[0])

        # Cache the penultimate message (typically a large context turn)
        if len(new_messages) >= 3:
            new_messages[-2] = _mark(new_messages[-2])

        # Cache tool definitions
        new_tools = tools
        if tools:
            new_tools = list(tools)
            for idx in cls._tool_cache_marker_indices(new_tools):
                new_tools[idx] = {**new_tools[idx], "cache_control": marker}

        return new_messages, new_tools

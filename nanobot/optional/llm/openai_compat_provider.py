"""OpenAI-compatible provider for all non-Anthropic LLM APIs, backed by LangChain."""

from __future__ import annotations

import os
import uuid
from typing import Any, TYPE_CHECKING

from langchain_openai import ChatOpenAI

from nanobot.optional.llm.langchain_provider import LangChainProvider

if TYPE_CHECKING:
    from nanobot.optional.llm.registry import ProviderSpec

_DEFAULT_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/HKUDS/nanobot",
    "X-OpenRouter-Title": "nanobot",
    "X-OpenRouter-Categories": "cli-agent,personal-agent",
}


def _uses_openrouter_attribution(spec: ProviderSpec | None, api_base: str | None) -> bool:
    if spec and spec.name == "openrouter":
        return True
    return bool(api_base and "openrouter" in api_base.lower())


class OpenAICompatProvider(LangChainProvider):
    """Unified provider for all OpenAI-compatible APIs, backed by LangChain's ChatOpenAI.

    A single ProviderSpec drives provider-specific behaviour: custom base URLs,
    prompt-caching, model overrides, provider-specific thinking params, etc.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-4o",
        extra_headers: dict[str, str] | None = None,
        spec: ProviderSpec | None = None,
    ) -> None:
        self._spec = spec

        if api_key and spec and spec.env_key:
            self._setup_env(api_key, api_base, spec)

        effective_base = api_base or (spec.default_api_base if spec else None) or None
        headers: dict[str, str] = {"x-session-affinity": uuid.uuid4().hex}
        if _uses_openrouter_attribution(spec, effective_base):
            headers.update(_DEFAULT_OPENROUTER_HEADERS)
        if extra_headers:
            headers.update(extra_headers)

        model = ChatOpenAI(
            model=default_model,
            api_key=api_key or "no-key",
            base_url=effective_base,
            default_headers=headers,
            max_retries=0,
        )
        super().__init__(model, default_model, api_key, api_base)

    @staticmethod
    def _setup_env(api_key: str, api_base: str | None, spec: ProviderSpec) -> None:
        if not spec.env_key:
            return
        if spec.is_gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key).replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)

    @staticmethod
    def _supports_temperature(model_name: str, reasoning_effort: str | None) -> bool:
        if reasoning_effort and reasoning_effort.lower() != "none":
            return False
        return not any(t in model_name.lower() for t in ("gpt-5", "o1", "o3", "o4"))

    def _get_model_for_call(
        self,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
    ) -> ChatOpenAI:
        spec = self._spec
        model_name = model or self._default_model

        if spec and spec.strip_model_prefix:
            model_name = model_name.split("/")[-1]

        kwargs: dict[str, Any] = {"model": model_name}

        if self._supports_temperature(model_name, reasoning_effort):
            kwargs["temperature"] = temperature

        if spec and getattr(spec, "supports_max_completion_tokens", False):
            kwargs["max_completion_tokens"] = max(1, max_tokens)
        else:
            kwargs["max_tokens"] = max(1, max_tokens)

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        # Per-model overrides from ProviderSpec
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_name.lower():
                    kwargs.update(overrides)
                    break

        # Provider-specific thinking parameters
        if spec and reasoning_effort is not None:
            thinking_on = reasoning_effort.lower() != "minimal"
            extra: dict[str, Any] | None = None
            if spec.name == "dashscope":
                extra = {"enable_thinking": thinking_on}
            elif spec.name in (
                "volcengine", "volcengine_coding_plan",
                "byteplus", "byteplus_coding_plan",
            ):
                extra = {"thinking": {"type": "enabled" if thinking_on else "disabled"}}
            if extra:
                kwargs.setdefault("extra_body", {}).update(extra)

        return self._model.bind(**kwargs)

    def _preprocess_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Apply prompt-caching markers for Anthropic models via gateways."""
        spec = self._spec
        if spec and spec.supports_prompt_caching:
            model_name = (self._default_model or "").lower()
            if any(model_name.startswith(k) for k in ("anthropic/", "claude")):
                messages, tools = self._apply_cache_control(messages, tools)
        return messages, tools

    @classmethod
    def _apply_cache_control(
        cls,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Inject cache_control markers into OpenAI-format messages/tools."""
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

        if new_messages and new_messages[0].get("role") == "system":
            new_messages[0] = _mark(new_messages[0])
        if len(new_messages) >= 3:
            new_messages[-2] = _mark(new_messages[-2])

        new_tools = tools
        if tools:
            new_tools = list(tools)
            for idx in cls._tool_cache_marker_indices(new_tools):
                new_tools[idx] = {**new_tools[idx], "cache_control": marker}

        return new_messages, new_tools

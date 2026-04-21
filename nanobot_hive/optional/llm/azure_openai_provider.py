"""Azure OpenAI provider backed by LangChain's AzureChatOpenAI."""

from __future__ import annotations

import uuid
from typing import Any

from langchain_openai import AzureChatOpenAI

from nanobot_hive.optional.llm.langchain_provider import LangChainProvider

# Default Azure OpenAI API version
_DEFAULT_API_VERSION = "2025-01-01-preview"

_REASONING_MODEL_TOKENS = frozenset({"gpt-5", "o1", "o3", "o4"})


class AzureOpenAIProvider(LangChainProvider):
    """Azure OpenAI provider using LangChain's AzureChatOpenAI (Chat Completions API).

    Supports reasoning models (o1/o3/o4/gpt-5) via reasoning_effort and
    standard models via temperature.
    """

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        default_model: str = "gpt-4o",
        api_version: str = _DEFAULT_API_VERSION,
    ) -> None:
        if not api_key:
            raise ValueError("Azure OpenAI api_key is required")
        if not api_base:
            raise ValueError("Azure OpenAI api_base is required")

        # Normalise endpoint: remove trailing slash
        endpoint = api_base.rstrip("/")

        model = AzureChatOpenAI(
            azure_deployment=default_model,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            default_headers={"x-session-affinity": uuid.uuid4().hex},
            max_retries=0,
        )
        super().__init__(model, default_model, api_key, api_base)
        self._endpoint = endpoint
        self._api_version = api_version

    @staticmethod
    def _supports_temperature(deployment: str, reasoning_effort: str | None) -> bool:
        if reasoning_effort:
            return False
        return not any(t in deployment.lower() for t in _REASONING_MODEL_TOKENS)

    def _get_model_for_call(
        self,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
    ) -> AzureChatOpenAI:
        deployment = model or self._default_model
        kwargs: dict[str, Any] = {
            "azure_deployment": deployment,
            "max_tokens": max(1, max_tokens),
        }

        if self._supports_temperature(deployment, reasoning_effort):
            kwargs["temperature"] = temperature

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        return self._model.bind(**kwargs)

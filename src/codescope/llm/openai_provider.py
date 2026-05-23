from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from typing import Any

from codescope.llm.providers import LLMRequest, LLMResponse

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"


class OpenAIProviderError(RuntimeError):
    """Safe, user-facing OpenAI provider error."""


class OpenAIProvider:
    """Optional OpenAI-compatible provider for LLM diagnosis.

    The OpenAI SDK is imported lazily only when the provider is used. This keeps
    normal CodeScope commands independent of optional LLM dependencies.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        env: Mapping[str, str] | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._env = env
        self._client = client

    @property
    def name(self) -> str:
        return "openai"

    def diagnose(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self._model or DEFAULT_OPENAI_MODEL
        try:
            client = self._client or self._create_client()
            response = client.responses.create(model=model, input=request.prompt)
            text = _extract_response_text(response)
        except OpenAIProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK exceptions vary by installed version.
            raise OpenAIProviderError(_safe_openai_error_message(exc)) from None

        if not text:
            raise OpenAIProviderError("OpenAI provider returned an empty response.")

        return LLMResponse(text=text, provider=self.name, model=model)

    def _create_client(self) -> Any:
        api_key = _clean_optional(_env(self._env).get(OPENAI_API_KEY_ENV_VAR))
        if api_key is None:
            raise OpenAIProviderError("OpenAI provider requires OPENAI_API_KEY.")

        try:
            openai_module = importlib.import_module("openai")
        except ImportError as exc:
            raise OpenAIProviderError(
                "OpenAI provider requires the optional OpenAI extra. "
                'Install it with: python -m pip install -e ".[openai]"'
            ) from exc

        openai_client = getattr(openai_module, "OpenAI", None)
        if openai_client is None:
            raise OpenAIProviderError("OpenAI SDK does not expose an OpenAI client.")

        return openai_client(api_key=api_key)


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    fallback_parts: list[str] = []
    for item in _iter_maybe(getattr(response, "output", None)):
        for content in _iter_maybe(getattr(item, "content", None)):
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                fallback_parts.append(text.strip())
                continue
            value = getattr(content, "value", None)
            if isinstance(value, str) and value.strip():
                fallback_parts.append(value.strip())

    return "\n".join(fallback_parts).strip()


def _iter_maybe(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return []


def _safe_openai_error_message(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()

    if "auth" in name or "permission" in name or "unauthorized" in message:
        return "OpenAI authentication failed."
    if "rate" in name or "rate limit" in message:
        return "OpenAI rate limit reached."
    if (
        "timeout" in name
        or "connection" in name
        or "network" in name
        or "timeout" in message
    ):
        return "OpenAI network or timeout error."
    return "OpenAI provider failed."


def _env(source: Mapping[str, str] | None) -> Mapping[str, str]:
    return source if source is not None else os.environ


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None

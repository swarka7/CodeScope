from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import pytest

import codescope.llm.openai_provider as openai_provider_module
from codescope.llm.openai_provider import (
    DEFAULT_OPENAI_MODEL,
    OpenAIProvider,
    OpenAIProviderError,
)
from codescope.llm.providers import LLMRequest


def test_openai_provider_sends_prompt_and_model_to_client() -> None:
    client = _FakeClient(response=_Response(output_text="Inspect the transfer flow."))
    provider = OpenAIProvider(model="configured-model", client=client)

    response = provider.diagnose(LLMRequest(prompt="retrieved context"))

    assert client.responses.calls == [
        {"model": "configured-model", "input": "retrieved context"}
    ]
    assert response.text == "Inspect the transfer flow."
    assert response.provider == "openai"
    assert response.model == "configured-model"


def test_request_model_overrides_configured_and_default_model() -> None:
    client = _FakeClient(response=_Response(output_text="Use the requested model."))
    provider = OpenAIProvider(model="configured-model", client=client)

    response = provider.diagnose(
        LLMRequest(prompt="retrieved context", model="request-model")
    )

    assert client.responses.calls[0]["model"] == "request-model"
    assert response.model == "request-model"


def test_default_model_is_used_when_no_model_is_configured() -> None:
    client = _FakeClient(response=_Response(output_text="Default model response."))
    provider = OpenAIProvider(client=client)

    response = provider.diagnose(LLMRequest(prompt="retrieved context"))

    assert client.responses.calls[0]["model"] == DEFAULT_OPENAI_MODEL
    assert response.model == DEFAULT_OPENAI_MODEL


def test_missing_openai_api_key_is_handled_clearly() -> None:
    provider = OpenAIProvider(env={})

    with pytest.raises(OpenAIProviderError) as exc_info:
        provider.diagnose(LLMRequest(prompt="retrieved context"))

    assert str(exc_info.value) == "OpenAI provider requires OPENAI_API_KEY."


def test_missing_openai_sdk_is_handled_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str) -> Any:
        if name == "openai":
            raise ImportError("No module named openai")
        return importlib.import_module(name)

    monkeypatch.setattr(openai_provider_module.importlib, "import_module", fake_import_module)
    provider = OpenAIProvider(env={"OPENAI_API_KEY": "sk-secret-value"})

    with pytest.raises(OpenAIProviderError) as exc_info:
        provider.diagnose(LLMRequest(prompt="retrieved context"))

    message = str(exc_info.value)
    assert "optional OpenAI extra" in message
    assert "sk-secret-value" not in message


def test_empty_openai_response_is_handled_clearly() -> None:
    provider = OpenAIProvider(client=_FakeClient(response=_Response(output_text="  ")))

    with pytest.raises(OpenAIProviderError) as exc_info:
        provider.diagnose(LLMRequest(prompt="retrieved context"))

    assert str(exc_info.value) == "OpenAI provider returned an empty response."


class _AuthenticationError(Exception):
    pass


class _RateLimitError(Exception):
    pass


class _APITimeoutError(Exception):
    pass


class _APIConnectionError(Exception):
    pass


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_AuthenticationError("bad key sk-secret-value"), "OpenAI authentication failed."),
        (_RateLimitError("too many requests"), "OpenAI rate limit reached."),
        (_APITimeoutError("request timed out"), "OpenAI network or timeout error."),
        (_APIConnectionError("network down"), "OpenAI network or timeout error."),
        (RuntimeError("provider failed with sk-secret-value"), "OpenAI provider failed."),
    ],
)
def test_openai_errors_are_normalized_without_secret_leakage(
    error: Exception, expected: str
) -> None:
    provider = OpenAIProvider(client=_FakeClient(error=error))

    with pytest.raises(OpenAIProviderError) as exc_info:
        provider.diagnose(LLMRequest(prompt="secret prompt should not leak"))

    message = str(exc_info.value)
    assert message == expected
    assert "sk-secret-value" not in message
    assert "secret prompt" not in message


def test_response_fallback_text_is_parsed_when_output_text_is_missing() -> None:
    provider = OpenAIProvider(
        client=_FakeClient(
            response=_Response(
                output=[
                    _OutputItem(content=[_Content(text="First part.")]),
                    _OutputItem(content=[_Content(value="Second part.")]),
                ]
            )
        )
    )

    response = provider.diagnose(LLMRequest(prompt="retrieved context"))

    assert response.text == "First part.\nSecond part."


@dataclass(slots=True)
class _Response:
    output_text: str | None = None
    output: list[Any] | None = None


@dataclass(slots=True)
class _OutputItem:
    content: list[Any]


@dataclass(slots=True)
class _Content:
    text: str | None = None
    value: str | None = None


class _FakeClient:
    def __init__(self, *, response: Any | None = None, error: Exception | None = None) -> None:
        self.responses = _FakeResponses(response=response, error=error)


class _FakeResponses:
    def __init__(self, *, response: Any | None, error: Exception | None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, str]] = []

    def create(self, *, model: str, input: str) -> Any:
        self.calls.append({"model": model, "input": input})
        if self._error is not None:
            raise self._error
        return self._response


from __future__ import annotations

import pytest

from codescope.llm.config import LLMConfig, load_llm_config, load_llm_provider
from codescope.llm.fake_provider import FakeLLMProvider
from codescope.llm.providers import LLMRequest


def test_fake_provider_returns_deterministic_response() -> None:
    provider = FakeLLMProvider(model="fake-model")
    request = LLMRequest(prompt="diagnose this failure")

    first = provider.diagnose(request)
    second = provider.diagnose(request)

    assert first.text == "Fake LLM diagnosis based on provided CodeScope context."
    assert second.text == first.text
    assert first.provider == "fake"
    assert first.model == "fake-model"


def test_fake_provider_captures_latest_request() -> None:
    provider = FakeLLMProvider()
    request = LLMRequest(prompt="context packet", model="request-model")

    response = provider.diagnose(request)

    assert provider.latest_request == request
    assert response.model == "request-model"


def test_load_llm_config_reads_provider_and_model_from_env_mapping() -> None:
    config = load_llm_config(
        {
            "CODESCOPE_LLM_PROVIDER": " fake ",
            "CODESCOPE_LLM_MODEL": " test-model ",
        }
    )

    assert config == LLMConfig(provider="fake", model="test-model")


def test_load_llm_provider_returns_none_when_provider_missing_or_none() -> None:
    assert load_llm_provider(LLMConfig()) is None
    assert load_llm_provider(LLMConfig(provider="none")) is None
    assert load_llm_provider(LLMConfig(provider="  ")) is None


def test_load_llm_provider_returns_fake_provider() -> None:
    provider = load_llm_provider(LLMConfig(provider="fake", model="fake-model"))

    assert isinstance(provider, FakeLLMProvider)
    response = provider.diagnose(LLMRequest(prompt="prompt"))
    assert response.provider == "fake"
    assert response.model == "fake-model"


def test_load_llm_provider_is_case_insensitive() -> None:
    provider = load_llm_provider(LLMConfig(provider="FaKe", model="fake-model"))

    assert isinstance(provider, FakeLLMProvider)


def test_load_llm_provider_rejects_invalid_provider_clearly() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider: unknown"):
        load_llm_provider(LLMConfig(provider="unknown"))


def test_no_network_or_api_key_required_for_fake_provider() -> None:
    provider = load_llm_provider(LLMConfig(provider="fake"))

    assert isinstance(provider, FakeLLMProvider)
    response = provider.diagnose(LLMRequest(prompt="no api key"))
    assert "Fake LLM diagnosis" in response.text

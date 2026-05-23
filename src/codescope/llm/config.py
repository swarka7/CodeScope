from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from codescope.llm.fake_provider import FakeLLMProvider
from codescope.llm.openai_provider import OpenAIProvider
from codescope.llm.providers import LLMProvider

PROVIDER_ENV_VAR = "CODESCOPE_LLM_PROVIDER"
MODEL_ENV_VAR = "CODESCOPE_LLM_MODEL"


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str | None = None
    model: str | None = None


def load_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    source = env if env is not None else os.environ
    provider = _clean_optional(source.get(PROVIDER_ENV_VAR))
    model = _clean_optional(source.get(MODEL_ENV_VAR))
    return LLMConfig(provider=provider, model=model)


def load_llm_provider(config: LLMConfig | None = None) -> LLMProvider | None:
    resolved = config or load_llm_config()
    provider = (resolved.provider or "").strip().lower()
    if not provider or provider == "none":
        return None
    if provider == "fake":
        return FakeLLMProvider(model=resolved.model)
    if provider == "openai":
        return OpenAIProvider(model=resolved.model)
    raise ValueError(f"Unsupported LLM provider: {resolved.provider}")


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None

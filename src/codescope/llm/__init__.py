"""Optional LLM provider interfaces for future diagnosis layers."""

from codescope.llm.config import LLMConfig, load_llm_config, load_llm_provider
from codescope.llm.fake_provider import FakeLLMProvider
from codescope.llm.openai_provider import OpenAIProvider, OpenAIProviderError
from codescope.llm.providers import LLMProvider, LLMRequest, LLMResponse

__all__ = [
    "FakeLLMProvider",
    "LLMConfig",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "OpenAIProvider",
    "OpenAIProviderError",
    "load_llm_config",
    "load_llm_provider",
]

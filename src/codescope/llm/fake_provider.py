from __future__ import annotations

from codescope.llm.providers import LLMRequest, LLMResponse


class FakeLLMProvider:
    """Deterministic non-network provider for tests and future CLI wiring."""

    def __init__(self, *, model: str | None = None) -> None:
        self._model = model
        self.latest_request: LLMRequest | None = None

    @property
    def name(self) -> str:
        return "fake"

    def diagnose(self, request: LLMRequest) -> LLMResponse:
        self.latest_request = request
        return LLMResponse(
            text="Fake LLM diagnosis based on provided CodeScope context.",
            provider=self.name,
            model=request.model or self._model,
        )

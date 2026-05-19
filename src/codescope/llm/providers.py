from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LLMRequest:
    prompt: str
    model: str | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str | None = None


class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    def diagnose(self, request: LLMRequest) -> LLMResponse: ...

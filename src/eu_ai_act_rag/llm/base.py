from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class LLMProvider(Protocol):
    """Minimal text-in/text-out interface. Structure is enforced by the caller (see
    `generation.answerer`), which keeps providers swappable and streaming-friendly."""

    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...

    def stream(self, system: str, prompt: str, max_tokens: int = 1024) -> Iterator[str]: ...


class ProviderNotConfigured(RuntimeError):
    """Raised when a provider is selected but its SDK or credentials are missing."""

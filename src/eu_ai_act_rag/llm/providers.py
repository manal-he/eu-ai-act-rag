"""Hosted LLM providers. SDKs are optional extras and imported lazily."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from eu_ai_act_rag.config import Settings
from eu_ai_act_rag.llm.base import LLMProvider, ProviderNotConfigured
from eu_ai_act_rag.llm.mock import MockLLM


class GeminiLLM:
    """Gemini on Vertex AI through the `google-genai` SDK (auth: Application Default
    Credentials / the Cloud Run service account - no API key)."""

    def __init__(self, model: str, project: str, location: str) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ProviderNotConfigured("pip install 'eu-ai-act-rag[vertex]'") from exc
        if not project:
            raise ProviderNotConfigured("GOOGLE_CLOUD_PROJECT is not set")
        self._types = types
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self.model = model
        self.name = f"gemini:{model}"

    def _config(self, system: str, max_tokens: int) -> Any:
        return self._types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens, temperature=0.0
        )

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._client.models.generate_content(
            model=self.model, contents=prompt, config=self._config(system, max_tokens)
        )
        return str(resp.text or "")

    def stream(self, system: str, prompt: str, max_tokens: int = 1024) -> Iterator[str]:
        for chunk in self._client.models.generate_content_stream(
            model=self.model, contents=prompt, config=self._config(system, max_tokens)
        ):
            if chunk.text:
                yield str(chunk.text)


class AnthropicLLM:
    def __init__(self, model: str, api_key: str) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ProviderNotConfigured("pip install 'eu-ai-act-rag[anthropic]'") from exc
        if not api_key:
            raise ProviderNotConfigured("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        return "".join(self.stream(system, prompt, max_tokens))

    def stream(self, system: str, prompt: str, max_tokens: int = 1024) -> Iterator[str]:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            yield from s.text_stream


class OpenAILLM:
    def __init__(self, model: str, api_key: str) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ProviderNotConfigured("pip install 'eu-ai-act-rag[openai]'") from exc
        if not api_key:
            raise ProviderNotConfigured("OPENAI_API_KEY is not set")
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"

    def _messages(self, system: str, prompt: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, prompt),
            max_tokens=max_tokens,
            temperature=0,
        )
        return str(resp.choices[0].message.content or "")

    def stream(self, system: str, prompt: str, max_tokens: int = 1024) -> Iterator[str]:
        for ev in self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, prompt),
            max_tokens=max_tokens,
            temperature=0,
            stream=True,
        ):
            delta = ev.choices[0].delta.content if ev.choices else None
            if delta:
                yield str(delta)


def build_llm(settings: Settings, provider: str | None = None) -> LLMProvider:
    name = provider or settings.llm_provider
    if name == "mock":
        return MockLLM()
    if name == "gemini":
        return GeminiLLM(
            settings.gemini_model, settings.google_cloud_project, settings.google_cloud_location
        )
    if name == "anthropic":
        return AnthropicLLM(settings.anthropic_model, settings.anthropic_api_key)
    if name == "openai":
        return OpenAILLM(settings.openai_model, settings.openai_api_key)
    raise ValueError(f"Unknown LLM provider: {name}")

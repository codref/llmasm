"""Ollama HTTP provider."""

from __future__ import annotations

from typing import Any

import httpx

from llmasm.errors import ProviderError
from llmasm.providers.base import EmbeddingOutput, ModelInfo, ModelOutput


class OllamaProvider:
    """Minimal Ollama provider implementation."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        default_model: str = "llama3.1:8b",
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_model = default_model
        self.embedding_model = embedding_model

    def list_models(self) -> list[ModelInfo]:
        """Return available Ollama models."""

        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama list_models failed: {exc}") from exc
        data = response.json()
        return [ModelInfo(name=item["name"]) for item in data.get("models", [])]

    def generate(
        self,
        prompt: str,
        options: dict[str, Any] | None = None,
        format_schema: dict[str, Any] | None = None,
    ) -> ModelOutput:
        """Generate text through Ollama."""

        payload: dict[str, Any] = {
            "model": (options or {}).get("model", self.default_model),
            "prompt": prompt,
            "stream": False,
            "options": options or {},
        }
        if format_schema is not None:
            payload["format"] = format_schema
        try:
            response = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama generate failed: {exc}") from exc
        data = response.json()
        usage = {
            "input_tokens": int(data.get("prompt_eval_count") or 0),
            "output_tokens": int(data.get("eval_count") or 0),
        }
        return ModelOutput(text=data.get("response", ""), raw=data, token_usage=usage)

    def embed(
        self,
        texts: list[str],
        options: dict[str, Any] | None = None,
    ) -> list[EmbeddingOutput]:
        """Embed texts through Ollama."""

        model = (options or {}).get("model", self.embedding_model)
        outputs: list[EmbeddingOutput] = []
        for text in texts:
            try:
                response = httpx.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderError(f"Ollama embed failed: {exc}") from exc
            data = response.json()
            outputs.append(EmbeddingOutput(vector=list(data.get("embedding", [])), raw=data))
        return outputs

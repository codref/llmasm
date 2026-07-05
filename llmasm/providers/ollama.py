"""Ollama HTTP provider."""

from __future__ import annotations

from typing import Any

import httpx

from llmasm.errors import ProviderError
from llmasm.providers.base import EmbeddingOutput, ModelInfo, ModelOutput, ToolCallOutput


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
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Generate text through Ollama.

        Uses ``/api/chat`` when ``tools`` or ``messages`` are provided,
        otherwise falls back to ``/api/generate`` for backward compatibility.
        """

        model = (options or {}).get("model", self.default_model)
        use_chat = bool(tools) or messages is not None

        if not use_chat:
            payload: dict[str, Any] = {
                "model": model,
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

        chat_messages = messages if messages is not None else [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": chat_messages,
            "stream": False,
            "options": options or {},
        }
        if tools:
            payload["tools"] = tools
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama chat failed: {exc}") from exc
        data = response.json()
        message = data.get("message") or {}
        usage = {
            "input_tokens": int(data.get("prompt_eval_count") or 0),
            "output_tokens": int(data.get("eval_count") or 0),
        }
        tool_calls = self._parse_tool_calls(message.get("tool_calls"))
        return ModelOutput(
            text=message.get("content", ""),
            raw=data,
            token_usage=usage,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _parse_tool_calls(raw: Any) -> list[ToolCallOutput]:
        """Parse Ollama tool_calls into ToolCallOutput objects."""

        if not raw:
            return []
        outputs: list[ToolCallOutput] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            function = item.get("function") or {}
            name = function.get("name") or item.get("name")
            if not name:
                continue
            arguments = function.get("arguments") or item.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            outputs.append(ToolCallOutput(name=str(name), arguments=arguments))
        return outputs

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

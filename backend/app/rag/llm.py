"""Provider-neutral language-model interfaces and Ollama integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import httpx

from app.rag.prompt_builder import ChatMessage


class LLMError(RuntimeError):
    """Base error raised by language-model providers."""


class LLMConnectionError(LLMError):
    """Raised when the configured model server cannot be reached."""


class LLMResponseError(LLMError):
    """Raised when a model server returns an invalid or failed response."""


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Provider-neutral text-generation settings."""

    temperature: float | None = 0.2
    top_p: float | None = None
    max_tokens: int | None = 512
    seed: int | None = None
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized response returned by any language-model provider."""

    content: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ns: int | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Interface implemented by all supported language-model clients."""

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        """Generate one assistant response."""
        ...


class OllamaLLMProvider:
    """Generate chat responses through Ollama's native HTTP API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        keep_alive: str | int | None = "5m",
        client: httpx.Client | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        message_list = list(messages)
        if not message_list:
            raise ValueError("messages must not be empty")

        generation_options = options or GenerationOptions()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in message_list
            ],
            "stream": False,
        }

        ollama_options = self._build_ollama_options(generation_options)
        if ollama_options:
            payload["options"] = ollama_options
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        try:
            response = self.client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                "Could not connect to Ollama. Confirm that Ollama is running "
                f"and available at {self.base_url}."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(
                f"Ollama did not respond within {self.timeout_seconds} seconds."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)
            raise LLMResponseError(
                f"Ollama returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMConnectionError(
                f"Ollama request failed: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                "Ollama returned a response that was not valid JSON."
            ) from exc

        return self._parse_response(data)

    def close(self) -> None:
        """Close the internally managed HTTP client."""
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "OllamaLLMProvider":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @staticmethod
    def _build_ollama_options(
        options: GenerationOptions,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}

        if options.temperature is not None:
            values["temperature"] = options.temperature
        if options.top_p is not None:
            values["top_p"] = options.top_p
        if options.max_tokens is not None:
            values["num_predict"] = options.max_tokens
        if options.seed is not None:
            values["seed"] = options.seed
        if options.stop:
            values["stop"] = list(options.stop)

        return values

    def _parse_response(self, data: Any) -> LLMResponse:
        if not isinstance(data, dict):
            raise LLMResponseError(
                "Ollama returned an unexpected response structure."
            )

        message = data.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError(
                "Ollama response is missing the assistant message."
            )

        content = message.get("content")
        if not isinstance(content, str):
            raise LLMResponseError(
                "Ollama response is missing assistant content."
            )

        model = data.get("model")
        if not isinstance(model, str) or not model.strip():
            model = self.model

        return LLMResponse(
            content=content.strip(),
            model=model,
            finish_reason=self._optional_string(data.get("done_reason")),
            prompt_tokens=self._optional_int(data.get("prompt_eval_count")),
            completion_tokens=self._optional_int(data.get("eval_count")),
            total_duration_ns=self._optional_int(data.get("total_duration")),
            raw_response=data,
        )

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            text = response.text.strip()
            return text or "unknown error"

        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, str) and error.strip():
                return error.strip()

        return "unknown error"

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

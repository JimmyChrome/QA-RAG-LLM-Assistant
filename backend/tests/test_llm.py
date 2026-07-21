"""Tests for the provider-neutral LLM layer and Ollama client."""

from __future__ import annotations

import json

import httpx
import pytest

from app.rag.llm import (
    GenerationOptions,
    LLMConnectionError,
    LLMResponseError,
    OllamaLLMProvider,
)
from app.rag.prompt_builder import ChatMessage


def messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="Use only supplied sources."),
        ChatMessage(role="user", content="What is the policy?"),
    ]


def test_generate_maps_messages_and_options() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "qwen-test",
                "message": {
                    "role": "assistant",
                    "content": "The policy is stated in [Source 1].",
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 42,
                "eval_count": 12,
                "total_duration": 123456,
            },
        )

    client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaLLMProvider(
        model="qwen-test",
        base_url="http://testserver",
        keep_alive="10m",
        client=client,
    )

    response = provider.generate(
        messages(),
        options=GenerationOptions(
            temperature=0.1,
            top_p=0.9,
            max_tokens=300,
            seed=7,
            stop=("END",),
        ),
    )

    assert captured["model"] == "qwen-test"
    assert captured["stream"] is False
    assert captured["keep_alive"] == "10m"
    assert captured["messages"] == [
        {"role": "system", "content": "Use only supplied sources."},
        {"role": "user", "content": "What is the policy?"},
    ]
    assert captured["options"] == {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 300,
        "seed": 7,
        "stop": ["END"],
    }
    assert response.content == "The policy is stated in [Source 1]."
    assert response.model == "qwen-test"
    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 42
    assert response.completion_tokens == 12
    assert response.total_duration_ns == 123456


def test_generate_uses_default_options() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "model",
                "message": {"role": "assistant", "content": "Answer"},
                "done": True,
            },
        )

    client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaLLMProvider(
        model="model",
        base_url="http://testserver",
        client=client,
    )

    provider.generate(messages())

    assert captured["options"]["temperature"] == 0.2
    assert captured["options"]["num_predict"] == 512


def test_http_error_includes_ollama_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": "model 'missing' not found"},
        )

    client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaLLMProvider(
        model="missing",
        base_url="http://testserver",
        client=client,
    )

    with pytest.raises(
        LLMResponseError,
        match="model 'missing' not found",
    ):
        provider.generate(messages())


def test_invalid_json_response_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not-json",
            headers={"content-type": "text/plain"},
        )

    client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaLLMProvider(
        model="model",
        base_url="http://testserver",
        client=client,
    )

    with pytest.raises(LLMResponseError, match="not valid JSON"):
        provider.generate(messages())


def test_missing_message_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "model", "done": True})

    client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaLLMProvider(
        model="model",
        base_url="http://testserver",
        client=client,
    )

    with pytest.raises(
        LLMResponseError,
        match="missing the assistant message",
    ):
        provider.generate(messages())


def test_connect_error_becomes_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaLLMProvider(
        model="model",
        base_url="http://testserver",
        client=client,
    )

    with pytest.raises(
        LLMConnectionError,
        match="Could not connect to Ollama",
    ):
        provider.generate(messages())


def test_empty_messages_are_rejected() -> None:
    provider = OllamaLLMProvider(
        model="model",
        client=httpx.Client(
            base_url="http://testserver",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={})
            ),
        ),
    )

    with pytest.raises(ValueError, match="messages must not be empty"):
        provider.generate([])


@pytest.mark.parametrize(
    "options",
    [
        GenerationOptions(temperature=0),
        GenerationOptions(top_p=1),
        GenerationOptions(max_tokens=1),
    ],
)
def test_valid_generation_boundaries(
    options: GenerationOptions,
) -> None:
    assert options is not None


def test_invalid_generation_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        GenerationOptions(temperature=-0.1)

    with pytest.raises(ValueError, match="top_p"):
        GenerationOptions(top_p=0)

    with pytest.raises(ValueError, match="max_tokens"):
        GenerationOptions(max_tokens=0)


def test_blank_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="model must not be empty"):
        OllamaLLMProvider(model=" ")

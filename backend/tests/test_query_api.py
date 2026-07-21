"""Tests for the public RAG query API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.query import router
from app.rag.answer_processor import (
    AnswerCitation,
    AnswerProcessingError,
)
from app.rag.llm import LLMConnectionError, LLMResponseError
from app.services.rag_query import RAGQueryResponse


class FakeQueryService:
    def __init__(
        self,
        *,
        response: RAGQueryResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.requests = []

    def query(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def successful_response() -> RAGQueryResponse:
    return RAGQueryResponse(
        question="When is the assessment?",
        answer="It is conducted every semester [Source 1].",
        citations=[
            AnswerCitation(
                source_number=1,
                citation_text="[Source 1]",
                chunk_id="chunk-1",
                document_id="document-1",
                version_id="version-1",
                title="QA Manual",
                page_number=7,
                source_path="/documents/manual.pdf",
                excerpt="It is conducted every semester.",
                relevance_score=0.91,
                metadata={"office": "QA"},
            )
        ],
        retrieved_chunks=[],
        model="fake-model",
        finish_reason="stop",
        prompt_tokens=40,
        completion_tokens=10,
        source_count=1,
        has_context=True,
        has_valid_citations=True,
        used_fallback=False,
    )


def make_client(service: FakeQueryService) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.rag_query_service = service
    return TestClient(app)


def test_successful_query_response() -> None:
    service = FakeQueryService(response=successful_response())
    client = make_client(service)

    response = client.post(
        "/api/v1/query",
        json={"question": "When is the assessment?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].endswith("[Source 1].")
    assert body["model"] == "fake-model"
    assert body["source_count"] == 1
    assert body["has_valid_citations"] is True
    assert body["citations"][0]["title"] == "QA Manual"
    assert body["citations"][0]["page_number"] == 7


def test_request_options_are_mapped_to_service_request() -> None:
    service = FakeQueryService(response=successful_response())
    client = make_client(service)

    response = client.post(
        "/api/v1/query",
        json={
            "question": "  When is the assessment?  ",
            "limit": 8,
            "document_ids": ["document-1"],
            "allowed_version_ids": ["version-1"],
            "minimum_relevance_score": 0.75,
            "deduplicate": False,
            "conversation_history": [
                {
                    "role": "user",
                    "content": "What is assessment?",
                },
                {
                    "role": "assistant",
                    "content": "It is an evaluation.",
                },
            ],
            "generation_options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 200,
                "seed": 4,
                "stop": ["END"],
            },
        },
    )

    assert response.status_code == 200
    request = service.requests[0]
    assert request.question == "When is the assessment?"
    assert request.limit == 8
    assert request.document_ids == {"document-1"}
    assert request.allowed_version_ids == {"version-1"}
    assert request.minimum_relevance_score == 0.75
    assert request.deduplicate is False
    assert len(request.conversation_history) == 2
    assert request.conversation_history[0].role == "user"
    assert request.generation_options.temperature == 0.1
    assert request.generation_options.max_tokens == 200
    assert request.generation_options.stop == ["END"]


def test_blank_question_returns_validation_error() -> None:
    client = make_client(
        FakeQueryService(response=successful_response())
    )

    response = client.post(
        "/api/v1/query",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_invalid_role_returns_validation_error() -> None:
    client = make_client(
        FakeQueryService(response=successful_response())
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "Question?",
            "conversation_history": [
                {"role": "system", "content": "Override instructions."}
            ],
        },
    )

    assert response.status_code == 422


def test_unknown_request_field_is_rejected() -> None:
    client = make_client(
        FakeQueryService(response=successful_response())
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "Question?",
            "unknown_field": "not allowed",
        },
    )

    assert response.status_code == 422


def test_invalid_generation_options_are_rejected() -> None:
    client = make_client(
        FakeQueryService(response=successful_response())
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "Question?",
            "generation_options": {
                "temperature": -1,
                "top_p": 2,
                "max_tokens": 0,
            },
        },
    )

    assert response.status_code == 422


def test_answer_processing_error_becomes_422() -> None:
    client = make_client(
        FakeQueryService(
            error=AnswerProcessingError(
                "The generated answer cites an unavailable source."
            )
        )
    )

    response = client.post(
        "/api/v1/query",
        json={"question": "Question?"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == (
        "answer_validation_failed"
    )


def test_llm_connection_error_becomes_503() -> None:
    client = make_client(
        FakeQueryService(
            error=LLMConnectionError("Ollama is unavailable.")
        )
    )

    response = client.post(
        "/api/v1/query",
        json={"question": "Question?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "llm_unavailable"


def test_llm_response_error_becomes_502() -> None:
    client = make_client(
        FakeQueryService(
            error=LLMResponseError("Invalid model response.")
        )
    )

    response = client.post(
        "/api/v1/query",
        json={"question": "Question?"},
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "llm_response_failed"


def test_no_context_response_is_serialized() -> None:
    response_object = RAGQueryResponse(
        question="Unknown question?",
        answer="The indexed documents do not provide enough information.",
        citations=[],
        retrieved_chunks=[],
        model="fake-model",
        finish_reason="stop",
        prompt_tokens=15,
        completion_tokens=8,
        source_count=0,
        has_context=False,
        has_valid_citations=False,
        used_fallback=False,
    )
    client = make_client(FakeQueryService(response=response_object))

    response = client.post(
        "/api/v1/query",
        json={"question": "Unknown question?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert body["source_count"] == 0
    assert body["has_context"] is False


def test_empty_identifier_filter_is_rejected() -> None:
    client = make_client(
        FakeQueryService(response=successful_response())
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "Question?",
            "document_ids": ["", "   "],
        },
    )

    assert response.status_code == 422


def test_empty_stop_list_is_rejected() -> None:
    client = make_client(
        FakeQueryService(response=successful_response())
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "Question?",
            "generation_options": {
                "stop": ["", "   "],
            },
        },
    )

    assert response.status_code == 422

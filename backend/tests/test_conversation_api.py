"""API tests for conversation endpoints using a fake service."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_conversation_service
from app.api.routes.conversations import router
from app.services.conversation import ConversationNotFoundError


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)


class FakeConversationService:
    def create_conversation(self, title=None):
        return conversation(title or "New conversation")

    def list_conversations(self, *, limit=50, offset=0):
        return (conversation("Conversation"),)

    def get_conversation(self, conversation_id):
        if conversation_id == "missing":
            raise ConversationNotFoundError("Conversation not found.")
        return conversation("Conversation", include_messages=True)

    def delete_conversation(self, conversation_id):
        if conversation_id == "missing":
            raise ConversationNotFoundError("Conversation not found.")

    def query_conversation(self, *, conversation_id, request):
        if conversation_id == "missing":
            raise ConversationNotFoundError("Conversation not found.")

        query = SimpleNamespace(
            question=request.question,
            answer="Answer",
            citations=(),
            model="fake-model",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            source_count=0,
            has_context=False,
            has_valid_citations=True,
            used_fallback=False,
        )
        return SimpleNamespace(
            conversation=conversation("Conversation", include_messages=True),
            query=query,
        )


def conversation(title, include_messages=False):
    messages = ()
    if include_messages:
        messages = (
            SimpleNamespace(
                id="message-1",
                role="user",
                content="Question",
                citations=(),
                sequence_number=1,
                created_at=NOW,
            ),
        )
    return SimpleNamespace(
        id="conversation-1",
        title=title,
        created_at=NOW,
        updated_at=NOW,
        messages=messages,
    )


app = FastAPI()
app.include_router(router)
app.dependency_overrides[get_conversation_service] = (
    lambda: FakeConversationService()
)
client = TestClient(app)


def test_create_conversation():
    response = client.post(
        "/api/v1/conversations",
        json={"title": "QA chat"},
    )

    assert response.status_code == 201
    assert response.json()["title"] == "QA chat"


def test_list_conversations():
    response = client.get("/api/v1/conversations")

    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_get_conversation():
    response = client.get("/api/v1/conversations/conversation-1")

    assert response.status_code == 200
    assert len(response.json()["messages"]) == 1


def test_get_missing_conversation():
    response = client.get("/api/v1/conversations/missing")

    assert response.status_code == 404


def test_delete_conversation():
    response = client.delete("/api/v1/conversations/conversation-1")

    assert response.status_code == 204


def test_delete_missing_conversation():
    response = client.delete("/api/v1/conversations/missing")

    assert response.status_code == 404


def test_query_conversation():
    response = client.post(
        "/api/v1/conversations/conversation-1/query",
        json={"question": "What is QA?"},
    )

    assert response.status_code == 200
    assert response.json()["query"]["answer"] == "Answer"


def test_query_missing_conversation():
    response = client.post(
        "/api/v1/conversations/missing/query",
        json={"question": "What is QA?"},
    )

    assert response.status_code == 404


def test_query_validation():
    response = client.post(
        "/api/v1/conversations/conversation-1/query",
        json={"question": "   "},
    )

    assert response.status_code == 422

"""Tests for ConversationService without a real database or LLM."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.services.conversation import (
    ConversationNotFoundError,
    ConversationService,
)
from app.services.rag_query import RAGQueryRequest


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)


class FakeRepository:
    def __init__(self):
        self.items = {}
        self.commits = 0
        self.rollbacks = 0

    def create(self, *, title):
        item = SimpleNamespace(
            id="conversation-1",
            title=title,
            created_at=NOW,
            updated_at=NOW,
            messages=[],
        )
        self.items[item.id] = item
        return item

    def get(self, conversation_id):
        return self.items.get(conversation_id)

    def list(self, *, limit=50, offset=0):
        return tuple(self.items.values())[offset : offset + limit]

    def add_message(
        self,
        *,
        conversation_id,
        role,
        content,
        citations=None,
    ):
        conversation = self.items[conversation_id]
        message = SimpleNamespace(
            id=f"message-{len(conversation.messages) + 1}",
            role=role,
            content=content,
            citations=citations or [],
            sequence_number=len(conversation.messages) + 1,
            created_at=NOW,
        )
        conversation.messages.append(message)
        return message

    def delete(self, conversation_id):
        return self.items.pop(conversation_id, None) is not None

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakeRAGService:
    def __init__(self):
        self.last_request = None

    def query(self, request):
        self.last_request = request
        return SimpleNamespace(
            question=request.question,
            answer="Grounded answer [1].",
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


def make_service():
    repository = FakeRepository()
    rag = FakeRAGService()
    service = ConversationService(
        repository=repository,
        rag_query_service=rag,
        history_limit=20,
    )
    return service, repository, rag


def test_create_conversation_uses_default_title():
    service, repository, _ = make_service()

    result = service.create_conversation()

    assert result.title == "New conversation"
    assert repository.commits == 1


def test_create_conversation_strips_title():
    service, _, _ = make_service()

    result = service.create_conversation("  QA policies  ")

    assert result.title == "QA policies"


def test_get_missing_conversation_raises():
    service, _, _ = make_service()

    with pytest.raises(ConversationNotFoundError):
        service.get_conversation("missing")


def test_list_conversations():
    service, _, _ = make_service()
    service.create_conversation("One")

    results = service.list_conversations()

    assert len(results) == 1
    assert results[0].title == "One"


def test_delete_conversation():
    service, repository, _ = make_service()
    created = service.create_conversation()

    service.delete_conversation(created.id)

    assert repository.get(created.id) is None


def test_delete_missing_conversation_raises():
    service, _, _ = make_service()

    with pytest.raises(ConversationNotFoundError):
        service.delete_conversation("missing")


def test_query_saves_user_and_assistant_messages():
    service, repository, _ = make_service()
    created = service.create_conversation()

    result = service.query_conversation(
        conversation_id=created.id,
        request=RAGQueryRequest(question="What is QA?"),
    )

    assert [message.role for message in result.conversation.messages] == [
        "user",
        "assistant",
    ]
    assert repository.commits == 2


def test_query_passes_existing_history_to_rag_service():
    service, repository, rag = make_service()
    created = service.create_conversation()
    repository.add_message(
        conversation_id=created.id,
        role="user",
        content="Earlier question",
    )
    repository.add_message(
        conversation_id=created.id,
        role="assistant",
        content="Earlier answer",
    )

    service.query_conversation(
        conversation_id=created.id,
        request=RAGQueryRequest(question="Follow-up"),
    )

    assert [message.content for message in rag.last_request.conversation_history] == [
        "Earlier question",
        "Earlier answer",
    ]


def test_query_missing_conversation_raises():
    service, _, _ = make_service()

    with pytest.raises(ConversationNotFoundError):
        service.query_conversation(
            conversation_id="missing",
            request=RAGQueryRequest(question="Question"),
        )

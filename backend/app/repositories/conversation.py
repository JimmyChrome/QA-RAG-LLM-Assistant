"""SQLAlchemy repository for conversations and messages."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Conversation, ConversationMessage

from datetime import datetime, timezone


class ConversationRepository:
    """Persist and retrieve conversation records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, *, title: str) -> Conversation:
        conversation = Conversation(title=title)
        self._session.add(conversation)
        self._session.flush()
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        statement = (
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.messages))
        )
        return self._session.scalar(statement)

    def list(self, *, limit: int = 50, offset: int = 0) -> Sequence[Conversation]:
        statement = (
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return tuple(self._session.scalars(statement).all())

    def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict] | None = None,
    ) -> ConversationMessage:
        highest_sequence = self._session.scalar(
            select(func.max(ConversationMessage.sequence_number)).where(
                ConversationMessage.conversation_id == conversation_id
            )
        )

        now = datetime.now(timezone.utc)

        message = ConversationMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations or [],
            sequence_number=(highest_sequence or 0) + 1,
            created_at=now,
        )
        self._session.add(message)

        conversation = self._session.get(Conversation, conversation_id)
        if conversation is not None:
            conversation.updated_at = now

        self._session.flush()
        return message

    def delete(self, conversation_id: str) -> bool:
        result = self._session.execute(
            delete(Conversation).where(Conversation.id == conversation_id)
        )
        return bool(result.rowcount)

    def commit(self) -> None:
        self._session.commit()

    def rollback(self) -> None:
        self._session.rollback()

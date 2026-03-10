"""
backend/database/models.py

SQLAlchemy ORM models for the Dynamic Survey Campaign Platform.

Key change from original:
  UUIDs are stored as String(36) instead of PostgreSQL-native UUID columns.
  This makes the models work identically on SQLite (dev) and PostgreSQL (prod)
  without any code changes between environments.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id            = Column(String(36), primary_key=True, default=_uuid)
    name          = Column(String(255), nullable=False)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role          = Column(String(20), nullable=False, default="interviewer")  # admin | interviewer
    created_at    = Column(DateTime, default=_now)

    campaigns = relationship("Campaign", back_populates="creator", lazy="select")

    def __repr__(self):
        return f"<User id={self.id} email={self.email} role={self.role}>"


# ---------------------------------------------------------------------------
# Campaigns
# ---------------------------------------------------------------------------
class Campaign(Base):
    __tablename__ = "campaigns"

    id                    = Column(String(36), primary_key=True, default=_uuid)
    name                  = Column(String(255), nullable=False)
    description           = Column(Text, nullable=True)
    created_by            = Column(String(36), ForeignKey("users.id"), nullable=False)
    google_sheet_tab_name = Column(String(255), nullable=True)
    status                = Column(String(20), nullable=False, default="draft")  # draft | active | closed
    created_at            = Column(DateTime, default=_now)

    creator   = relationship("User", back_populates="campaigns")
    questions = relationship(
        "Question", back_populates="campaign",
        order_by="Question.question_order",
        cascade="all, delete-orphan",
    )
    responses = relationship(
        "Response", back_populates="campaign",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Campaign id={self.id} name={self.name} status={self.status}>"


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
class Question(Base):
    __tablename__ = "questions"

    id               = Column(String(36), primary_key=True, default=_uuid)
    campaign_id      = Column(String(36), ForeignKey("campaigns.id"), nullable=False)
    question_order   = Column(Integer, nullable=False)
    question_text    = Column(Text, nullable=False)
    question_type    = Column(String(30), nullable=False, default="open_text")
    # scale_1_5 | multiple_choice | yes_no | numeric | open_text
    options          = Column(JSON, nullable=True)
    validation_rules = Column(JSON, nullable=True)
    required         = Column(Boolean, nullable=False, default=True)
    # Logic/branching rules — evaluated after the respondent answers this question.
    # Format: [{"condition": "Yes", "next_order": 5}, {"condition": "default", "next_order": 3}]
    # "condition" can be an exact answer value or "default" (fallback for unmatched answers).
    # "next_order" is the question_order of the next question to ask.
    # If null/absent the agent simply moves to the next sequential question.
    question_logic   = Column(JSON, nullable=True)
    created_at       = Column(DateTime, default=_now)

    campaign = relationship("Campaign", back_populates="questions")

    def __repr__(self):
        return f"<Question id={self.id} order={self.question_order} type={self.question_type}>"


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class Response(Base):
    __tablename__ = "responses"

    id                 = Column(String(36), primary_key=True, default=_uuid)
    campaign_id        = Column(String(36), ForeignKey("campaigns.id"), nullable=False)
    call_id            = Column(String(255), nullable=False, index=True)
    structured_answers = Column(JSON, nullable=True)
    transcript         = Column(Text, nullable=True)
    detected_gender    = Column(String(10), nullable=True)   # M / F / Unknown
    created_at         = Column(DateTime, default=_now)

    campaign = relationship("Campaign", back_populates="responses")

    def __repr__(self):
        return f"<Response id={self.id} call_id={self.call_id}>"
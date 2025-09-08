from __future__ import annotations
import enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, String, Text, Enum, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

DATABASE_URL = "mysql+pymysql://user:Ananyas%40028@localhost:3306/policypal"

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class Base(DeclarativeBase): pass

class RoleEnum(str, enum.Enum):
    user = "user"
    admin = "admin"

class FeedbackEnum(str, enum.Enum):
    up = "up"
    down = "down"
    none = "none"

class User(Base):
    __tablename__ = "users"
    user_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(100))
    email: Mapped[Optional[str]] = mapped_column(String(255))
    password_hash: Mapped[Optional[str]] = mapped_column(Text)
    role: Mapped[RoleEnum] = mapped_column(Enum(RoleEnum), default=RoleEnum.user)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    policies: Mapped[List["Policy"]] = relationship(back_populates="user", cascade="all,delete-orphan")
    queries: Mapped[List["Query"]] = relationship(back_populates="user", cascade="all,delete-orphan")

class Policy(Base):
    __tablename__ = "policies"
    policy_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.user_id"), index=True)
    policy_name: Mapped[Optional[str]] = mapped_column(String(200))
    policy_type: Mapped[Optional[str]] = mapped_column(String(100))
    storage_bucket: Mapped[Optional[str]] = mapped_column(String(100))
    storage_path: Mapped[str] = mapped_column(String(255))
    policy_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    user: Mapped["User"] = relationship(back_populates="policies")
    texts: Mapped[List["PolicyText"]] = relationship(back_populates="policy", cascade="all,delete-orphan")
    queries: Mapped[List["Query"]] = relationship(back_populates="policy", cascade="all,delete-orphan")
    __table_args__ = (Index("idx_policies_user_id", "user_id"),)

class PolicyText(Base):
    __tablename__ = "policy_texts"
    policy_text_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    policy_id: Mapped[str] = mapped_column(String(36), ForeignKey("policies.policy_id"), index=True)
    raw_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    policy: Mapped["Policy"] = relationship(back_populates="texts")

class Query(Base):
    __tablename__ = "queries"
    query_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.user_id"), index=True)
    policy_id: Mapped[str] = mapped_column(String(36), ForeignKey("policies.policy_id"), index=True)
    query_text: Mapped[str] = mapped_column(Text)
    answer_text: Mapped[str] = mapped_column(Text)
    feedback: Mapped[FeedbackEnum] = mapped_column(Enum(FeedbackEnum), default=FeedbackEnum.none)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    user: Mapped["User"] = relationship(back_populates="queries")
    policy: Mapped["Policy"] = relationship(back_populates="queries")

def init_models():
    Base.metadata.create_all(engine)

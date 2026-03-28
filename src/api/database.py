"""
SQLite database setup with SQLAlchemy async sessions.
Database file: data/agents.db
"""

import os
from sqlalchemy import (
    Column, String, Text, DateTime, CheckConstraint,
    ForeignKey, create_engine, event
)
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# ── Database Path ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "agents.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)

# Enable WAL mode for better concurrent read performance
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ─────────────────────────────────────────────────────────────────

class AgentDB(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="agent")
    status = Column(String, nullable=False, default="offline")
    last_ping = Column(DateTime, default=datetime.utcnow)
    avatar = Column(String, default="")
    display_name = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("role IN ('admin', 'agent')", name="ck_agent_role"),
        CheckConstraint("status IN ('offline', 'online', 'on-call')", name="ck_agent_status"),
    )


class SessionDB(Base):
    __tablename__ = "sessions"

    token = Column(String, primary_key=True)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)


# ── Dependency ─────────────────────────────────────────────────────────────────

def get_db():
    """FastAPI dependency: yields a DB session, auto-closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

"""
CRUD operations for agents and sessions.
"""

from datetime import datetime
from sqlalchemy.orm import Session
from api.database import AgentDB, SessionDB


# ── Agent CRUD ─────────────────────────────────────────────────────────────────

def get_agent_by_id(db: Session, agent_id: str):
    return db.query(AgentDB).filter(AgentDB.id == agent_id).first()


def get_agent_by_username(db: Session, username: str):
    return db.query(AgentDB).filter(AgentDB.username == username).first()


def get_all_agents(db: Session):
    return db.query(AgentDB).all()


def create_agent(db: Session, agent_id: str, username: str, password_hash: str,
                 role: str = "agent", display_name: str = "", avatar: str = ""):
    agent = AgentDB(
        id=agent_id,
        username=username,
        password_hash=password_hash,
        role=role,
        display_name=display_name or username,
        avatar=avatar,
        status="offline",
        last_ping=datetime.utcnow(),
        created_at=datetime.utcnow(),
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def update_agent_status(db: Session, agent_id: str, status: str):
    agent = get_agent_by_id(db, agent_id)
    if agent:
        agent.status = status
        agent.last_ping = datetime.utcnow()
        db.commit()
        db.refresh(agent)
    return agent


def heartbeat_agent(db: Session, agent_id: str):
    """Touch last_ping timestamp to keep agent online."""
    agent = get_agent_by_id(db, agent_id)
    if agent:
        agent.last_ping = datetime.utcnow()
        if agent.status == "offline":
            agent.status = "online"
        db.commit()
        db.refresh(agent)
    return agent


def delete_agent(db: Session, agent_id: str):
    agent = get_agent_by_id(db, agent_id)
    if agent:
        db.delete(agent)
        db.commit()
        return True
    return False


# ── Session CRUD ───────────────────────────────────────────────────────────────

def create_session(db: Session, token: str, agent_id: str, expires_at: datetime):
    session = SessionDB(token=token, agent_id=agent_id, expires_at=expires_at)
    db.add(session)
    db.commit()
    return session


def get_session(db: Session, token: str):
    return db.query(SessionDB).filter(SessionDB.token == token).first()


def delete_session(db: Session, token: str):
    session = get_session(db, token)
    if session:
        db.delete(session)
        db.commit()


def cleanup_expired_sessions(db: Session):
    """Remove all expired sessions."""
    db.query(SessionDB).filter(SessionDB.expires_at < datetime.utcnow()).delete()
    db.commit()

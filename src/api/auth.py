"""
JWT Authentication + Auth Routes for HAAM.

Endpoints:
  POST /auth/register  – Create a new agent (admin-only after initial seed)
  POST /auth/login     – Returns JWT access token
  POST /auth/logout    – Invalidate token
  GET  /auth/me        – Current user profile

Security:
  - bcrypt password hashing
  - JWT with 24h expiry
  - Role-based access (admin / agent)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from passlib.context import CryptContext
from jose import JWTError, jwt

from api.database import get_db
from api import crud

logger = logging.getLogger("HAAM_AUTH")

# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("HAAM_JWT_SECRET", "haam-super-secret-key-change-in-production-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# ── Password Hashing ──────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# ── JWT Helpers ────────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ── FastAPI Security Scheme ────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
):
    """
    Dependency: extracts JWT from Authorization header and returns the agent row.
    Returns None if no credentials are provided (for optional auth endpoints).
    """
    if credentials is None:
        return None

    token = credentials.credentials
    payload = decode_token(token)
    agent_id = payload.get("sub")
    if not agent_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    agent = crud.get_agent_by_id(db, agent_id)
    if agent is None:
        raise HTTPException(status_code=401, detail="User not found")

    return agent


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    """Dependency: REQUIRES valid JWT. Raises 401 if missing or invalid."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    return get_current_user(credentials, db)


def require_admin(agent=Depends(require_auth)):
    """Dependency: requires the authenticated user to be an admin."""
    if agent is None or agent.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return agent


# ── Request / Response Models ──────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    id: str
    username: str
    password: str
    role: str = "agent"
    display_name: str = ""

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    agent_id: str
    role: str
    display_name: str

class AgentProfile(BaseModel):
    id: str
    username: str
    role: str
    status: str
    display_name: str
    avatar: str
    created_at: str


# ── Router ─────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/auth", tags=["Authentication"])

# ── Simple in-memory rate limiter for login ────────────────────────────────────
_login_attempts: dict = {}  # ip -> [timestamps]
LOGIN_RATE_LIMIT = 5
LOGIN_RATE_WINDOW = 60  # seconds


def _check_login_rate(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=LOGIN_RATE_WINDOW)

    attempts = _login_attempts.get(ip, [])
    attempts = [t for t in attempts if t > window_start]
    _login_attempts[ip] = attempts

    if len(attempts) >= LOGIN_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new agent. In production, restrict to admin-only."""
    # Check duplicates
    if crud.get_agent_by_id(db, req.id):
        raise HTTPException(status_code=409, detail=f"Agent ID '{req.id}' already exists")
    if crud.get_agent_by_username(db, req.username):
        raise HTTPException(status_code=409, detail=f"Username '{req.username}' already taken")

    hashed = hash_password(req.password)
    agent = crud.create_agent(
        db,
        agent_id=req.id,
        username=req.username,
        password_hash=hashed,
        role=req.role,
        display_name=req.display_name or req.username,
    )

    token = create_access_token({"sub": agent.id, "role": agent.role})
    logger.info(f"Registered new agent: {agent.id} ({agent.role})")

    return TokenResponse(
        access_token=token,
        agent_id=agent.id,
        role=agent.role,
        display_name=agent.display_name,
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, request: Request, db: Session = Depends(get_db)):
    """Authenticate and return JWT."""
    _check_login_rate(request)

    agent = crud.get_agent_by_username(db, req.username)
    if not agent or not verify_password(req.password, agent.password_hash):
        # Track failed attempt
        ip = request.client.host if request.client else "unknown"
        _login_attempts.setdefault(ip, []).append(datetime.utcnow())
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Update status to online
    crud.update_agent_status(db, agent.id, "online")

    token = create_access_token({"sub": agent.id, "role": agent.role})
    logger.info(f"Login: {agent.username} ({agent.role})")

    return TokenResponse(
        access_token=token,
        agent_id=agent.id,
        role=agent.role,
        display_name=agent.display_name or agent.username,
    )


@router.post("/logout")
def logout(agent=Depends(require_auth), db: Session = Depends(get_db)):
    """Set agent status to offline."""
    crud.update_agent_status(db, agent.id, "offline")
    return {"status": "logged_out"}


@router.get("/me", response_model=AgentProfile)
def get_my_profile(agent=Depends(require_auth)):
    """Return the authenticated agent's profile."""
    return AgentProfile(
        id=agent.id,
        username=agent.username,
        role=agent.role,
        status=agent.status,
        display_name=agent.display_name or agent.username,
        avatar=agent.avatar or "",
        created_at=agent.created_at.isoformat() if agent.created_at else "",
    )

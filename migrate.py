"""
HAAM Database Migration Script
Run once: python migrate.py

Creates the SQLite database and seeds:
  - 1 Admin  (admin / admin123)
  - 3 Agents (sham / pass123, priya / pass123, rahul / pass123)
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from api.database import init_db, SessionLocal, DB_PATH
from api.crud import get_agent_by_id, create_agent
from api.auth import hash_password


SEED_AGENTS = [
    {
        "id": "admin_001",
        "username": "admin",
        "password": "admin123",
        "role": "admin",
        "display_name": "HAAM Admin",
    },
    {
        "id": "agent_001",
        "username": "sham",
        "password": "pass123",
        "role": "agent",
        "display_name": "Sham",
    },
    {
        "id": "agent_002",
        "username": "priya",
        "password": "pass123",
        "role": "agent",
        "display_name": "Priya",
    },
    {
        "id": "agent_003",
        "username": "rahul",
        "password": "pass123",
        "role": "agent",
        "display_name": "Rahul",
    },
]


def migrate():
    print(f"📦 Database path: {DB_PATH}")

    # 1. Create tables
    init_db()
    print("✅ Tables created (agents, sessions)")

    # 2. Seed demo users
    db = SessionLocal()
    try:
        for agent_data in SEED_AGENTS:
            existing = get_agent_by_id(db, agent_data["id"])
            if existing:
                print(f"   ⏭️  {agent_data['id']} ({agent_data['username']}) already exists — skipped")
                continue

            create_agent(
                db,
                agent_id=agent_data["id"],
                username=agent_data["username"],
                password_hash=hash_password(agent_data["password"]),
                role=agent_data["role"],
                display_name=agent_data["display_name"],
            )
            print(f"   ✅ Created {agent_data['role'].upper()}: {agent_data['username']} (id={agent_data['id']})")
    finally:
        db.close()

    print()
    print("🎉 Migration complete! Demo credentials:")
    print("   ┌──────────┬──────────┬──────────┐")
    print("   │ Username │ Password │ Role     │")
    print("   ├──────────┼──────────┼──────────┤")
    print("   │ admin    │ admin123 │ admin    │")
    print("   │ sham     │ pass123  │ agent    │")
    print("   │ priya    │ pass123  │ agent    │")
    print("   │ rahul    │ pass123  │ agent    │")
    print("   └──────────┴──────────┴──────────┘")


if __name__ == "__main__":
    migrate()

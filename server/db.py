from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "pipeline.db"


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS jobs (
        name TEXT PRIMARY KEY, video TEXT NOT NULL, route TEXT NOT NULL,
        status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        error TEXT
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at DESC)")
    db.execute("PRAGMA optimize")
    return db


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create(name: str, video: str, route: str) -> None:
    with connect() as db:
        db.execute("INSERT INTO jobs VALUES (?, ?, ?, 'QUEUED', ?, ?, NULL)", (name, video, route, now(), now()))


def update(name: str, status: str, error: str | None = None) -> None:
    with connect() as db:
        db.execute("UPDATE jobs SET status=?, updated_at=?, error=? WHERE name=?", (status, now(), error, name))


def all_jobs() -> list[dict]:
    with connect() as db:
        return [dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY updated_at DESC")]

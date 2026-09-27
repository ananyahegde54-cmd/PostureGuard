"""
database.py — SQLite database for PostureGuard
────────────────────────────────────────────────
One file, no server: postureguard.db
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "postureguard.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            baseline_done INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS baselines (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            username         TEXT NOT NULL,
            metrics_json     TEXT NOT NULL,
            frames_used      INTEGER,
            stability_score  INTEGER,
            saved_at         TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_user_by_username(username):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def create_user(username, password_hash):
    conn = get_conn()
    conn.execute(
        "INSERT INTO users (username, password_hash, baseline_done, created_at) VALUES (?, ?, 0, ?)",
        (username, password_hash, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def mark_baseline_done(username):
    conn = get_conn()
    conn.execute("UPDATE users SET baseline_done = 1 WHERE username = ?", (username,))
    conn.commit()
    conn.close()


def save_baseline(username, metrics, frames_used, stability_score):
    conn = get_conn()
    conn.execute(
        """INSERT INTO baselines (username, metrics_json, frames_used, stability_score, saved_at)
           VALUES (?, ?, ?, ?, ?)""",
        (username, json.dumps(metrics), frames_used, stability_score, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
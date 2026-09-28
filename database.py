"""SQLite database helpers for PostureGuard.

Stores users, personalized baselines, and per-user monitoring sessions.
"""

import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "postureguard.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create all required tables without deleting existing data."""
    conn = get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                baseline_done INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                frames_used INTEGER,
                stability_score INTEGER,
                saved_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                session_id TEXT NOT NULL UNIQUE,
                start_time TEXT NOT NULL,
                end_time TEXT,
                baseline_used INTEGER NOT NULL DEFAULT 0,
                duration_s REAL NOT NULL DEFAULT 0,
                total_frames INTEGER NOT NULL DEFAULT 0,
                good_pct REAL NOT NULL DEFAULT 0,
                moderate_pct REAL NOT NULL DEFAULT 0,
                bad_pct REAL NOT NULL DEFAULT 0,
                total_alerts INTEGER NOT NULL DEFAULT 0,
                frames_json TEXT NOT NULL DEFAULT '[]',
                alerts_json TEXT NOT NULL DEFAULT '[]',
                summary_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_username_start
            ON sessions(username, start_time DESC)
        """)

        conn.commit()
    finally:
        conn.close()


def get_user_by_username(username):
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, baseline_done, created_at "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_user(username, password_hash):
    conn = get_conn()
    try:
        now = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO users (username, password_hash, baseline_done, created_at) "
            "VALUES (?, ?, 0, ?)",
            (username, password_hash, now),
        )
        conn.commit()
    finally:
        conn.close()


def mark_baseline_done(username):
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE users SET baseline_done = 1 WHERE username = ?",
            (username,),
        )
        conn.commit()
    finally:
        conn.close()


def save_baseline(username, metrics, frames_used=None, stability_score=None):
    conn = get_conn()
    try:
        # Keep one current baseline per user. Existing historical users/data
        # are preserved; saving a new calibration replaces that user's old one.
        conn.execute("DELETE FROM baselines WHERE username = ?", (username,))
        conn.execute(
            "INSERT INTO baselines "
            "(username, metrics_json, frames_used, stability_score, saved_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                username,
                json.dumps(metrics),
                frames_used,
                stability_score,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_baseline(username):
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id, username, metrics_json, frames_used, stability_score, saved_at "
            "FROM baselines WHERE username = ? ORDER BY id DESC LIMIT 1",
            (username,),
        ).fetchone()
        if not row:
            return None

        data = dict(row)
        try:
            data["metrics"] = json.loads(data.pop("metrics_json"))
        except (TypeError, json.JSONDecodeError):
            data["metrics"] = None
        return data
    finally:
        conn.close()


def save_session(username, posture_session):
    """Store one completed monitoring session for exactly one user."""
    summary = posture_session.get("summary") or {}
    frames = posture_session.get("frames") or []
    alerts = posture_session.get("alerts") or []

    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (
                username, session_id, start_time, end_time, baseline_used,
                duration_s, total_frames, good_pct, moderate_pct, bad_pct,
                total_alerts, frames_json, alerts_json, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                posture_session.get("session_id", ""),
                posture_session.get("start_time", ""),
                posture_session.get("end_time"),
                1 if posture_session.get("baseline_used") else 0,
                float(summary.get("duration_s", 0) or 0),
                int(summary.get("total_frames", len(frames)) or 0),
                float(summary.get("good_pct", 0) or 0),
                float(summary.get("moderate_pct", 0) or 0),
                float(summary.get("bad_pct", 0) or 0),
                int(summary.get("total_alerts", len(alerts)) or 0),
                json.dumps(frames),
                json.dumps(alerts),
                json.dumps(summary),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_sessions(username):
    """Return only sessions belonging to the requested username."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, username, session_id, start_time, end_time,
                   baseline_used, duration_s, total_frames,
                   good_pct, moderate_pct, bad_pct, total_alerts,
                   frames_json, alerts_json, summary_json
            FROM sessions
            WHERE username = ?
            ORDER BY start_time DESC, id DESC
            """,
            (username,),
        ).fetchall()

        out = []
        for row in rows:
            item = dict(row)
            try:
                item["frames"] = json.loads(item.pop("frames_json"))
            except (TypeError, json.JSONDecodeError):
                item["frames"] = []
            try:
                item["alerts"] = json.loads(item.pop("alerts_json"))
            except (TypeError, json.JSONDecodeError):
                item["alerts"] = []
            try:
                item["summary"] = json.loads(item.pop("summary_json"))
            except (TypeError, json.JSONDecodeError):
                item["summary"] = {}
            out.append(item)
        return out
    finally:
        conn.close()

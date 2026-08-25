"""
Database layer for the fraud detection application.

Uses SQLite through Python's built-in sqlite3 module. SQLite was chosen over
MySQL because it requires no server installation, no configuration and no
credentials -- the entire database is a single file that travels with the
project. For a system of this size the two are functionally equivalent, and
the SQL used here is standard, so migrating to MySQL later would require
changing only the connection function.

Two tables:
  users       -- accounts that can log into the dashboard
  predictions -- an audit log of every transaction scored by the system

The predictions table is what makes this a system rather than a script. A real
fraud engine must be able to answer 'why was this transaction blocked, and
when?' months after the event, which requires the decision to be persisted.
"""

import sqlite3
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = Path(__file__).resolve().parent / "fraud_system.db"


def get_connection():
    """
    Open a connection to the database.

    row_factory is set to sqlite3.Row so that query results can be accessed by
    column name rather than by numeric index, which keeps the calling code
    readable and resistant to column reordering.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the tables if they do not already exist, and seed a demo user."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            full_name     TEXT,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER,
            amount       REAL    NOT NULL,
            time_seconds REAL    NOT NULL,
            probability  REAL    NOT NULL,
            is_fraud     INTEGER NOT NULL,
            risk_band    TEXT    NOT NULL,
            threshold    REAL    NOT NULL,
            actual_class INTEGER,          -- known label when scoring a sample
            source       TEXT,             -- 'manual' or 'sample'
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Seed a single demo account so the application is usable immediately.
    # Passwords are never stored in plain text -- only a salted hash, so the
    # original password cannot be recovered from the database file.
    cur.execute("SELECT COUNT(*) AS n FROM users")
    if cur.fetchone()["n"] == 0:
        cur.execute(
            "INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("admin123"), "System Administrator"),
        )

    conn.commit()
    conn.close()


def verify_user(username, password):
    """Return the user row if the credentials are valid, otherwise None."""
    conn = get_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if user and check_password_hash(user["password_hash"], password):
        return user
    return None


def create_user(username, password, full_name=""):
    """Register a new account. Returns False if the username already exists."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), full_name),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Raised by the UNIQUE constraint on username.
        return False
    finally:
        conn.close()


def log_prediction(user_id, amount, time_seconds, result,
                   actual_class=None, source="manual"):
    """Persist one scoring decision to the audit log."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO predictions
            (user_id, amount, time_seconds, probability, is_fraud,
             risk_band, threshold, actual_class, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, amount, time_seconds,
        result["probability"], int(result["is_fraud"]),
        result["risk_band"], result["threshold"],
        actual_class, source,
    ))
    conn.commit()
    conn.close()


def recent_predictions(limit=25):
    """Most recent scoring decisions, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows


def dashboard_stats():
    """Aggregate counts used by the dashboard summary cards."""
    conn = get_connection()
    stats = conn.execute("""
        SELECT
            COUNT(*)                                   AS total,
            COALESCE(SUM(is_fraud), 0)                 AS flagged,
            COALESCE(SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END), 0) AS cleared,
            COALESCE(AVG(probability), 0)              AS avg_probability
        FROM predictions
    """).fetchone()
    conn.close()
    return stats

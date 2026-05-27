"""
User authentication module — registration, login, admin panel.
Uses SQLite + hashed passwords (SHA-256 salted).
Source: Student + AI collaboration.
"""
import sqlite3
import hashlib
import os
import secrets
from datetime import datetime
from typing import Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist. Seed admin account."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',   -- 'user' | 'admin'
            email TEXT,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT,
            intent TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS upload_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            filename TEXT NOT NULL,
            rows_count INTEGER,
            columns_count INTEGER,
            file_size_kb REAL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()

    # Seed default admin if not exists
    existing = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
    if not existing:
        salt = secrets.token_hex(16)
        pwd_hash = _hash_password("admin123", salt)
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role, email, created_at) VALUES (?,?,?,?,?,?)",
            ("admin", pwd_hash, salt, "admin", "admin@dataqa.local", datetime.now().isoformat()),
        )
        conn.commit()

    conn.close()


def _hash_password(password: str, salt: str) -> str:
    """SHA-256 salted hash."""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def register_user(username: str, password: str, email: str = "") -> Tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    username = username.strip().lower()
    if len(username) < 3:
        return False, "用户名至少需要 3 个字符"
    if len(password) < 6:
        return False, "密码至少需要 6 个字符"
    if not username.isalnum():
        return False, "用户名只能包含字母和数字"

    conn = _get_conn()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        conn.close()
        return False, "用户名已存在"

    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role, email, created_at) VALUES (?,?,?,?,?,?)",
            (username, pwd_hash, salt, "user", email, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        return True, "注册成功！请登录。"
    except Exception as e:
        conn.close()
        return False, f"注册失败: {str(e)}"


def login_user(username: str, password: str) -> Tuple[bool, str, Optional[dict]]:
    """Authenticate a user. Returns (success, message, user_dict)."""
    username = username.strip().lower()
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()

    if not row:
        conn.close()
        return False, "用户名或密码错误", None

    pwd_hash = _hash_password(password, row["salt"])
    if not secrets.compare_digest(pwd_hash, row["password_hash"]):
        conn.close()
        return False, "用户名或密码错误", None

    # Update last login
    conn.execute(
        "UPDATE users SET last_login = ? WHERE id = ?",
        (datetime.now().isoformat(), row["id"]),
    )
    conn.commit()

    user = {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "email": row["email"],
        "created_at": row["created_at"],
    }
    conn.close()
    return True, "登录成功", user


def is_admin(user: dict) -> bool:
    return user and user.get("role") == "admin"


def get_all_users() -> list:
    """Admin: get all registered users."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, username, role, email, created_at, last_login FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_query_logs() -> list:
    """Admin: get query logs from all users."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM query_log ORDER BY timestamp DESC LIMIT 200"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_query_logs(username: str) -> list:
    """Get query logs for a specific user."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM query_log WHERE username = ? ORDER BY timestamp DESC LIMIT 100",
        (username,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_query(username: str, query: str, answer: str, intent: str):
    """Record a QA interaction."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO query_log (username, query, answer, intent, timestamp) VALUES (?,?,?,?,?)",
        (username, query, answer[:2000], intent, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def delete_user(user_id: int) -> Tuple[bool, str]:
    """Admin: delete a user (cannot delete self)."""
    conn = _get_conn()
    row = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        conn.close()
        return False, "用户不存在"
    if row["username"] == "admin":
        conn.close()
        return False, "不能删除默认管理员"
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.execute("DELETE FROM query_log WHERE username = ?", (row["username"],))
    conn.commit()
    conn.close()
    return True, f"已删除用户 {row['username']}"


def log_upload(username: str, filename: str, rows_count: int, columns_count: int, file_size_kb: float):
    """Record a data upload."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO upload_log (username, filename, rows_count, columns_count, file_size_kb, timestamp) VALUES (?,?,?,?,?,?)",
        (username, filename, rows_count, columns_count, file_size_kb, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_upload_logs(username: str = None) -> list:
    """Get upload logs. If username is None, get all (admin)."""
    conn = _get_conn()
    if username:
        rows = conn.execute(
            "SELECT * FROM upload_log WHERE username = ? ORDER BY timestamp DESC LIMIT 200",
            (username,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM upload_log ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def change_password(username: str, old_pwd: str, new_pwd: str) -> Tuple[bool, str]:
    """User: change own password."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        conn.close()
        return False, "用户不存在"

    old_hash = _hash_password(old_pwd, row["salt"])
    if not secrets.compare_digest(old_hash, row["password_hash"]):
        conn.close()
        return False, "原密码错误"

    if len(new_pwd) < 6:
        conn.close()
        return False, "新密码至少需要 6 个字符"

    new_salt = secrets.token_hex(16)
    new_hash = _hash_password(new_pwd, new_salt)
    conn.execute(
        "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
        (new_hash, new_salt, username),
    )
    conn.commit()
    conn.close()
    return True, "密码修改成功"

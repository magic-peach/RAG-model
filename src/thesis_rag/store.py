"""Chat history: a small SQLite store behind the sidebar of saved chats.

SQLite (standard library, one file in ``data/chats.db``) is enough for one
user or a small demo, needs no server, and survives restarts. A host whose
disk is wiped on restart (Cloud Run) passes a ``libsql://`` URL instead: Turso
serves the same SQLite dialect over the network, so the SQL here is shared. Every chat
belongs to an ``owner``: a random id the browser generates once and sends
with each request. That is not authentication; it keeps visitors of the
public demo from seeing each other's chats without asking anyone to log in.

Messages keep the full answer object (sources, reasoning, mode) as JSON, so
an old chat re-renders exactly as it was answered.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS chats_owner ON chats(owner, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,          -- "user" | "assistant"
    content TEXT NOT NULL,
    payload TEXT,                -- assistant: the full Answer as JSON
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, id);
"""

MAX_TITLE = 80


def title_from(question: str) -> str:
    """First line of the first question, trimmed to a sidebar-friendly length."""
    text = " ".join(question.split())
    return text if len(text) <= 48 else text[:47].rsplit(" ", 1)[0] + "…"


def is_remote(location: str | Path) -> bool:
    """True for a libSQL/Turso URL, False for a local SQLite file."""
    return str(location).startswith(("libsql://", "https://", "http://"))


def _connect(location: str | Path, auth_token: str | None):
    if is_remote(location):
        # Optional dependency, only needed for the hosted demo: pip install ".[deploy]"
        import libsql

        return libsql.connect(str(location), auth_token=auth_token or "")
    path = Path(location)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _dicts(cursor) -> list[dict[str, Any]]:
    """Rows as dicts keyed by column name. Both drivers fill ``description``;
    only sqlite3 has ``row_factory``, so this is the portable way."""
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


class ChatStore:
    def __init__(self, location: str | Path, auth_token: str | None = None):
        self._conn = _connect(location, auth_token)
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        # One connection shared by the API's worker threads. Every statement
        # takes the lock: sqlite3 needs it for writes, and a remote libSQL
        # connection carries one stream that must not interleave requests.
        self._lock = threading.Lock()

    def _query(self, sql: str, args: tuple | list = ()) -> list[dict[str, Any]]:
        with self._lock:
            return _dicts(self._conn.execute(sql, args))

    def _owned(self, owner: str, chat_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM chats WHERE id = ? AND owner = ?", (chat_id, owner))
        return rows[0] if rows else None

    def list_chats(self, owner: str, query: str = "") -> list[dict]:
        sql = "SELECT id, title, created_at, updated_at FROM chats WHERE owner = ?"
        args: list = [owner]
        if query:
            sql += " AND title LIKE ?"
            args.append(f"%{query}%")
        return self._query(sql + " ORDER BY updated_at DESC LIMIT 200", args)

    def create_chat(self, owner: str, title: str = "New chat") -> dict:
        now = time.time()
        chat = {"id": uuid.uuid4().hex, "title": title[:MAX_TITLE], "created_at": now}
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO chats VALUES (?, ?, ?, ?, ?)",
                (chat["id"], owner, chat["title"], now, now),
            )
        return {**chat, "updated_at": now}

    def get_chat(self, owner: str, chat_id: str) -> dict | None:
        chat = self._owned(owner, chat_id)
        if chat is None:
            return None
        rows = self._query(
            "SELECT role, content, payload, created_at FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        )
        messages = [
            {
                "role": r["role"],
                "content": r["content"],
                "answer": json.loads(r["payload"]) if r["payload"] else None,
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        return {
            **{k: chat[k] for k in ("id", "title", "created_at", "updated_at")},
            "messages": messages,
        }

    def rename_chat(self, owner: str, chat_id: str, title: str) -> bool:
        title = " ".join(title.split())[:MAX_TITLE]
        if not title:
            return False
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE chats SET title = ? WHERE id = ? AND owner = ?", (title, chat_id, owner)
            )
        return cur.rowcount == 1

    def delete_chat(self, owner: str, chat_id: str) -> bool:
        if self._owned(owner, chat_id) is None:
            return False
        # Messages are deleted explicitly rather than through ON DELETE
        # CASCADE: foreign keys are a per-connection pragma in SQLite, and a
        # remote libSQL connection doesn't keep it between requests.
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            cur = self._conn.execute(
                "DELETE FROM chats WHERE id = ? AND owner = ?", (chat_id, owner)
            )
        return cur.rowcount == 1

    def add_exchange(self, owner: str, chat_id: str, question: str, answer: dict) -> None:
        """Store one question and its answer, and bump the chat to the top of the list."""
        if self._owned(owner, chat_id) is None:
            raise KeyError(chat_id)
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages (chat_id, role, content, payload, created_at) "
                "VALUES (?, 'user', ?, NULL, ?)",
                (chat_id, question, now),
            )
            self._conn.execute(
                "INSERT INTO messages (chat_id, role, content, payload, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (chat_id, answer.get("answer", ""), json.dumps(answer), now),
            )
            self._conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))

    def history(self, owner: str, chat_id: str, turns: int) -> list[tuple[str, str]]:
        """The last ``turns`` (question, answer) pairs, oldest first, for follow-ups."""
        chat = self.get_chat(owner, chat_id)
        if not chat or turns <= 0:
            return []
        msgs = chat["messages"]
        pairs = [
            (msgs[i]["content"], msgs[i + 1]["content"])
            for i in range(0, len(msgs) - 1, 2)
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant"
        ]
        return pairs[-turns:]

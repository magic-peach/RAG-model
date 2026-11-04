"""Chat history: a small SQLite store behind the sidebar of saved chats.

SQLite (standard library, one file in ``data/chats.db``) is enough for one
user or a small demo, needs no server, and survives restarts. Every chat
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


class ChatStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()  # one connection shared by the API's worker threads

    def _owned(self, owner: str, chat_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM chats WHERE id = ? AND owner = ?", (chat_id, owner)
        ).fetchone()

    def list_chats(self, owner: str, query: str = "") -> list[dict]:
        sql = "SELECT id, title, created_at, updated_at FROM chats WHERE owner = ?"
        args: list = [owner]
        if query:
            sql += " AND title LIKE ?"
            args.append(f"%{query}%")
        rows = self._conn.execute(sql + " ORDER BY updated_at DESC LIMIT 200", args).fetchall()
        return [dict(r) for r in rows]

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
        rows = self._conn.execute(
            "SELECT role, content, payload, created_at FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
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
        with self._lock, self._conn:
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

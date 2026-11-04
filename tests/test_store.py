from thesis_rag.store import ChatStore, title_from


def test_title_from_trims_long_questions():
    assert title_from("  Which   tool? ") == "Which tool?"
    long = "What were the filtering thresholds applied to SNPs in the variant calling pipeline"
    assert len(title_from(long)) <= 48 and title_from(long).endswith("…")


def test_store_round_trip_and_history(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    chat = store.create_chat("owner-1", "First")
    for i in range(3):
        store.add_exchange("owner-1", chat["id"], f"q{i}", {"answer": f"a{i}", "sources": []})
    assert store.history("owner-1", chat["id"], 2) == [("q1", "a1"), ("q2", "a2")]
    assert store.history("owner-2", chat["id"], 2) == []
    full = store.get_chat("owner-1", chat["id"])
    assert len(full["messages"]) == 6 and full["messages"][1]["answer"]["answer"] == "a0"


def test_store_survives_reopen_and_cascades_delete(tmp_path):
    path = tmp_path / "chats.db"
    chat = ChatStore(path).create_chat("o", "Kept")
    store = ChatStore(path)
    store.add_exchange("o", chat["id"], "q", {"answer": "a"})
    assert store.list_chats("o")[0]["title"] == "Kept"
    assert store.delete_chat("o", chat["id"])
    assert store.get_chat("o", chat["id"]) is None
    assert store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_rename_rejects_blank_titles(tmp_path):
    store = ChatStore(tmp_path / "c.db")
    chat = store.create_chat("o")
    assert not store.rename_chat("o", chat["id"], "   ")
    assert store.rename_chat("o", chat["id"], "Ok")

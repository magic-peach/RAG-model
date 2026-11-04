"""API tests with the LLM, retriever and index build replaced by fakes."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from thesis_rag.api import create_app
from thesis_rag.config import Settings
from thesis_rag.generate import ThesisQA

from .fakes import fake_llm
from .test_analyze import CSV
from .test_generate import StubRetriever, _passage


def _write_manifest(settings):
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    (settings.index_dir / "manifest.json").write_text(
        json.dumps({"title": "A Study of Mulberry", "source": "thesis.pdf", "chunk_counts": {}})
    )


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


def make_client(settings, answers=("Yield fell [Passage 1].",), build_fn=None):
    passages = [_passage(1, "4.3", 62), _passage(2, "4.5", 66)]

    def qa_factory(s):
        return ThesisQA(
            s, index=object(), retriever=StubRetriever(passages), llm=fake_llm(list(answers))
        )

    app = create_app(
        settings, qa_factory=qa_factory, build_fn=build_fn, check_ollama=lambda url: "ok"
    )
    return TestClient(app)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_healthz(settings):
    assert make_client(settings).get("/healthz").json() == {"status": "ok"}


def test_status_before_and_after_indexing(settings):
    client = make_client(settings)
    s = client.get("/status").json()
    assert s["indexed"] is False and s["document"] is None and s["ollama"] == "ok"
    _write_manifest(settings)
    s = client.get("/status").json()
    assert s["indexed"] is True and s["document"]["title"] == "A Study of Mulberry"


def test_ask_requires_index(settings):
    res = make_client(settings).post("/ask", json={"question": "Anything?"})
    assert res.status_code == 409


def test_ask_validates_question(settings):
    _write_manifest(settings)
    assert make_client(settings).post("/ask", json={"question": ""}).status_code == 422


def test_ask_streams_tokens_then_answer(settings):
    _write_manifest(settings)
    res = make_client(settings).post("/ask", json={"question": "How did yield change?"})
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(res.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "mode" and kinds[-2:] == ["answer", "done"]
    streamed = "".join(d["text"] for k, d in events if k == "token")
    assert streamed == "Yield fell [Passage 1]."
    answer = events[-2][1]
    # The final answer is normalised, so the chip matches the source's citation.
    assert answer["answer"] == "Yield fell [Ch.4 §4.3, p.62]."
    assert answer["cited"] == [1] and answer["sources"][0]["citation"] == "[Ch.4 §4.3, p.62]"


def test_ask_reports_llm_failure_in_stream(settings):
    _write_manifest(settings)

    class Down:
        def stream(self, *_):
            raise ConnectionError("Connection refused")

    def qa_factory(s):
        return ThesisQA(
            s, index=object(), retriever=StubRetriever([_passage(1, "4.3", 62)]), llm=Down()
        )

    app = create_app(settings, qa_factory=qa_factory, check_ollama=lambda url: "offline")
    events = parse_sse(TestClient(app).post("/ask", json={"question": "x"}).text)
    assert events[-1][0] == "error" and "Ollama" in events[-1][1]["message"]


def test_ingest_upload_runs_in_background(settings):
    calls = []

    def fake_build(path, s, force=False, progress=None):
        progress("Parsing PDF", 0.5)
        calls.append((path.name, force))
        _write_manifest(s)

    client = make_client(settings, build_fn=fake_build)
    res = client.post(
        "/ingest", files={"file": ("my thesis.pdf", b"%PDF-1.4 fake", "application/pdf")}
    )
    assert res.status_code == 202 and res.json()["source"] == "my_thesis.pdf"
    for _ in range(50):
        if client.get("/status").json()["ingest"]["state"] == "done":
            break
        time.sleep(0.05)
    status = client.get("/status").json()
    assert status["ingest"]["state"] == "done" and status["indexed"]
    assert calls == [("my_thesis.pdf", False)]
    assert (settings.data_dir / "raw" / "my_thesis.pdf").read_bytes() == b"%PDF-1.4 fake"


def test_ingest_rejects_non_pdf_and_missing_input(settings):
    client = make_client(settings)
    res = client.post("/ingest", files={"file": ("x.pdf", b"hello", "application/pdf")})
    assert res.status_code == 400
    assert client.post("/ingest").status_code == 400
    assert client.post("/ingest", data={"path": "/no/such.pdf"}).status_code == 400


def test_ingest_failure_is_reported(settings):
    def broken(path, s, force=False, progress=None):
        raise RuntimeError("bad PDF")

    client = make_client(settings, build_fn=broken)
    client.post("/ingest", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
    for _ in range(50):
        job = client.get("/status").json()["ingest"]
        if job["state"] != "running":
            break
        time.sleep(0.05)
    assert job["state"] == "error" and job["error"] == "bad PDF"


def test_analyze_stats_only_without_index(settings):
    res = make_client(settings).post("/analyze", files={"file": ("d.csv", CSV, "text/csv")})
    body = res.json()
    assert res.status_code == 200 and "explanation" not in body
    assert body["stats"]["group_column"] == "genotype" and "note" in body


def test_analyze_with_explanation(settings):
    _write_manifest(settings)
    client = make_client(settings, answers=["V1 is higher [Passage 2]. Limitations: small n."])
    res = client.post(
        "/analyze", files={"file": ("d.csv", CSV, "text/csv")}, data={"group_col": "genotype"}
    )
    body = res.json()
    assert res.status_code == 200 and body["cited"] == [2]
    assert body["label"].startswith("Decision support")


def test_analyze_bad_csv(settings):
    res = make_client(settings).post("/analyze", files={"file": ("d.csv", b"", "text/csv")})
    assert res.status_code == 400


def test_serves_web_ui(settings):
    res = make_client(settings).get("/")
    assert res.status_code == 200 and "Ask the thesis" in res.text
    assert make_client(settings).get("/app.js").status_code == 200


def test_ollama_status_offline_for_closed_port():
    from thesis_rag.api import _ollama_status

    assert _ollama_status("http://127.0.0.1:9", timeout=1) == "offline"


CLIENT = {"X-Client-Id": "browser-1234abcd"}
OTHER = {"X-Client-Id": "browser-9999zzzz"}


def test_ask_saves_a_chat_and_uses_its_history(settings):
    _write_manifest(settings)
    client = make_client(settings, answers=["GATK [Passage 1].\nReasoning:\n- Passage 1."])
    events = parse_sse(client.post("/ask", json={"question": "Which tool?"}, headers=CLIENT).text)
    assert events[0][0] == "chat"
    chat_id = events[0][1]["id"]
    assert events[0][1]["title"] == "Which tool?"
    answer = dict(events)["answer"]
    assert answer["chat_id"] == chat_id and answer["reasoning"] == "- Passage 1."

    # A follow-up in the same chat is appended, not a new chat.
    events = parse_sse(
        client.post("/ask", json={"question": "why?", "chat_id": chat_id}, headers=CLIENT).text
    )
    assert events[0][0] == "mode"
    chat = client.get(f"/chats/{chat_id}", headers=CLIENT).json()
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant", "user", "assistant"]
    assert chat["messages"][1]["answer"]["sources"][0]["citation"] == "[Ch.4 §4.3, p.62]"


def test_chats_list_rename_delete_and_owner_scoping(settings):
    _write_manifest(settings)
    client = make_client(settings)
    chat = client.post("/chats", headers=CLIENT).json()
    assert client.get("/chats", headers=CLIENT).json()[0]["id"] == chat["id"]
    assert client.get("/chats", headers=OTHER).json() == []  # another browser sees nothing
    assert client.get(f"/chats/{chat['id']}", headers=OTHER).status_code == 404

    res = client.patch(
        f"/chats/{chat['id']}", json={"title": "  Variant  calling "}, headers=CLIENT
    )
    assert res.json()["title"] == "Variant calling"
    assert client.get("/chats", headers=CLIENT, params={"q": "variant"}).json()[0]["title"] == (
        "Variant calling"
    )
    assert (
        client.patch(f"/chats/{chat['id']}", json={"title": "x"}, headers=OTHER).status_code == 404
    )

    assert client.delete(f"/chats/{chat['id']}", headers=OTHER).status_code == 404
    assert client.delete(f"/chats/{chat['id']}", headers=CLIENT).status_code == 204
    assert client.get("/chats", headers=CLIENT).json() == []


def test_chats_require_a_client_id(settings):
    client = make_client(settings)
    assert client.get("/chats").status_code == 400
    assert client.get("/chats", headers={"X-Client-Id": "bad id!"}).status_code == 400


def test_ask_unknown_chat_is_404(settings):
    _write_manifest(settings)
    res = make_client(settings).post(
        "/ask", json={"question": "x", "chat_id": "nope"}, headers=CLIENT
    )
    assert res.status_code == 404


def test_ingest_can_be_disabled(settings):
    client = make_client(settings.model_copy(update={"allow_ingest": False}))
    res = client.post("/ingest", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
    assert res.status_code == 403
    assert client.get("/status").json()["ingest_enabled"] is False


def test_ask_reports_missing_model_configuration(settings):
    _write_manifest(settings)

    def broken_factory(s):
        raise RuntimeError("The api_key client option must be set")

    client = TestClient(
        create_app(settings, qa_factory=broken_factory, check_ollama=lambda u: "ok")
    )
    res = client.post("/ask", json={"question": "x"})
    assert res.status_code == 503 and "missing API key" in res.json()["detail"]

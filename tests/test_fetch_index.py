import sys
import types

from thesis_rag import fetch_index
from thesis_rag.config import Settings


def test_fetch_skips_when_index_exists(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path)
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / "manifest.json").write_text("{}")
    monkeypatch.setattr(fetch_index, "get_settings", lambda: settings)
    called = []
    fake = types.SimpleNamespace(snapshot_download=lambda **kw: called.append(kw))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    fetch_index.fetch("user/index")
    assert called == []


def test_fetch_downloads_into_data_dir(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(fetch_index, "get_settings", lambda: settings)

    def fake_download(**kw):
        assert kw["repo_type"] == "dataset" and kw["local_dir"] == tmp_path
        settings.index_dir.mkdir(parents=True)
        (settings.index_dir / "manifest.json").write_text("{}")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_download)
    )
    fetch_index.fetch("user/index", token="t")
    assert (settings.index_dir / "manifest.json").is_file()

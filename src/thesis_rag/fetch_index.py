"""Download a prebuilt index for the hosted demo.

    RAG_INDEX_REPO=user/thesis-rag-index HF_TOKEN=... python -m thesis_rag.fetch_index

The hosted demo cannot build the index itself (no Ollama for embeddings or
summaries, and uploads are disabled), so it starts from the index built
locally. That index contains the thesis text, so it lives in a *private*
Hugging Face dataset and is fetched at startup with a read token; visitors
get answers, not a downloadable copy of the document.

Dataset layout (mirrors ``data/``)::

    index/structured-600-72/{manifest.json, docstore.json, chroma/...}
    knowledge_card.json
"""

from __future__ import annotations

import logging
import os
import sys

from .config import get_settings

log = logging.getLogger(__name__)


def fetch(repo_id: str, token: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    settings = get_settings()
    if (settings.index_dir / "manifest.json").is_file():
        log.info("Index already present at %s", settings.index_dir)
        return
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=settings.data_dir,
        token=token,
        allow_patterns=["index/**", "knowledge_card.json"],
    )
    if not (settings.index_dir / "manifest.json").is_file():
        raise SystemExit(f"{repo_id} has no index for {settings.index_dir.name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo_id = os.environ.get("RAG_INDEX_REPO", "")
    if not repo_id:
        print("RAG_INDEX_REPO is not set; nothing to fetch.", file=sys.stderr)
        return
    fetch(repo_id, os.environ.get("HF_TOKEN"))
    print(f"Index ready from {repo_id}")


if __name__ == "__main__":
    main()

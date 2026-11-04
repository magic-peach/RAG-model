"""Publish the hosted demo to Hugging Face Spaces.

    hf auth login                      # once, with a write token
    python deploy/huggingface/publish.py

What it does:

1. Uploads the locally built index (``data/index/structured-600-72``) and
   ``data/knowledge_card.json`` to a **private** dataset,
   ``<user>/thesis-rag-index``. The index contains the thesis text, so it is
   never put in a public repository.
2. Creates or updates the Docker Space ``<user>/thesis-rag`` with the app
   code, ``deploy/huggingface/Dockerfile`` and the Space card, and sets
   ``RAG_INDEX_REPO`` so the Space fetches the index at startup.

Two secrets must then be added in the Space settings (never committed):
``GROQ_API_KEY`` for the model and ``HF_TOKEN`` (read access) for the
private index.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INDEX_DIR = "index/structured-600-72"


def space_files() -> list[CommitOperationAdd]:
    ops = [
        CommitOperationAdd("Dockerfile", str(HERE / "Dockerfile")),
        CommitOperationAdd("README.md", str(HERE / "SPACE_README.md")),
        CommitOperationAdd("config.yaml", str(ROOT / "config.yaml")),
        CommitOperationAdd("pyproject.toml", str(ROOT / "pyproject.toml")),
    ]
    for folder in ("src", "web"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("."):
                ops.append(CommitOperationAdd(str(path.relative_to(ROOT)), str(path)))
    return ops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", default="thesis-rag", help="Space name")
    parser.add_argument("--dataset", default="thesis-rag-index", help="private index dataset")
    parser.add_argument("--skip-index", action="store_true", help="only update the app code")
    args = parser.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    dataset_id, space_id = f"{user}/{args.dataset}", f"{user}/{args.space}"

    if not args.skip_index:
        if not (ROOT / "data" / INDEX_DIR / "manifest.json").is_file():
            raise SystemExit(
                "No local index; run `python -m thesis_rag.ingest data/raw/thesis.pdf`"
            )
        api.create_repo(dataset_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=dataset_id,
            repo_type="dataset",
            folder_path=str(ROOT / "data"),
            allow_patterns=[f"{INDEX_DIR}/**", "knowledge_card.json"],
            commit_message="Update index",
        )
        print(f"Index uploaded to private dataset {dataset_id}")

    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_variable(space_id, "RAG_INDEX_REPO", dataset_id)
    api.create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=space_files(),
        commit_message="Deploy Thesis RAG",
    )
    subdomain = space_id.lower().replace("/", "-").replace("_", "-")
    print(f"Space: https://huggingface.co/spaces/{space_id}")
    print(f"App:   https://{subdomain}.hf.space")
    print("Add GROQ_API_KEY and HF_TOKEN (read) under Settings → Variables and secrets.")


if __name__ == "__main__":
    main()

"""Application settings.

Settings are resolved in this order (highest priority first):

1. Keyword arguments passed to ``Settings(...)`` (used by tests and the eval
   harness to run several configurations side by side).
2. Environment variables prefixed with ``RAG_`` (e.g. ``RAG_TOP_K=8``).
3. The YAML file named by ``RAG_CONFIG`` (default: ``config.yaml`` in the
   working directory).
4. The defaults declared below.

Keeping every tunable in one typed object means the eval harness can describe
a configuration as a plain dict, and the API can report exactly which settings
produced an answer.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    # --- Models ---
    # Locally everything runs on Ollama. The hosted demo uses Groq for generation
    # (the same Llama 3.1 8B, much faster) because a free CPU host cannot run an
    # 8B model, and fastembed for query embeddings: its ONNX export of
    # nomic-embed-text v1.5 gives the same vectors as Ollama's (cosine 1.0), so
    # it can search an index that was built with Ollama.
    llm_provider: Literal["ollama", "groq"] = "ollama"
    embedding_provider: Literal["ollama", "fastembed"] = "ollama"
    fastembed_model: str = "nomic-ai/nomic-embed-text-v1.5"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    groq_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "nomic-embed-text"
    num_ctx: int = Field(8192, description="Ollama context window for generation")
    temperature: float = 0.1
    keep_alive: str = Field("30m", description="How long Ollama keeps the model loaded")
    max_answer_tokens: int = 600

    # --- Chunking ---
    chunk_size: int = Field(600, description="Target tokens per child chunk")
    chunk_overlap: int = Field(72, description="Token overlap between neighbouring chunks")
    max_parent_tokens: int = Field(
        1500,
        description="Sibling sections are merged into one parent only if they fit this budget",
    )
    structured_chunking: bool = Field(
        True, description="False reproduces the v1 page-based splitter (eval baseline)"
    )

    # --- Retrieval ---
    retrieve_k: int = 20
    top_k: int = Field(3, description="Passages sent to the LLM (fewer = faster prompts)")
    rrf_k: int = 60
    use_bm25: bool = True
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_multi_query: bool = False
    num_query_variants: int = 3
    use_parent_context: bool = True
    max_passage_tokens: int = Field(
        700,
        description="A parent longer than this is not sent to the LLM; the matched chunk is",
    )

    # --- Ingestion ---
    summarize_sections: bool = True

    # --- Answers ---
    allow_general_knowledge: bool = Field(
        True,
        description="Answer off-thesis questions from general knowledge, labelled as such",
    )
    history_turns: int = Field(2, description="Previous Q&A pairs sent with a follow-up")

    # --- Storage ---
    data_dir: Path = Path("data")
    allow_ingest: bool = Field(True, description="False disables POST /ingest (public demo)")

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k > self.retrieve_k:
            raise ValueError("top_k cannot exceed retrieve_k")
        return self

    @property
    def index_dir(self) -> Path:
        """Where Chroma, the docstore and the ingest cache live.

        The directory name includes the chunking parameters, so changing
        chunk_size or switching to the baseline splitter builds a separate
        index instead of silently mixing chunks from two configurations.
        """
        mode = "structured" if self.structured_chunking else "baseline"
        return self.data_dir / "index" / f"{mode}-{self.chunk_size}-{self.chunk_overlap}"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = Path(os.environ.get("RAG_CONFIG", "config.yaml"))
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if yaml_path.is_file():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        return tuple(sources)


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton (the API and CLI share one instance)."""
    return Settings()

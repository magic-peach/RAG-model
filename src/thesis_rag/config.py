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
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    num_ctx: int = Field(8192, description="Ollama context window for generation")
    temperature: float = 0.1

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
    top_k: int = 5
    rrf_k: int = 60
    use_bm25: bool = True
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    use_multi_query: bool = False
    num_query_variants: int = 3
    use_parent_context: bool = True
    max_passage_tokens: int = Field(
        1200,
        description="A parent longer than this is not sent to the LLM; the matched chunk is",
    )

    # --- Ingestion ---
    summarize_sections: bool = True

    # --- Storage ---
    data_dir: Path = Path("data")

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

# Cloud Run image for the public demo. Generation runs on Groq, query
# embeddings on fastembed. The image goes to a private Artifact Registry
# repository, so the prebuilt index is baked into it; it is never pushed
# anywhere public. Saved chats go to Turso (RAG_CHATS_DB) because a Cloud Run
# instance's disk is wiped when it shuts down.
FROM python:3.11-slim

RUN useradd --create-home --uid 1000 user

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    RAG_LLM_PROVIDER=groq \
    RAG_EMBEDDING_PROVIDER=fastembed \
    RAG_ALLOW_INGEST=false \
    RAG_DATA_DIR=/home/user/data \
    RAG_CHATS_DB=/home/user/state/chats.db \
    RAG_WEB_DIR=/home/user/app/web

WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install ".[deploy]"
COPY --chown=user web ./web
COPY --chown=user config.yaml ./
COPY --chown=user data/index/structured-600-72 /home/user/data/index/structured-600-72
COPY --chown=user data/knowledge_card.json /home/user/data/knowledge_card.json

RUN mkdir -p /home/user/state && chown user /home/user/state
USER user
# Bake the reranker and the embedding model into the image so a cold start
# does not download them.
RUN python -c "from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2'); TextEmbedding('nomic-ai/nomic-embed-text-v1.5')"
# The models are in the image now, so skip Hugging Face's online checks at
# runtime, and load everything before the port opens (see RAG_WARMUP in api.py).
ENV HF_HUB_OFFLINE=1 \
    RAG_WARMUP=startup

EXPOSE 8080
CMD ["uvicorn", "thesis_rag.api:app", "--host", "0.0.0.0", "--port", "8080"]

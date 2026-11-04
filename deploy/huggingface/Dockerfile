# Hugging Face Space image for the public demo (Docker SDK, port 7860).
# Generation runs on Groq (same Llama 3.1 8B as locally), query embeddings on
# fastembed, and the index is downloaded at startup from a private dataset.
FROM python:3.11-slim

RUN useradd --create-home --uid 1000 user

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    RAG_LLM_PROVIDER=groq \
    RAG_EMBEDDING_PROVIDER=fastembed \
    RAG_ALLOW_INGEST=false \
    RAG_DATA_DIR=/home/user/data \
    RAG_WEB_DIR=/home/user/app/web

WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install ".[deploy]"
COPY --chown=user web ./web
COPY --chown=user config.yaml ./

USER user
# Bake the reranker and the embedding model into the image so a cold start
# only has to fetch the index.
RUN python -c "from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2'); TextEmbedding('nomic-ai/nomic-embed-text-v1.5')"

EXPOSE 7860
CMD ["sh", "-c", "python -m thesis_rag.fetch_index && uvicorn thesis_rag.api:app --host 0.0.0.0 --port 7860"]

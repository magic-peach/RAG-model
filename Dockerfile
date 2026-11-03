# Thesis RAG API + web UI. Ollama runs in its own container (docker-compose.yml)
# or natively on the host.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_WEB_DIR=/app/web \
    RAG_DATA_DIR=/app/data \
    HF_HOME=/app/hf-cache

WORKDIR /app

# CPU-only torch first (the default wheel bundles CUDA and is several GB).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY web ./web
COPY config.yaml ./

RUN useradd --create-home app && mkdir -p /app/data /app/hf-cache && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

CMD ["uvicorn", "thesis_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

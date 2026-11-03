"""Test doubles that stand in for Ollama (no models are downloaded in tests or CI)."""

import hashlib
import math
import re

from langchain_core.language_models.fake_chat_models import FakeListChatModel


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: similar words -> similar vectors."""

    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def fake_llm(responses: list[str]) -> FakeListChatModel:
    return FakeListChatModel(responses=responses)


class FakeReranker:
    """Scores a passage by how many query words it contains (no model download)."""

    def __init__(self):
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        return [
            len(words & set(re.findall(r"[a-z0-9]+", t.lower()))) / (len(words) or 1) for t in texts
        ]

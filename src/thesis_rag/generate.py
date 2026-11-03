"""LLM calls: hierarchical summaries at ingest time, grounded answers at query time.

All generation goes through :func:`get_llm`, a ``ChatOllama`` client for the
local Llama 3.1 model. Low temperature (0.1) because we want faithful
extraction, not creativity; ``num_ctx`` is raised from Ollama's small default
so several retrieved sections fit in one prompt.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .config import Settings

log = logging.getLogger(__name__)


def get_llm(settings: Settings, **overrides) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    params = {
        "model": settings.llm_model,
        "base_url": settings.ollama_base_url,
        "temperature": settings.temperature,
        "num_ctx": settings.num_ctx,
    }
    params.update(overrides)
    return ChatOllama(**params)


def _complete(llm: BaseChatModel, system: str, user: str) -> str:
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return str(msg.content).strip()


# ---------------------------------------------------------------------------
# Hierarchical summaries (ingest time)
# ---------------------------------------------------------------------------
#
# Broad questions ("what is the main contribution?") have no single passage
# that answers them; similarity search returns a random-looking set of
# fragments. Summaries give those questions something to retrieve. They are
# built bottom-up (section -> chapter -> thesis) so each prompt stays small:
# 80 sections would not fit in one 8k context, but 5 chapter summaries do.

SUMMARY_PROMPT_VERSION = "v1"  # bump to invalidate cached summaries

SUMMARY_SYSTEM = (
    "You summarise sections of a PhD thesis for a search index. Write plain prose, "
    "no bullet points, no preamble such as 'This section'. Keep every specific name, "
    "method, tool, genotype, trait and number exactly as written. Do not add facts "
    "that are not in the text."
)


def summarize_section(llm: BaseChatModel, header: str, text: str) -> str:
    user = f"Section: {header}\n\n{text}\n\nSummarise this section in 2-4 sentences."
    return _complete(llm, SUMMARY_SYSTEM, user)


def summarize_chapter(llm: BaseChatModel, chapter: str, section_summaries: list[str]) -> str:
    joined = "\n".join(f"- {s}" for s in section_summaries)
    user = (
        f"{chapter}\n\nSummaries of its sections, in order:\n{joined}\n\n"
        "Summarise the whole chapter in one paragraph of 4-6 sentences."
    )
    return _complete(llm, SUMMARY_SYSTEM, user)


def summarize_thesis(
    llm: BaseChatModel, title: str, abstract: str, chapter_summaries: list[str]
) -> str:
    joined = "\n".join(f"- {s}" for s in chapter_summaries)
    user = (
        f"Thesis title: {title}\n\nAbstract:\n{abstract}\n\nChapter summaries:\n{joined}\n\n"
        "Write one paragraph (5-8 sentences) stating the thesis's main aim, objectives, "
        "methods, key findings and overall contribution."
    )
    return _complete(llm, SUMMARY_SYSTEM, user)

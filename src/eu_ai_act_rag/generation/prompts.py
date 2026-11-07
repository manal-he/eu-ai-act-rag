"""Prompt contract between the answerer and every LLM provider (including the mock)."""

from __future__ import annotations

import re

from eu_ai_act_rag.schemas import Lang, SearchHit

ABSTAIN_SENTINEL = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You are a careful assistant answering questions about the EU Artificial \
Intelligence Act (Regulation (EU) 2024/1689) and the GDPR (Regulation (EU) 2016/679).

Rules:
1. Use ONLY the numbered context passages. Do not rely on prior knowledge of the law.
2. Every sentence that states a legal rule must end with one or more citation markers such as \
[1] or [2][3], referring to the passage numbers. Never cite a number that is not in the context.
3. If the passages do not contain enough information to answer, reply with exactly \
{ABSTAIN_SENTINEL} and nothing else.
4. Answer in {{language}}. Be concise (at most ~150 words), precise about who the obligation \
applies to (provider, deployer, controller, processor...) and about dates, thresholds and amounts.
5. You are not giving legal advice; do not speculate beyond the text."""

LANG_NAMES: dict[Lang, str] = {"en": "English", "fr": "French"}

_PASSAGE = re.compile(r"^\[(\d+)\] \((.*?)\)\n(.*?)(?=^\[\d+\] \(|^=== QUESTION ===)", re.S | re.M)


def system_prompt(lang: Lang) -> str:
    return SYSTEM_PROMPT.replace("{language}", LANG_NAMES[lang])


def _label(h: SearchHit) -> str:
    c = h.chunk
    return f"{c.citation_label} - {c.title}" if c.title else c.citation_label


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    blocks = [f"[{i}] ({_label(h)})\n{h.chunk.text}\n" for i, h in enumerate(hits, 1)]
    return "=== CONTEXT ===\n" + "\n".join(blocks) + f"\n=== QUESTION ===\n{question}\n"


def parse_prompt(prompt: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Inverse of `build_prompt` (marker, label, text) - used by the offline extractive
    provider."""
    question = prompt.split("=== QUESTION ===", 1)[-1].strip()
    passages = [
        (int(m.group(1)), m.group(2), m.group(3).strip()) for m in _PASSAGE.finditer(prompt)
    ]
    return question, passages

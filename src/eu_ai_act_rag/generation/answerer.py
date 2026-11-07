"""Grounded answer generation with mandatory, verified citations and abstention.

The model writes plain text with inline ``[n]`` markers (streamable token by token); markers are
then resolved against the passages actually supplied and turned into a pydantic `Answer`.
Markers pointing outside the context are dropped. An answer with no valid citation is treated as
an abstention - an uncited legal claim is worse than "I don't know".
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from eu_ai_act_rag.generation.prompts import (
    ABSTAIN_SENTINEL,
    build_prompt,
    system_prompt,
)
from eu_ai_act_rag.llm.base import LLMProvider
from eu_ai_act_rag.schemas import Answer, Citation, Lang, SearchHit

_MARKER = re.compile(r"\[(\d{1,2})\]")

ABSTAIN_MESSAGES: dict[Lang, str] = {
    "en": "I could not find enough support in the AI Act or GDPR passages I retrieved to answer "
    "this reliably. Try rephrasing, or name the article you are interested in.",
    "fr": "Je n'ai pas trouvé d'éléments suffisants dans les passages du règlement IA ou du RGPD "
    "retrouvés pour répondre de manière fiable. Reformulez la question ou précisez l'article visé.",
}


@dataclass
class StreamEvent:
    kind: str  # "token" | "final"
    token: str = ""
    answer: Answer | None = None


def finalize(question: str, lang: Lang, raw: str, hits: list[SearchHit], provider: str) -> Answer:
    text = raw.strip()
    if not hits or text.startswith(ABSTAIN_SENTINEL) or text == "":
        return _abstain(question, lang, hits, provider, "insufficient_context")

    def keep(m: re.Match[str]) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= len(hits) else ""

    text = _MARKER.sub(keep, text).replace(ABSTAIN_SENTINEL, "").strip()
    used = sorted({int(n) for n in _MARKER.findall(text)})
    if not used:
        return _abstain(question, lang, hits, provider, "no_valid_citation")
    citations = [
        Citation(
            marker=n,
            chunk_id=hits[n - 1].chunk.chunk_id,
            unit_id=hits[n - 1].chunk.unit_id,
            label=hits[n - 1].chunk.citation_label,
            quote=hits[n - 1].chunk.text[:280],
        )
        for n in used
    ]
    return Answer(
        question=question,
        language=lang,
        answer=text,
        citations=citations,
        provider=provider,
        retrieval=hits,
    )


def _abstain(
    question: str, lang: Lang, hits: list[SearchHit], provider: str, reason: str
) -> Answer:
    return Answer(
        question=question,
        language=lang,
        answer=ABSTAIN_MESSAGES[lang],
        abstained=True,
        abstain_reason=reason,
        provider=provider,
        retrieval=hits,
    )


class Answerer:
    def __init__(self, llm: LLMProvider, max_tokens: int = 1024) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def answer(self, question: str, lang: Lang, hits: list[SearchHit]) -> Answer:
        if not hits:
            return _abstain(question, lang, hits, self.llm.name, "no_retrieval_results")
        raw = self.llm.complete(system_prompt(lang), build_prompt(question, hits), self.max_tokens)
        return finalize(question, lang, raw, hits, self.llm.name)

    def stream(self, question: str, lang: Lang, hits: list[SearchHit]) -> Iterator[StreamEvent]:
        if not hits:
            yield StreamEvent(
                "final",
                answer=_abstain(question, lang, hits, self.llm.name, "no_retrieval_results"),
            )
            return
        buf = ""
        released = False
        for tok in self.llm.stream(
            system_prompt(lang), build_prompt(question, hits), self.max_tokens
        ):
            buf += tok
            if released:
                yield StreamEvent("token", token=tok)
            else:
                # hold back the first characters until we know it is not the abstain sentinel
                head = buf.lstrip()
                decided = len(head) >= len(ABSTAIN_SENTINEL) or not ABSTAIN_SENTINEL.startswith(
                    head
                )
                if decided and not head.startswith(ABSTAIN_SENTINEL):
                    released = True
                    yield StreamEvent("token", token=buf)
        yield StreamEvent("final", answer=finalize(question, lang, buf, hits, self.llm.name))

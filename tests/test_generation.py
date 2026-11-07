from __future__ import annotations

from collections.abc import Iterator

from eu_ai_act_rag.generation.answerer import Answerer, finalize
from eu_ai_act_rag.generation.prompts import ABSTAIN_SENTINEL, build_prompt, parse_prompt
from eu_ai_act_rag.llm.mock import MockLLM
from eu_ai_act_rag.pipeline import RAGService
from eu_ai_act_rag.schemas import Chunk, SearchHit


def _hits() -> list[SearchHit]:
    c1 = Chunk(
        chunk_id="c1",
        unit_id="gdpr:art_33",
        regulation="gdpr",
        lang="en",
        paragraph="1",
        text="The controller shall notify the breach not later than 72 hours.",
    )
    c2 = Chunk(
        chunk_id="c2",
        unit_id="gdpr:art_34",
        regulation="gdpr",
        lang="en",
        text="The controller shall communicate the breach to the data subject.",
    )
    return [SearchHit(chunk=c1, score=2.0, rank=1), SearchHit(chunk=c2, score=1.0, rank=2)]


def test_prompt_roundtrip() -> None:
    q, passages = parse_prompt(build_prompt("When to notify?", _hits()))
    assert q == "When to notify?"
    assert [m for m, _, _ in passages] == [1, 2]
    assert passages[0][2].startswith("The controller shall notify")


def test_finalize_resolves_and_drops_invalid_markers() -> None:
    a = finalize("q", "en", "Within 72 hours [1][7]. Also inform people [2].", _hits(), "x")
    assert not a.abstained
    assert [c.marker for c in a.citations] == [1, 2]
    assert "[7]" not in a.answer
    assert a.citations[0].label == "GDPR, Art. 33(1)"


def test_uncited_answer_becomes_abstention() -> None:
    a = finalize("q", "fr", "Une réponse sans source.", _hits(), "x")
    assert a.abstained and a.abstain_reason == "no_valid_citation"
    assert a.answer.startswith("Je n'ai pas")


def test_sentinel_abstains() -> None:
    a = finalize("q", "en", ABSTAIN_SENTINEL, _hits(), "x")
    assert a.abstained and a.abstain_reason == "insufficient_context"


class _SentinelLLM:
    name = "fake"

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        return ABSTAIN_SENTINEL

    def stream(self, system: str, prompt: str, max_tokens: int = 1024) -> Iterator[str]:
        yield from ["INSUFF", "ICIENT_", "CONTEXT"]


def test_stream_never_leaks_sentinel() -> None:
    events = list(Answerer(_SentinelLLM()).stream("q", "en", _hits()))
    assert [e.kind for e in events] == ["final"]
    assert events[0].answer is not None and events[0].answer.abstained


def test_mock_extracts_and_cites() -> None:
    text = MockLLM().complete("", build_prompt("When must a breach be notified (hours)?", _hits()))
    assert "[1]" in text and "72 hours" in text


def test_service_answers_french_question_in_french(service: RAGService) -> None:
    ans = service.ask(
        "Dans quel délai notifier une violation de données à l'autorité de contrôle ?"
    )
    assert ans.language == "fr"
    assert not ans.abstained
    assert any(c.unit_id == "gdpr:art_33" for c in ans.citations)
    assert ans.answer.startswith("D'après")


def test_service_abstains_out_of_scope(service: RAGService) -> None:
    ans = service.ask("Who won the football world cup in 2018?")
    assert ans.abstained

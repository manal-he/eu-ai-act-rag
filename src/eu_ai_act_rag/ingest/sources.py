"""Registry of the official texts we ingest (CELEX identifiers on EUR-Lex)."""

from __future__ import annotations

from dataclasses import dataclass

from eu_ai_act_rag.schemas import Lang


@dataclass(frozen=True)
class Source:
    key: str
    celex: str
    name_en: str
    name_fr: str


SOURCES: dict[str, Source] = {
    "ai_act": Source(
        "ai_act",
        "32024R1689",
        "Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
        "Règlement (UE) 2024/1689 (règlement sur l'intelligence artificielle)",
    ),
    "gdpr": Source(
        "gdpr",
        "32016R0679",
        "Regulation (EU) 2016/679 (General Data Protection Regulation)",
        "Règlement (UE) 2016/679 (règlement général sur la protection des données)",
    ),
}

LANGS: tuple[Lang, ...] = ("en", "fr")
EURLEX_LANG = {"en": "EN", "fr": "FR"}
CELLAR_LANG = {"en": "eng", "fr": "fra"}


def eurlex_html_url(celex: str, lang: Lang) -> str:
    return (
        f"https://eur-lex.europa.eu/legal-content/{EURLEX_LANG[lang]}/TXT/HTML/?uri=CELEX:{celex}"
    )


def eurlex_page_url(celex: str, lang: Lang, anchor: str = "") -> str:
    url = f"https://eur-lex.europa.eu/legal-content/{EURLEX_LANG[lang]}/TXT/?uri=CELEX:{celex}"
    return f"{url}#{anchor}" if anchor else url


def cellar_url(celex: str) -> str:
    """Publications Office content-negotiation endpoint (same XHTML as EUR-Lex)."""
    return f"http://publications.europa.eu/resource/celex/{celex}"

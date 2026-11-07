"""Download the official FR/EN XHTML of each regulation at runtime.

EUR-Lex's HTML endpoint sits behind a bot-protection challenge that often answers scripted
clients with an empty `202`. The Publications Office "Cellar" content-negotiation endpoint
serves the very same Formex-generated XHTML, so we try EUR-Lex first and fall back to Cellar.
Texts are reused under Commission Decision 2011/833/EU; source: (c) European Union,
https://eur-lex.europa.eu. They are written to data/raw/ (git-ignored), never committed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from eu_ai_act_rag.ingest.sources import CELLAR_LANG, LANGS, SOURCES, cellar_url, eurlex_html_url
from eu_ai_act_rag.schemas import Lang

log = logging.getLogger(__name__)
_UA = "eu-ai-act-rag/0.1 (+https://github.com/manal-he/eu-ai-act-rag)"


def _looks_like_act(body: bytes) -> bool:
    return len(body) > 50_000 and b"oj-ti-art" in body


def fetch_act(client: httpx.Client, celex: str, lang: Lang) -> bytes:
    try:
        r = client.get(eurlex_html_url(celex, lang))
        if r.status_code == 200 and _looks_like_act(r.content):
            return r.content
        log.info(
            "EUR-Lex returned %s (%d bytes); falling back to Cellar", r.status_code, len(r.content)
        )
    except httpx.HTTPError as exc:  # pragma: no cover - network dependent
        log.info("EUR-Lex request failed (%s); falling back to Cellar", exc)
    r = client.get(
        cellar_url(celex),
        headers={"Accept": "application/xhtml+xml", "Accept-Language": CELLAR_LANG[lang]},
    )
    r.raise_for_status()
    if not _looks_like_act(r.content):
        raise RuntimeError(f"Unexpected payload for {celex}/{lang} ({len(r.content)} bytes)")
    return r.content


def download_all(raw_dir: Path, *, force: bool = False) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": _UA}) as client:
        for src in SOURCES.values():
            for lang in LANGS:
                path = raw_dir / f"{src.celex}_{lang}.html"
                if force or not path.exists():
                    log.info("Downloading %s (%s)", src.celex, lang)
                    path.write_bytes(fetch_act(client, src.celex, lang))
                paths.append(path)
    return paths

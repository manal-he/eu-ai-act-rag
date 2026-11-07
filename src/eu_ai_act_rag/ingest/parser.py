"""Structure-aware parser for Official Journal XHTML (Formex/CONVEX output used by EUR-Lex).

Relevant markup (identical across languages and across the AI Act / GDPR):

* ``div.eli-subdivision#art_5``  - an article; ``p.oj-ti-art`` number, ``p.oj-sti-art`` title
* ``div#005.001``                - paragraph 1 of article 5
* ``table`` with two ``td``      - an enumerated point: label "(a)" + body
                                   (nested tables are sub-points)
* ``div.eli-subdivision#rct_12`` - recital 12
* ``div.eli-container#anx_III``  - annex III
* ``div#cpt_III`` / ``div#cpt_III.sct_2`` - chapter / section ancestors
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from eu_ai_act_rag.ingest.sources import SOURCES, eurlex_page_url
from eu_ai_act_rag.schemas import Lang, LegalUnit, Paragraph
from eu_ai_act_rag.text import normalize_ws

_PARA_ID = re.compile(r"^\d{3}\.\d{3}$")
_CHAPTER_ID = re.compile(r"^cpt_[IVXLC]+$")
_SECTION_ID = re.compile(r"^cpt_[IVXLC]+\.sct_\d+$")


def _tag_children(el: Tag) -> list[Tag]:
    return [c for c in el.children if isinstance(c, Tag)]


def _render(el: Tag) -> list[str]:
    """Render an element into text lines, keeping enumeration labels ("(a) ...")."""
    if el.name == "p":
        txt = normalize_ws(el.get_text(" ")).replace(" ^", "^")
        return [txt] if txt else []
    if el.name == "table":
        lines: list[str] = []
        for tr in el.find_all("tr", recursive=False) or el.select(":scope > tbody > tr"):
            tds = tr.find_all("td", recursive=False)
            if len(tds) >= 2:
                label = normalize_ws(tds[0].get_text(" "))
                body: list[str] = []
                for child in _tag_children(tds[1]):
                    body.extend(_render(child))
                if body:
                    lines.append(f"{label} {body[0]}".strip())
                    lines.extend(body[1:])
            else:
                lines.extend(
                    normalize_ws(td.get_text(" ")) for td in tds if td.get_text(strip=True)
                )
        return lines
    if el.name in {"div", "tbody", "td"}:
        out: list[str] = []
        for child in _tag_children(el):
            out.extend(_render(child))
        return out
    txt = normalize_ws(el.get_text(" "))
    return [txt] if txt else []


def _paragraph_from_blocks(blocks: list[Tag], number: str | None) -> Paragraph | None:
    lead = ""
    points: list[str] = []
    for b in blocks:
        rendered = _render(b)
        if not rendered:
            continue
        if b.name == "table":
            points.append("\n".join(rendered))
        elif not points and not lead:
            lead = "\n".join(rendered)
        else:
            points.append("\n".join(rendered))
    if not lead and not points:
        return None
    return Paragraph(number=number, lead=lead, points=points)


def _heading(container: Tag) -> str:
    parts = [normalize_ws(p.get_text(" ")) for p in container.find_all("p", recursive=False)]
    title_div = container.find("div", class_="eli-title", recursive=False)
    if isinstance(title_div, Tag):
        parts.append(normalize_ws(title_div.get_text(" ")))
    return " - ".join(p for p in parts if p)


def _ancestor_heading(el: Tag, pattern: re.Pattern[str]) -> str:
    for parent in el.parents:
        pid = parent.get("id") if isinstance(parent, Tag) else None
        if isinstance(pid, str) and pattern.match(pid):
            return _heading(parent)
    return ""


def _strip_notes(soup: BeautifulSoup) -> None:
    for tag in soup.select("span.oj-note-tag"):
        anchor = tag.find_parent("a")
        (anchor or tag).decompose()
    for sup in soup.select("span.oj-super"):  # exponents, e.g. 10^25 FLOPs in Article 51
        sup.string = "^" + sup.get_text(strip=True)


def parse_articles(soup: BeautifulSoup, regulation: str, lang: Lang, celex: str) -> list[LegalUnit]:
    units: list[LegalUnit] = []
    for art in soup.select("div.eli-subdivision[id^=art_]"):
        art_id = str(art["id"])
        if "." in art_id:
            continue
        number = art_id.removeprefix("art_")
        title_el = art.select_one("p.oj-sti-art")
        title = normalize_ws(title_el.get_text(" ")) if title_el else ""
        children = _tag_children(art)
        para_divs = [
            c for c in children if c.name == "div" and _PARA_ID.match(str(c.get("id", "")))
        ]
        paragraphs: list[Paragraph] = []
        if para_divs:
            for div in para_divs:
                para = _paragraph_from_blocks(_tag_children(div), str(int(str(div["id"])[4:])))
                if para:
                    paragraphs.append(para)
        else:
            body = [
                c
                for c in children
                if not (
                    "oj-ti-art" in (c.get("class") or []) or c.get("id", "") == f"{art_id}.tit_1"
                )
            ]
            para = _paragraph_from_blocks(body, None)
            if para:
                paragraphs.append(para)
        if not paragraphs:
            continue
        units.append(
            LegalUnit(
                unit_id=f"{regulation}:art_{number}",
                regulation=regulation,
                kind="article",
                number=number,
                title=title,
                lang=lang,
                chapter=_ancestor_heading(art, _CHAPTER_ID),
                section=_ancestor_heading(art, _SECTION_ID),
                paragraphs=paragraphs,
                source_url=eurlex_page_url(celex, lang, art_id),
            )
        )
    return units


def parse_recitals(soup: BeautifulSoup, regulation: str, lang: Lang, celex: str) -> list[LegalUnit]:
    units: list[LegalUnit] = []
    for rct in soup.select("div.eli-subdivision[id^=rct_]"):
        rid = str(rct["id"])
        number = rid.removeprefix("rct_")
        lines: list[str] = []
        for child in _tag_children(rct):
            lines.extend(_render(child))
        text = "\n".join(lines)
        text = re.sub(r"^\(\d+\)\s*", "", text)
        if not text:
            continue
        units.append(
            LegalUnit(
                unit_id=f"{regulation}:rct_{number}",
                regulation=regulation,
                kind="recital",
                number=number,
                lang=lang,
                paragraphs=[Paragraph(lead=text)],
                source_url=eurlex_page_url(celex, lang, rid),
            )
        )
    return units


def parse_annexes(soup: BeautifulSoup, regulation: str, lang: Lang, celex: str) -> list[LegalUnit]:
    units: list[LegalUnit] = []
    for anx in soup.select("div.eli-container[id^=anx_]"):
        aid = str(anx["id"])
        number = aid.removeprefix("anx_")
        children = _tag_children(anx)
        titles = [c for c in children if c.name == "p" and "oj-doc-ti" in (c.get("class") or [])]
        title = normalize_ws(titles[1].get_text(" ")) if len(titles) > 1 else ""
        body = [c for c in children if c not in titles]
        # One paragraph: the annex's introductory sentence is the lead, each top-level block a
        # point, so the chunker carries the lead into every piece when it splits the annex.
        para = _paragraph_from_blocks(body, None)
        if para is None:
            continue
        paragraphs = [para]
        units.append(
            LegalUnit(
                unit_id=f"{regulation}:anx_{number}",
                regulation=regulation,
                kind="annex",
                number=number,
                title=title,
                lang=lang,
                paragraphs=paragraphs,
                source_url=eurlex_page_url(celex, lang, aid),
            )
        )
    return units


def parse_html(html: str | bytes, regulation: str, lang: Lang) -> list[LegalUnit]:
    celex = SOURCES[regulation].celex
    with warnings.catch_warnings():
        # the OJ files are XHTML; the lenient HTML parser is intentional (robust to quirks)
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    _strip_notes(soup)
    return (
        parse_articles(soup, regulation, lang, celex)
        + parse_recitals(soup, regulation, lang, celex)
        + parse_annexes(soup, regulation, lang, celex)
    )


def parse_file(path: Path) -> list[LegalUnit]:
    """Parse ``data/raw/{celex}_{lang}.html``."""
    celex, lang = path.stem.split("_")
    regulation = next(k for k, s in SOURCES.items() if s.celex == celex)
    if lang not in ("en", "fr"):
        raise ValueError(f"Unsupported language in {path.name}")
    return parse_html(path.read_bytes(), regulation, lang)  # type: ignore[arg-type]

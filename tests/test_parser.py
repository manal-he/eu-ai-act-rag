"""Parser tests on a hand-written snippet that mirrors the Official Journal XHTML markup."""

from __future__ import annotations

from eu_ai_act_rag.ingest.parser import parse_html

SNIPPET = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<div class="eli-subdivision" id="rct_1"><table><tbody><tr>
  <td><p class="oj-normal">(1)</p></td><td><p class="oj-normal">The purpose of this
  Regulation is to improve the internal market<a href="#n1">(<span class="oj-super oj-note-tag">1</span>)</a>.</p></td>
</tr></tbody></table></div>
<div id="cpt_II"><p class="oj-ti-section-1">CHAPTER II</p>
 <div class="eli-title" id="cpt_II.tit_1"><p class="oj-ti-section-2">PROHIBITED AI PRACTICES</p></div>
 <div class="eli-subdivision" id="art_5">
  <p class="oj-ti-art">Article 5</p>
  <div class="eli-title" id="art_5.tit_1"><p class="oj-sti-art">Prohibited AI practices</p></div>
  <div id="005.001">
   <p class="oj-normal">1.   The following AI practices shall be prohibited:</p>
   <table><tbody><tr><td><p class="oj-normal">(a)</p></td>
     <td><p class="oj-normal">subliminal techniques;</p></td></tr></tbody></table>
   <table><tbody><tr><td><p class="oj-normal">(b)</p></td>
     <td><p class="oj-normal">exploiting vulnerabilities;</p>
       <table><tbody><tr><td><p class="oj-normal">(i)</p></td>
         <td><p class="oj-normal">due to age;</p></td></tr></tbody></table></td></tr></tbody></table>
  </div>
  <div id="005.002"><p class="oj-normal">2.   Training above 10<span class="oj-super">25</span> FLOPs.</p></div>
 </div>
</div>
<div class="eli-container" id="anx_III">
  <p class="oj-doc-ti">ANNEX III</p><p class="oj-doc-ti">High-risk AI systems</p>
  <p class="oj-normal">Areas:</p>
  <table><tbody><tr><td><p class="oj-normal">1.</p></td>
    <td><p class="oj-normal">Biometrics.</p></td></tr></tbody></table>
</div>
</body></html>"""


def test_parses_articles_paragraphs_points() -> None:
    units = {u.unit_id: u for u in parse_html(SNIPPET, "ai_act", "en")}
    art = units["ai_act:art_5"]
    assert art.kind == "article" and art.title == "Prohibited AI practices"
    assert art.chapter == "CHAPTER II - PROHIBITED AI PRACTICES"
    assert [p.number for p in art.paragraphs] == ["1", "2"]
    p1 = art.paragraphs[0]
    assert p1.lead.startswith("1. The following AI practices")
    assert p1.points[0] == "(a) subliminal techniques;"
    assert "(i) due to age;" in p1.points[1]  # nested sub-point kept under its point
    assert "10^25" in art.paragraphs[1].text  # exponent preserved
    assert "eur-lex.europa.eu" in art.source_url and art.source_url.endswith("#art_5")


def test_recitals_strip_footnotes_and_number() -> None:
    units = {u.unit_id: u for u in parse_html(SNIPPET, "ai_act", "en")}
    rct = units["ai_act:rct_1"]
    assert rct.text.startswith("The purpose of this Regulation")
    assert "(1)" not in rct.text


def test_annex_lead_and_title() -> None:
    units = {u.unit_id: u for u in parse_html(SNIPPET, "ai_act", "fr")}
    anx = units["ai_act:anx_III"]
    assert anx.title == "High-risk AI systems" and anx.lang == "fr"
    assert anx.paragraphs[0].lead == "Areas:"
    assert anx.paragraphs[0].points == ["1. Biometrics."]

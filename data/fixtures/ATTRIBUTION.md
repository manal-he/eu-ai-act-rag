# Fixture attribution

`corpus_fixture.jsonl` is a small excerpt (36 articles/annexes x 2 languages) of:

- Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024
  laying down harmonised rules on artificial intelligence (Artificial Intelligence Act),
  OJ L, 2024/1689, 12.7.2024. CELEX 32024R1689.
- Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016
  (General Data Protection Regulation), OJ L 119, 4.5.2016, p. 1. CELEX 32016R0679.

Source: (c) European Union, https://eur-lex.europa.eu. Reuse is authorised under Commission
Decision 2011/833/EU on the reuse of Commission documents. Only the official Official Journal
text is authentic; this excerpt was produced by the parser in `src/eu_ai_act_rag/ingest/`
(structure only, no editorial change) and may contain parsing artefacts.

It exists for tests and the offline demo. The full corpora are downloaded at runtime with
`make ingest` into `data/raw/` and `data/processed/`, which are git-ignored.

Regenerate: `make ingest && euai make-fixture` (unit list: `FIXTURE_UNITS` in `src/eu_ai_act_rag/cli.py`).

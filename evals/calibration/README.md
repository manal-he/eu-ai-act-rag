# Judge calibration

An LLM judge is a measurement instrument. Before its faithfulness / correctness numbers are
reported, they must be shown to agree with human judgement on this task.

## Protocol (v1 - not yet executed)

1. Run the generation eval with the judge you want to calibrate, e.g.
   `python evals/run_ablation.py --profile full --generation --llm gemini --judge llm`.
   This writes `evals/results/full/generation_rows.json` (git-ignored).
2. Sample >= 50 answered rows, stratified by language (FR/EN) and category. Copy them into
   `human_labels.jsonl` using `human_labels_template.jsonl` as the schema.
3. Two annotators (ideally one with legal training) label each row blind to the judge score:
   `faithfulness` and `correctness` in {0, 0.5, 1}. Resolve disagreements; report the
   inter-annotator kappa as the ceiling the judge can be expected to reach.
4. `python evals/calibration/calibrate.py human_labels.jsonl judged_rows.json` prints Cohen's
   kappa (binarised at 0.5), Spearman rho and raw agreement per criterion.
5. Acceptance rule used here: kappa >= 0.6 and no systematic bias by language. If it fails,
   revise the rubric (`src/eu_ai_act_rag/evaluation/judge.py`), not the labels.

## Known biases to check

- **Self-preference**: do not judge a model's answers with the same model family; the default
  pairs a Gemini generator with a Claude judge.
- **Verbosity bias**: longer answers tend to be scored higher on correctness; compare score vs.
  answer length.
- **Language bias**: rubric is in English; check FR rows separately.
- **Reference leakage**: faithfulness must be graded against the context only - spot-check
  rationales.

No calibration result is reported in the README until this protocol has been run.

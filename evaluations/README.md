# Evaluations

Two eval sets. Both come from `data/synthetic-priorauth-dataset-v1.csv`.

| File | Cases | Purpose |
|---|---|---|
| `eval-sets/intake-triage-full-dataset.json` | 25 | Alignment against the whole dataset |
| `eval-sets/pii-detection-smoke.json` | 2 | PA-1021 and PA-1017, the rows with planted identifiers |

Do not edit the eval sets by hand. Edit the CSV or the generator, then run:

    uv run python evaluations/build_eval_sets.py

## Evaluators

| Evaluator | Type | Reads | Scoring |
|---|---|---|---|
| `ReviewPathExactMatch` | `uipath-exact-match` | `review_path` | 1.0 or 0.0 for each case. Macro precision, recall, F-score, and a 3x3 confusion matrix |
| `PiiDetectionMatch` | `uipath-json-similarity` | `pii_detected_by_type` | Ten leaves for each case, one for each identifier type |

### ReviewPathExactMatch

The prompt and the routing rules match the masking variant of this agent. So
this score should match that agent's score. A large difference means the prompt
edit went wrong. It does not mean the guardrail went wrong.

### PiiDetectionMatch

The expected value is a map with one key for each identifier type. The
generator derives it from the row. It does not come from a hand-written list,
so it cannot drift from the data.

Three structured fields put identifiers into every prompt: `member_id`,
`member_dob`, and `provider_npi`. Only PA-1021 and PA-1017 add more, in their
clinical notes.

The evaluator compares a map and not a list on purpose. `uipath-json-similarity`
walks a list by position. One missed type would move every later item and the
score would collapse for a single error. A map scores one leaf for each type,
which counts a miss and a false positive equally.

This is an expectation about the text, not about Jev. When Jev disagrees with
a row, the finding is a calibration result. Change `TYPESAFE_PII_THRESHOLD`,
or change the question wording in `jev_pii.py`. Do not change the expectation
to match the model.

The first run is a calibration run. Treat the score as a measurement.

## Run

    uip login                                    # the eval needs a live token
    uv run uipath eval agent evaluations/eval-sets/pii-detection-smoke.json --no-report
    uv run uipath eval agent evaluations/eval-sets/intake-triage-full-dataset.json \
        --no-report --workers 4 --output-file results.json

Each case makes two TypeSafe requests, one for each boundary. The full set
therefore makes 50. Watch the cost before you raise `--workers`.

Add `--output-file` to keep the score for each case. `--report` needs
`UIPATH_PROJECT_ID` in `.env`, which this project does not have yet.

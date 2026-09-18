# Prior-auth triage agent with a Jev PII guardrail

A UiPath coded agent that triages prior-authorization requests. A custom
guardrail on the model boundary detects PII and writes it to the record.

This is a variant of an intake triage agent that masks every identifier before
the model call. This one does not mask anything. It sends the request in the
clear and measures what crosses the boundary.

## How the guardrail works

`jev_pii.py` asks the TypeSafe Jev model one narrow question for each
identifier type, plus one score for how identifiable the patient is. All
questions go in one request and run in parallel.

Jev returns a probability for each type. It does not return an offset. You
cannot redact from a Jev answer. The guardrail therefore logs and records. It
never masks and never blocks.

Detection is fail-open. A detector outage must not stop the agent. The output
field `pii_detection_status` says whether each boundary was checked, so a
clean run cannot be confused with an unchecked one.

## Graph

```
START -> triage --(Exception)--> escalate --(ReturnToAgent)--> triage
  |            --(otherwise)-------------------------------> finalize -> END
  |                                        --(otherwise)--> finalize
```

`escalate` interrupts the run and creates an Action Center task in the
`prior-auth-review-app` Coded Action App. Only the Exception path reaches it.

When the reviewer sends the case back, the run triages again. The reviewer's
notes go to the model as written. The guardrail scores them. Those notes are
the most likely cleartext PHI in the graph, because the reviewer writes them
while reading a screen full of identifiers.

## Files

| File | Contents |
|---|---|
| `main.py` | The graph: `triage`, `escalate`, `finalize` |
| `jev_pii.py` | The guardrail middleware and the Jev question battery |
| `evaluations/` | Two eval sets, two evaluators, and the generator |
| `tests/test_jev_pii.py` | Guardrail tests. No network |
| `tests/test_return_loop.py` | Return-loop tests. No network |

## Setup

Copy `.env.example` to `.env` and fill it in. Add a TypeSafe API key. Get one from
https://console.typesafe.ai/settings/keys.

```
TYPESAFE_API_KEY=...
```

Without a key the agent still runs. Every boundary reports "unchecked".

Two optional variables:

| Variable | Default | Purpose |
|---|---|---|
| `TYPESAFE_MODEL` | `jev-latest` | Pin the Jev version |
| `TYPESAFE_PII_THRESHOLD` | `0.60` | Probability that counts as a detection |

## Run

```bash
uv sync

# Offline tests. No UiPath token and no TypeSafe key needed.
uv run python tests/test_jev_pii.py
uv run python tests/test_return_loop.py

# One case, live.
uv run uipath run agent --file tests/inputs/PA-1017.json --output-file out.json

# Regenerate the schema after you change Input or Output.
uv run uipath init
```

`PA-1017` and `PA-1021` carry planted synthetic identifiers. Use them to see
the guardrail fire. `PA-1025` routes to Exception and suspends the run, so use
it to see the escalation and the return loop.

## Output

Three fields carry the guardrail result.

| Field | Contents |
|---|---|
| `pii_reports` | One entry for each boundary: the probability for each type, and the exposure score |
| `pii_detected_by_type` | Each type, answered true or false. The graded field |
| `pii_detection_status` | "checked", or the reason a boundary was not |

No field holds an identifier value.

## Data

`evaluations/data/synthetic-priorauth-dataset-v1.csv` holds 25 synthetic
prior-authorization requests. No row describes a real person. Two rows,
PA-1017 and PA-1021, carry invented identifiers in their clinical notes so the
guardrail has something to find.

## Status

This is a demo. It shows one control on one boundary. It is not a
de-identification pipeline, and it is not hardened for production use.

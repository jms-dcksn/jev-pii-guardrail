"""Generate the eval sets from the synthetic dataset.

One CSV row becomes one evaluation. `ReviewPath` becomes the expected value for
`ReviewPathExactMatch`, so the eval set cannot drift from the answer key: it is
derived from it. Re-run this after any edit to the CSV.

    uv run python evaluations/build_eval_sets.py

Two sets come out:

- `intake-triage-full-dataset.json` -- all 25 rows. The alignment set.
- `pii-detection-smoke.json` -- PA-1021 and PA-1017 only. The two rows whose
  clinical notes carry planted identifiers, for a fast loop and for the demo.

Held-out columns: `InterQualCriteriaSet` (unused), `PriorAuthStatus`
(downstream of triage), `PlantedTestCondition` (states the answer). None of
them reach the agent.
"""

import csv
import json
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = Path(__file__).resolve().parent / "data" / "synthetic-priorauth-dataset-v1.csv"
OUT_DIR = Path(__file__).resolve().parent / "eval-sets"

# CSV column -> PriorAuthIntake field. Every column not listed here is held out.
FIELD_MAP = {
    "CaseID": "case_id",
    "SyntheticMemberID": "member_id",
    "SyntheticDOB": "member_dob",
    "PlanType": "plan_type",
    "RequestingProviderNPI": "provider_npi",
    "ServiceRequested": "service_requested",
    "CPTCode": "cpt_code",
    "ICD10Code": "icd10_code",
    "ClinicalIndication": "clinical_indication",
    "SubmittedDate": "submitted_date",
    "RequestedUrgency": "requested_urgency",
    "EstimatedCost": "estimated_cost",
    "ClinicalNoteFreeText": "clinical_note",
}

SMOKE_CASES = ("PA-1021", "PA-1017")

# The guardrail's battery, in order. Kept as a literal rather than imported
# from jev_pii so that generating an eval set never needs the TypeSafe SDK
# installed. If the battery changes, this list has to follow.
PII_QUESTION_IDS = (
    "person_name",
    "date_of_birth",
    "ssn",
    "mrn",
    "member_or_policy_id",
    "phone",
    "email",
    "address",
    "provider_npi",
    "part2_content",
)

# Which structured field, when populated, puts which identifier type into the
# prompt. `_render_request` lays all three out as labelled lines, so they are
# in the text the guardrail sees whether or not the clinical note repeats them.
STRUCTURED_FIELD_TYPES = {
    "member_id": "member_or_policy_id",
    "member_dob": "date_of_birth",
    "provider_npi": "provider_npi",
}

# What the clinical note adds, on the two rows whose notes carry planted
# identifiers. Everything else these rows contain is already covered by
# STRUCTURED_FIELD_TYPES. Read off the notes, not guessed:
#
#   PA-1021  name, numeric DOB, MRN, member id, SSN, delimited phone,
#            street address, email, referring physician and NPI
#   PA-1017  name, spelled-out DOB, member id, policy number, undelimited
#            phone, email, street address, clinician and NPI, plus
#            42 CFR Part 2 substance-use content
NOTE_TYPES = {
    "PA-1021": ["person_name", "ssn", "mrn", "phone", "email", "address"],
    "PA-1017": ["person_name", "phone", "email", "address", "part2_content"],
}


def read_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_inputs(row: dict[str, str]) -> dict[str, object]:
    """Map one row onto the agent's input schema.

    Empty cells are omitted rather than sent as "". Every field but `case_id`
    is optional, so an omitted field arrives as None and renders as <absent>,
    which is what a real intake feed does. PA-1025 depends on this: its blank
    ICD-10 is the reason it routes Exception.
    """
    inputs: dict[str, object] = {}
    for column, field in FIELD_MAP.items():
        value = (row.get(column) or "").strip()
        if not value:
            continue
        inputs[field] = float(value) if field == "estimated_cost" else value
    return inputs


def build_detection_expectation(inputs: dict[str, object]) -> dict[str, bool]:
    """What the guardrail ought to detect in this case, one leaf per type.

    Derived from the row, not hand-written, so it cannot drift from the data.
    A fixed-shape map rather than a list: `uipath-json-similarity` walks lists
    positionally, so one miss would shift every later element and collapse the
    score. Ten keys, ten leaves, one per identifier type.

    This is an expectation about the TEXT, not about Jev. A row where Jev
    disagrees is a calibration finding -- either the threshold is wrong or the
    question wording is -- and reading it as a bug in the agent would be
    reading it backwards.
    """
    expected = {
        pii_type
        for field, pii_type in STRUCTURED_FIELD_TYPES.items()
        if inputs.get(field)
    }
    expected.update(NOTE_TYPES.get(str(inputs["case_id"]), []))
    return {qid: qid in expected for qid in PII_QUESTION_IDS}


def build_evaluation(row: dict[str, str]) -> dict[str, object]:
    case_id = row["CaseID"]
    expected_path = row["ReviewPath"].strip()
    service = row["ServiceRequested"].strip()
    inputs = build_inputs(row)
    return {
        "id": f"{case_id.lower()}-{expected_path.lower()}",
        "name": f"{case_id} {service} -> {expected_path}",
        "inputs": inputs,
        "evaluationCriterias": {
            "ReviewPathExactMatch": {"expectedOutput": {"review_path": expected_path}},
            "PiiDetectionMatch": {
                "expectedOutput": {
                    "pii_detected_by_type": build_detection_expectation(inputs)
                }
            },
        },
    }


def write_set(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    count = len(payload["evaluations"])  # type: ignore[arg-type]
    print(f"{path.relative_to(AGENT_ROOT)}: {count} evaluations")


def main() -> None:
    rows = read_rows()
    evaluations = [build_evaluation(row) for row in rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_set(
        OUT_DIR / "intake-triage-full-dataset.json",
        {
            "version": "1.0",
            "id": "intake-triage-full-dataset",
            "name": "Intake triage -- full synthetic dataset",
            "description": (
                "All 25 rows of synthetic-priorauth-dataset-v1.csv. Routing is "
                "graded against the ReviewPath column; PII detection is graded per "
                "identifier type against what the request text actually contains. "
                "Generated by evaluations/build_eval_sets.py -- do not hand-edit."
            ),
            "evaluatorRefs": ["ReviewPathExactMatch", "PiiDetectionMatch"],
            "evaluations": evaluations,
        },
    )

    smoke = [e for e in evaluations if e["inputs"]["case_id"] in SMOKE_CASES]  # type: ignore[index]
    write_set(
        OUT_DIR / "pii-detection-smoke.json",
        {
            "version": "1.0",
            "id": "pii-detection-smoke",
            "name": "PII detection smoke -- PA-1021 and PA-1017",
            "description": (
                "The two rows whose clinical notes carry planted synthetic identifiers, "
                "in different formats: PA-1021 numeric DOB, MRN, SSN and a delimited "
                "phone; PA-1017 spelled-out DOB, an undelimited phone and 42 CFR Part 2 "
                "content. Between them they exercise every question in the battery. "
                "Generated by evaluations/build_eval_sets.py -- do not hand-edit."
            ),
            "evaluatorRefs": ["ReviewPathExactMatch", "PiiDetectionMatch"],
            "evaluations": smoke,
        },
    )


if __name__ == "__main__":
    main()

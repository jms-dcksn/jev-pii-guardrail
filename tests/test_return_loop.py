"""Exercise the ReturnToAgent loop offline.

Real interrupt, real routers, real finalize, real resume through a
checkpointer. Only the model call is stubbed, because the loop is graph
plumbing and the stub makes the three outcomes deterministic.

The loop is where this variant differs from the masking agent. There, a
returned case went back through `redact` so the reviewer's free text was masked
with the rest of the request. Here it goes straight back to `triage`: the notes
reach the model as written, and the guardrail is the only thing that sees them.
The test prints what the second prompt carries so that is visible, not implied.

    uv run python tests/test_return_loop.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import asyncio
import json

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from jev_pii import PiiDetectionReport, PiiFinding
from main import (PriorAuthIntake, TriageDecision, TriageResult, TriageState,
                  _render_request, escalate, finalize, route_after_escalation,
                  route_after_triage)

PASSES = []


async def stub_triage(state: TriageState) -> dict:
    """Exception first. On the second pass, triage if the guidance repaired it.

    Stands in for the real node, guardrail included: it records the prompt the
    model would have seen and fabricates one report per pass, so `finalize` has
    the same shape of input it gets in a live run.
    """
    prompt = _render_request(state, state.reviewer_guidance)
    PASSES.append(prompt)
    repaired = "CPT 29881" in prompt
    decision = TriageDecision(
        case_id=state.case_id,
        review_path="Auto-Criteria" if repaired else "Exception",
        routing_rationale="reviewer supplied the codes" if repaired else "cannot triage",
        exception_reason="" if repaired else "CPT/HCPCS code and ICD-10 code are required.",
        data_quality="Complete" if repaired else "Malformed",
        missing_fields=[] if repaired else ["CPT/HCPCS code", "ICD-10 code"],
        high_cost=False,
        urgency="Routine",
        clinical_summary="Knee arthroscopy requested." if repaired else "No clinical content.",
        confidence=0.9 if repaired else 0.0,
    )
    report = PiiDetectionReport(
        boundary="request",
        status="checked",
        exposure=2.5 if state.triage_passes else 1.5,
        findings=[
            PiiFinding(pii_type="mrn", probability=0.93, detected=True),
            PiiFinding(pii_type="date_of_birth", probability=0.88, detected=True),
            # Only the second pass carries the reviewer's free text.
            PiiFinding(
                pii_type="person_name",
                probability=0.91 if state.triage_passes else 0.03,
                detected=bool(state.triage_passes),
            ),
        ],
    )
    return {
        "decision": decision,
        "pii_reports": [*state.pii_reports, report],
        "triage_passes": state.triage_passes + 1,
    }


builder = StateGraph(TriageState, input_schema=PriorAuthIntake, output_schema=TriageResult)
builder.add_node("triage", stub_triage)
builder.add_node("escalate", escalate)
builder.add_node("finalize", finalize)
builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", route_after_triage, ["escalate", "finalize"])
builder.add_conditional_edges("escalate", route_after_escalation, ["triage", "finalize"])
builder.add_edge("finalize", END)
graph = builder.compile(checkpointer=MemorySaver())

INPUTS = pathlib.Path(__file__).parent / "inputs"
intake = json.loads((INPUTS / "PA-1025.json").read_text())


async def run(fixture: str, thread: str) -> TriageResult:
    PASSES.clear()
    cfg = {"configurable": {"thread_id": thread}}
    out = await graph.ainvoke(intake, cfg)
    assert "__interrupt__" in out, "expected an interrupt on the first Exception"
    event = out["__interrupt__"][0].value
    print(f"  interrupt -> app={event.app_name!r} title={event.title!r}")
    print(f"  escalationReason : {event.data['escalationReason'][:58]}...")
    print(f"  redactionStatus  : {event.data['redactionStatus'][:70]}")

    task = json.loads((INPUTS / fixture).read_text())
    final = await graph.ainvoke(Command(resume=task), cfg)
    result = TriageResult(**final)
    print(f"  triage passes run: {len(PASSES)}")
    print(f"  review_path      : {result.review_path}")
    print(f"  escalation_status: {result.escalation_status}")
    print(
        f"  human outcome    : {result.human_decision.outcome} | "
        f"attestation {result.human_decision.attestation} | "
        f"task {result.human_decision.task_id}"
    )
    print(f"  pii detected     : {result.pii_types_detected}")
    print(f"  pii exposure     : {result.pii_exposure} | {result.pii_detection_status}")

    guidance_seen = len(PASSES) > 1 and "Reviewer guidance" in PASSES[-1]
    print(f"  guidance reached the model: {guidance_seen}")
    if guidance_seen:
        # The masking agent masked these before the second call. This one does
        # not, and saying which values survived is the whole demonstration.
        cleartext = [v for v in ("4471982", "9/12/26") if v in PASSES[-1]]
        print(f"  cleartext PHI in the second prompt: {cleartext or 'none'}")
    return result


async def main_() -> None:
    print("\n== ReturnToAgent, reviewer repaired the intake ==")
    result = await run("PA-1025.resume-repaired.json", "t1")
    assert result.review_path == "Auto-Criteria"
    assert result.escalation_status == "returned-to-agent"
    assert len(result.pii_reports) == 2, "both passes must be recorded"
    assert "person_name" in result.pii_types_detected, "the reviewer's notes named someone"
    assert result.pii_exposure == 2.5, "exposure must be the worst of the two passes"

    print("\n== ReturnToAgent, no new information ==")
    result = await run("PA-1025.resume.json", "t2")
    assert result.review_path == "Exception"
    assert result.escalation_status == "returned-unresolved"

    print("\n== Deny, reviewer settled it ==")
    result = await run("PA-1025.resume-denied.json", "t3")
    assert result.review_path == "Exception"
    assert result.escalation_status == "escalated"
    assert len(PASSES) == 1, "Deny must not re-triage"
    assert result.pii_detection_status == "checked"

    print("\nall assertions passed")


asyncio.run(main_())

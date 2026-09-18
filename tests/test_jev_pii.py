"""Exercise the Jev guardrail offline.

Real middleware, real create_agent, real graph. Only two things are stubbed:
the chat model, and the TypeSafe call. Both for the same reason -- the thing
under test is the boundary wiring, and a stub makes it deterministic and free.

The last case is the one that matters. It proves the guardrail sees the
STRUCTURED response, which is the boundary the UiPath `@guardrail` decorator
cannot reach: with `response_format` set the AIMessage content is empty, and
`_apply_llm_post` returns on its first line.

    uv run python tests/test_jev_pii.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import asyncio
import json

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_typesafe.types import ClassificationResponse, NoulAnswer, ScoreAnswer

import jev_pii
from jev_pii import (
    EXPOSURE_ID,
    PII_QUESTION_IDS,
    JevPIIDetectionMiddleware,
    detected_types,
    peak_exposure,
)
from main import TriageDecision


# -- stubs -----------------------------------------------------------------


class StubClassifier:
    """Answers the battery from a fixed probability table. Records its inputs."""

    def __init__(self, probabilities: dict[str, float], exposure: float = 0.0):
        self.probabilities = probabilities
        self.exposure = exposure
        self.seen: list[object] = []

    async def ainvoke(self, state, config=None, **_):
        self.seen.append(state)
        answers = {
            qid: NoulAnswer(type="noul", noul=self.probabilities.get(qid, 0.0))
            for qid in PII_QUESTION_IDS
        }
        answers[EXPOSURE_ID] = ScoreAnswer(
            type="score",
            score=self.exposure,
            legend={0: "none", 1: "weak", 2: "direct", 3: "name plus"},
            probabilities={0: 0.1, 1: 0.1, 2: 0.3, 3: 0.5},
            confidence=0.8,
        )
        return ClassificationResponse(model="stub", answers=answers)


class BrokenClassifier:
    async def ainvoke(self, state, config=None, **_):
        raise RuntimeError("TypeSafe unreachable")


class StubModel(BaseChatModel):
    """Answers with a structured-output tool call, the way a real model does."""

    @property
    def _llm_type(self) -> str:
        return "stub"

    def bind_tools(self, tools, **kwargs):
        return self.bind(**{k: v for k, v in kwargs.items() if k != "tools"})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        decision = TriageDecision(
            case_id="PA-1017",
            review_path="Human-Review",
            routing_rationale="Behavioral health is an always-review category.",
            exception_reason="",
            data_quality="Complete",
            missing_fields=[],
            high_cost=False,
            urgency="Routine",
            sensitive_categories=["Behavioral-Health"],
            phi_types_observed=["Name", "DOB", "Phone"],
            clinical_summary=(
                "Michael O'Brien-Reyes, d.o.b. September 6 1997, requests an "
                "intensive outpatient programme. PHQ-9 of 18."
            ),
            confidence=0.9,
        )
        message = AIMessage(
            content="",  # <-- empty, because the answer is a tool call
            tool_calls=[
                {
                    "name": "TriageDecision",
                    "args": decision.model_dump(),
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def middleware_with(classifier, **kwargs) -> JevPIIDetectionMiddleware:
    mw = JevPIIDetectionMiddleware(**kwargs)
    mw._classifier = classifier
    return mw


PLANTED = {
    "person_name": 0.97,
    "date_of_birth": 0.94,
    "member_or_policy_id": 0.88,
    "phone": 0.91,
    "email": 0.93,
    "address": 0.89,
    "provider_npi": 0.72,
    "part2_content": 0.81,
    "ssn": 0.04,
    "mrn": 0.06,
}


# -- unit checks -----------------------------------------------------------


def test_egress_reads_the_structured_response():
    """The egress boundary must see the tool-call answer, not the empty content."""

    class Response:
        structured_response = TriageDecision(
            case_id="PA-1017",
            review_path="Human-Review",
            routing_rationale="r",
            exception_reason="",
            data_quality="Complete",
            missing_fields=[],
            high_cost=False,
            urgency="Routine",
            clinical_summary="Michael O'Brien-Reyes, d.o.b. September 6 1997.",
            confidence=0.9,
        )
        result = [AIMessage(content="")]

    text = JevPIIDetectionMiddleware._egress_text(Response())
    assert "O'Brien-Reyes" in text, "structured response was not read"
    assert json.loads(text)["review_path"] == "Human-Review"
    print("  egress reads structured_response      : ok")


def test_egress_falls_back_to_message_text():
    class Response:
        structured_response = None
        result = [AIMessage(content="plain answer")]

    assert JevPIIDetectionMiddleware._egress_text(Response()) == "plain answer"
    print("  egress falls back to AIMessage text   : ok")


def test_ingress_includes_the_system_message():
    class Request:
        system_message = SystemMessage("you are a triage agent")
        messages = [HumanMessage("Member DOB: 9/6/97")]

    state = JevPIIDetectionMiddleware._ingress_state(Request())
    assert len(state) == 2 and isinstance(state[0], SystemMessage)
    print("  ingress includes the system message   : ok")


async def test_threshold():
    mw = middleware_with(StubClassifier(PLANTED, exposure=2.9), threshold=0.60)
    report = await mw._check("text", boundary="request")
    assert report.status == "checked"
    assert set(report.detected_types) == {
        "person_name", "date_of_birth", "member_or_policy_id",
        "phone", "email", "address", "provider_npi", "part2_content",
    }, report.detected_types
    assert "ssn" not in report.detected_types and "mrn" not in report.detected_types
    assert report.exposure == 2.9
    print(f"  threshold 0.60 -> {len(report.detected_types)} of 10 types  : ok")

    strict = middleware_with(StubClassifier(PLANTED, exposure=2.9), threshold=0.90)
    strict_report = await strict._check("text", boundary="request")
    assert set(strict_report.detected_types) == {"person_name", "date_of_birth", "phone", "email"}
    print(f"  threshold 0.90 -> {len(strict_report.detected_types)} of 10 types  : ok")


async def test_fail_open():
    """A detector outage must not take the run down, and must not read clean."""
    mw = middleware_with(BrokenClassifier())
    report = await mw._check("text", boundary="request")
    assert report.status.startswith("unchecked"), report.status
    assert "TypeSafe unreachable" in report.status
    assert report.findings == [] and report.exposure == 0.0
    print(f"  detector outage -> {report.status[:46]}...")


# -- through a real agent --------------------------------------------------


async def test_both_boundaries_through_create_agent():
    classifier = StubClassifier(PLANTED, exposure=3.0)
    mw = middleware_with(classifier)
    agent = create_agent(
        model=StubModel(),
        system_prompt="triage the request",
        response_format=TriageDecision,
        middleware=[mw],
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("Member DOB: 9/6/97\nContact 4015550147")]}
    )

    reports = result["pii_reports"]
    assert [r.boundary for r in reports] == ["request", "response"], reports
    assert all(r.status == "checked" for r in reports)
    print(f"  boundaries checked                    : {[r.boundary for r in reports]}")

    # The request side got messages; the response side got serialised JSON.
    ingress, egress = classifier.seen
    assert isinstance(ingress, list) and isinstance(ingress[0], SystemMessage)
    assert isinstance(egress, str) and "O'Brien-Reyes" in egress
    print("  request side saw system + messages    : ok")
    print("  response side saw the tool-call answer: ok  <-- the decorator cannot")

    assert peak_exposure(reports) == 3.0
    assert "part2_content" in detected_types(reports)
    print(f"  detected                              : {detected_types(reports)}")


async def test_no_content_boundary():
    """An empty egress is reported unchecked, not clean."""

    class Response:
        structured_response = None
        result = [AIMessage(content="")]

    mw = middleware_with(StubClassifier(PLANTED))
    report = await mw._check(JevPIIDetectionMiddleware._egress_text(Response()),
                             boundary="response")
    assert report.status == "unchecked (no content)"
    print("  empty egress -> unchecked, not clean  : ok")


async def main_():
    print("\n== unit ==")
    test_egress_reads_the_structured_response()
    test_egress_falls_back_to_message_text()
    test_ingress_includes_the_system_message()
    await test_threshold()
    await test_fail_open()
    await test_no_content_boundary()

    print("\n== through create_agent ==")
    await test_both_boundaries_through_create_agent()

    print("\nall assertions passed")


asyncio.run(main_())

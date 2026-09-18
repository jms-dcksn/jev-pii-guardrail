"""PII detection on the model boundary, using the TypeSafe Jev model.

Detect and log. Nothing is masked and nothing is blocked.

Jev answers typed questions with probabilities, not spans. Ask it whether a
text carries a date of birth and you get 0.94; you do not get an offset. So a
Jev answer cannot drive a redaction. It can tell you what crossed the boundary
and how identifiable the person is, which is what this agent records.

## Why this is a LangChain middleware and not a UiPath @guardrail

The `@guardrail` decorator in `uipath_langchain.guardrails` cannot carry a
custom validator on a chat model. Two separate reasons, both verified against
uipath-langchain 0.18.11:

1. `_langchain_adapter._apply_llm_pre` calls the evaluator as
   `evaluator(text, PRE, None, None)` -- the text goes in `data` and both
   `input_data` and `output_data` are None. `CustomValidator.evaluate` ignores
   `data` and reads only those two, so it returns "Rule skipped: data
   unavailable at this stage" and the rule function is never called. TOOL scope
   passes real dicts and works; LLM and AGENT scope do not.

2. `_apply_llm_post` returns immediately unless `response.content` is a
   non-empty `str`. With `response_format=` set, LangChain binds a ToolStrategy
   and the model answers with a tool call, so content is empty. The output
   boundary is never checked at all.

`UiPathDeterministicGuardrailMiddleware`, the only local-rule middleware, is
tool scope only, so there is no supported UiPath route to a local custom
detector on the model boundary either.

`awrap_model_call` is. It sees the whole request -- system message included,
not just the last human turn -- and it sees `ModelResponse.structured_response`,
which is the egress surface reason 2 hides.
"""

import json
import logging
import os
from typing import Any, Awaitable, Callable, Sequence

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.types import Command
from langchain_typesafe import Noul, NoulCriteria, Score, TypeSafeClassifier
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

logger = logging.getLogger(__name__)

# Jev model to ask. "jev-latest" is the SDK default; pin it per-tenant if a
# demo has to be reproducible across a model release.
TYPESAFE_MODEL = os.getenv("TYPESAFE_MODEL", "jev-latest")

# Probability at or above which a Noul counts as a detection. Calibrate this on
# your own traffic -- it is a policy dial, not a property of the model. 0.60 is
# a starting point, deliberately below the cookbook's 0.70 action threshold
# because this guardrail only logs.
DETECTION_THRESHOLD = float(os.getenv("TYPESAFE_PII_THRESHOLD", "0.60"))

# The Score question's id. Held apart from the Nouls because it is graded on an
# ordered scale, not a yes/no.
EXPOSURE_ID = "exposure"


def _noul(instructions: str, yes: str, no: str) -> Noul:
    return Noul(instructions=instructions, criteria=NoulCriteria(true=yes, false=no))


# One narrow question per identifier type, asked together in a single request
# so they run in parallel and cannot see one another's answers.
#
# The types mirror what the masking variant of this agent removed before the
# model call: the UiPath PII service categories plus local rules for MRN and
# policy number. Findings here are therefore directly comparable with it.
PHI_BATTERY: dict[str, Any] = {
    "person_name": _noul(
        "Does this text name a specific real person: the member, a family member, "
        "or a treating or referring clinician?",
        yes="A real person is named. Evidence is first name and last name present such as 'John Smith' or 'Jane C. Thompson' ",
        no="No person is named. A clinical eponym such as 'McMurray test', "
        "'Crohn disease' or 'Epworth' is medical vocabulary, not a person.",
    ),
    "date_of_birth": _noul(
        "Does this text contain a person's date of birth, written either numerically "
        "or spelled out?",
        yes="A date of birth is present.",
        no="No date of birth. A submission date, a service date or a duration is not "
        "a date of birth.",
    ),
    "ssn": _noul(
        "Does this text contain a US Social Security Number?",
        yes="A nine-digit Social Security Number is present, hyphenated or not.",
        no="No Social Security Number. Other nine-digit or ten-digit numbers, such as "
        "a provider NPI or a claim number, do not count.",
    ),
    "mrn": _noul(
        "Does this text contain a medical record number?",
        yes="A medical record number is present, usually labelled MRN. MRN is a unique alphanumeric or numeric identifier assigned by a healthcare provider to track a specific patient's medical history, clinical documents, and charts within their hospital or clinic network.",
        no="No medical record number.",
    ),
    "member_or_policy_id": _noul(
        "Does this text contain a health-plan member identifier or policy number?",
        yes="A member id or policy number is present, for example a letter prefix "
        "followed by digit groups.",
        no="Neither is present. A prior-authorization case id such as PA-1021 "
        "identifies the request, not the member, and does not count.",
    ),
    "phone": _noul(
        "Does this text contain a telephone number?",
        yes="A telephone number is present, with or without punctuation.",
        no="No telephone number. A ten-digit provider NPI is not a telephone number.",
    ),
    "email": _noul(
        "Does this text contain an email address?",
        yes="An email address is present.",
        no="No email address.",
    ),
    "address": _noul(
        "Does this text contain a residential or street address, or a city together "
        "with a postcode?",
        yes="A street address, or a city with a postcode, is present.",
        no="No address. A plan name, a facility type or a bare state name is not an "
        "address.",
    ),
    "provider_npi": _noul(
        "Does this text contain a ten-digit National Provider Identifier?",
        yes="A provider NPI is present.",
        no="No provider NPI.",
    ),
    # Not an identifier. A content category, and the reason it lives in this
    # battery is that it replaces the PART2_PATTERN regex that redaction.py
    # owned and that this variant deletes along with the node.
    "part2_content": _noul(
        "Does this text describe substance-use disorder assessment or treatment, the "
        "kind of content protected under 42 CFR Part 2?",
        yes="Substance-use treatment content is present, for example detox, an "
        "intensive outpatient programme for substance use, or buprenorphine, "
        "methadone or naltrexone.",
        no="No substance-use treatment content. Mental-health content on its own, "
        "such as depression or anxiety, is not 42 CFR Part 2 content.",
    ),
    EXPOSURE_ID: Score(
        instructions="How identifiable is the patient from this text alone?",
        criteria=[
            "Not identifiable: clinical content only, with no identifier of any kind.",
            "Weakly identifiable: only indirect identifiers, such as a plan type, an "
            "age or a service date.",
            "Identifiable: one direct identifier, such as a member id, a medical "
            "record number or a date of birth.",
            "Highly identifiable: a person's name together with one or more other "
            "direct identifiers.",
        ],
    ),
}

#: Question ids that are identifier detections, in battery order.
PII_QUESTION_IDS: tuple[str, ...] = tuple(q for q in PHI_BATTERY if q != EXPOSURE_ID)


class PiiFinding(BaseModel):
    """What Jev said about one identifier type. Never carries a value."""

    pii_type: str = Field(description="Identifier type asked about, e.g. person_name.")
    probability: float = Field(
        ge=0.0, le=1.0, description="Jev's probability that this type is present."
    )
    detected: bool = Field(
        description="True when probability is at or above the detection threshold."
    )


class PiiDetectionReport(BaseModel):
    """One boundary, checked once.

    `status` exists so a boundary that was never checked cannot be mistaken for
    one that came back clean. Detection is fail-open -- a detector outage must
    not take the agent down -- and a silent skip is how you end up believing a
    boundary was inspected when it was not.
    """

    boundary: str = Field(description='"request" (into the model) or "response" (out of it).')
    status: str = Field(description='"checked", or "unchecked" with the reason.')
    exposure: float = Field(
        default=0.0,
        description="How identifiable the patient is from this text, 0.0 to 3.0.",
    )
    findings: list[PiiFinding] = Field(
        default_factory=list, description="One entry per identifier type asked about."
    )

    @property
    def detected_types(self) -> list[str]:
        return [f.pii_type for f in self.findings if f.detected]


class PiiDetectionState(AgentState):
    """Extra keys the middleware writes back onto the inner agent's state."""

    pii_reports: NotRequired[list[PiiDetectionReport]]


#: Longest reason kept on a report. A pydantic ValidationError runs to several
#: lines, and this string is shown in Action Center and in the job output.
_MAX_REASON = 160


def _reason(exc: BaseException) -> str:
    """One readable line. The traceback is in the log; this is for a human."""
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= _MAX_REASON else text[: _MAX_REASON - 3] + "..."


def _unchecked(boundary: str, exc: BaseException) -> PiiDetectionReport:
    return PiiDetectionReport(boundary=boundary, status=f"unchecked ({_reason(exc)})")


class JevPIIDetectionMiddleware(AgentMiddleware[PiiDetectionState]):
    """Detect PII on both sides of the model call. Log it. Change nothing.

    Example:
        ```python
        agent = create_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            response_format=TriageDecision,
            middleware=[JevPIIDetectionMiddleware()],
        )
        result = await agent.ainvoke({"messages": [HumanMessage(request)]})
        result["pii_reports"]  # one report per boundary
        ```

    Args:
        threshold: Probability at or above which a Noul counts as a detection.
        model: Jev model id.
        timeout: Seconds allowed for one TypeSafe request.
    """

    state_schema = PiiDetectionState

    def __init__(
        self,
        *,
        threshold: float = DETECTION_THRESHOLD,
        model: str = TYPESAFE_MODEL,
        timeout: float = 30.0,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self._model = model
        self._timeout = timeout
        self._classifier: TypeSafeClassifier | None = None

    @property
    def classifier(self) -> TypeSafeClassifier:
        """Built on first use, not at import.

        `TypeSafeClassifier` raises when TYPESAFE_API_KEY is absent. Building it
        here turns a missing key into an unchecked boundary the run reports,
        rather than an import-time crash in a node that has other work to do.
        """
        if self._classifier is None:
            if not os.getenv("TYPESAFE_API_KEY"):
                # Checked here rather than left to pydantic, whose multi-line
                # ValidationError is the single most likely thing to land in
                # this field and is unreadable in Action Center.
                raise RuntimeError(
                    "TYPESAFE_API_KEY is not set. Get a key from "
                    "https://console.typesafe.ai/settings/keys"
                )
            self._classifier = TypeSafeClassifier(
                questions=PHI_BATTERY, model=self._model, timeout=self._timeout
            )
        return self._classifier

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ExtendedModelResponse:
        """Check the request, run the model, check the response."""
        ingress = await self._check(self._ingress_state(request), boundary="request")

        response = await handler(request)

        egress = await self._check(self._egress_text(response), boundary="response")

        reports = [ingress, egress]
        self._log(reports)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"pii_reports": reports}),
        )

    # -- boundaries ---------------------------------------------------------

    @staticmethod
    def _ingress_state(request: ModelRequest) -> list[BaseMessage]:
        """Everything about to be sent, system message included.

        `TypeSafeClassifier` takes LangChain messages directly and serialises
        role and content itself, so no flattening is needed here.

        The built-in adapter checks only the last HumanMessage. On a returned
        case this agent's prompt carries the reviewer's free text, which is the
        likeliest cleartext PHI in the whole graph -- so the whole list goes.
        """
        if request.system_message is not None:
            return [request.system_message, *request.messages]
        return list(request.messages)

    @staticmethod
    def _egress_text(response: ModelResponse) -> str:
        """What actually came back out.

        With `response_format` set, the AIMessage content is empty and the
        answer is in `structured_response`. Serialising that is the only way to
        see the egress text at all, and it is exactly what the built-in POST
        hook misses.
        """
        structured = response.structured_response
        if structured is not None:
            payload = (
                structured.model_dump()
                if hasattr(structured, "model_dump")
                else structured
            )
            return json.dumps(payload, default=str)
        return "\n".join(
            text
            for message in response.result
            if isinstance(message, AIMessage) and (text := message.text)
        )

    # -- detection ----------------------------------------------------------

    async def _check(self, state: Any, *, boundary: str) -> PiiDetectionReport:
        """One TypeSafe request. Every question in the battery, in parallel."""
        if not state:
            return PiiDetectionReport(boundary=boundary, status="unchecked (no content)")
        try:
            answers = (await self.classifier.ainvoke(state)).answers
        except Exception as exc:  # noqa: BLE001 -- fail open, and say so
            logger.warning(
                "[JEV-PII] %s boundary not checked: %s", boundary, _reason(exc)
            )
            return _unchecked(boundary, exc)

        findings = [
            PiiFinding(
                pii_type=qid,
                probability=answers[qid].noul,
                detected=answers[qid].noul >= self.threshold,
            )
            for qid in PII_QUESTION_IDS
            if qid in answers
        ]
        exposure = answers[EXPOSURE_ID].score if EXPOSURE_ID in answers else 0.0
        return PiiDetectionReport(
            boundary=boundary, status="checked", exposure=exposure, findings=findings
        )

    def _log(self, reports: Sequence[PiiDetectionReport]) -> None:
        """One line per boundary. Types and probabilities, never values."""
        for report in reports:
            if report.status != "checked":
                continue
            detected = [f for f in report.findings if f.detected]
            if not detected:
                logger.info(
                    "[JEV-PII] %s: clean, exposure %.2f", report.boundary, report.exposure
                )
                continue
            logger.warning(
                "[JEV-PII] %s: %s | exposure %.2f (threshold %.2f)",
                report.boundary,
                ", ".join(f"{f.pii_type}={f.probability:.2f}" for f in detected),
                report.exposure,
                self.threshold,
            )


def detection_map(reports: Sequence[PiiDetectionReport]) -> dict[str, bool]:
    """Every question in the battery, answered true or false, across all boundaries.

    Fixed shape on purpose. The list form below is what a human reads, but it
    is the wrong thing to grade: `uipath-json-similarity` walks lists
    positionally, so one missed type shifts every later element and the score
    collapses for a single error. A map with all ten keys always present grades
    one leaf per identifier type, which is what precision and recall actually
    need.
    """
    hits = {t for report in reports for t in report.detected_types}
    return {qid: qid in hits for qid in PII_QUESTION_IDS}


def detected_types(reports: Sequence[PiiDetectionReport]) -> list[str]:
    """Distinct identifier types detected across every boundary, in battery order."""
    return [qid for qid, hit in detection_map(reports).items() if hit]


def peak_exposure(reports: Sequence[PiiDetectionReport]) -> float:
    """The worst exposure score seen on any boundary."""
    return max((report.exposure for report in reports), default=0.0)

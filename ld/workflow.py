"""Agentic workflow planner for deciding when tools are required."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from google import genai

ConversationTurn = Tuple[str, str]

WORKFLOW_PROMPT = """You are a planner for a retrieval-augmented banking assistant.
Decide how to handle the user's next question.

Return strict JSON with keys:
- intent: high-level category such as "retrieval", "recommendation", "calculation", or "other".
- needs_structured_tool: true/false. True only when answering correctly would require a structured product catalog or eligibility tool.
- reasoning: short English sentence explaining your decision.
- plan: array of up to 3 steps this assistant should follow in order.

Rules:
- Favor grounded answers based on the provided PDFs whenever possible. Requests for recommendations, improvements, or “pretend you are a strategist” should still be answered by citing the concrete initiatives, risks, or priorities described in the documents.
- Only set needs_structured_tool=true when pricing/eligibility data from the deferred product catalog is essential; otherwise keep it false and plan to answer with the retrieved context.
- When needs_structured_tool=true, include a plan step that calls the future `lookup_product` tool before answering.
- Keep the JSON compact and valid even if unsure; never include backticks or commentary.

Question: {question}
Recent history:
{history}
"""


@dataclass
class WorkflowDecision:
    intent: str
    needs_structured_tool: bool
    reasoning: str
    plan: List[str]

    def pretty(self) -> str:
        steps = "\n".join(f"  {idx+1}. {step}" for idx, step in enumerate(self.plan)) or "  (none)"
        return (
            "Workflow Decision:\n"
            f"- Intent: {self.intent}\n"
            f"- Needs structured tool: {self.needs_structured_tool}\n"
            f"- Reasoning: {self.reasoning}\n"
            f"- Plan:\n{steps}"
        )


def _format_history(history: Sequence[ConversationTurn]) -> str:
    if not history:
        return "None."
    chunks: List[str] = []
    for question, answer in history[-3:]:
        chunks.append(f"User: {question}\nAssistant: {answer}")
    return "\n\n".join(chunks)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _normalize_plan(value: object) -> List[str]:
    if isinstance(value, list):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        return normalized[:3]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def plan_workflow(
    client: genai.Client,
    *,
    question: str,
    model: str,
    history: Iterable[ConversationTurn] | None = None,
) -> WorkflowDecision:
    prompt = WORKFLOW_PROMPT.format(
        question=question.strip(),
        history=_format_history(list(history or [])),
    )
    response = client.models.generate_content(model=model, contents=prompt)
    raw = (response.text or "").strip()

    default = WorkflowDecision(
        intent="retrieval",
        needs_structured_tool=False,
        reasoning="Defaulted to retrieval flow.",
        plan=["Rewrite query if needed", "Retrieve filings", "Answer with citations"],
    )
    if not raw:
        return default

    parsed = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON snippet in case of stray text.
        start = raw.find("{")
        end = raw.rfind("}")
        if 0 <= start < end:
            snippet = raw[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return default

    intent = str(parsed.get("intent", default.intent)).strip() or default.intent
    needs_structured = _coerce_bool(
        parsed.get("needs_structured_tool")
        or parsed.get("needs_tool")
        or parsed.get("requires_tool")
    )
    reasoning = str(parsed.get("reasoning") or parsed.get("explanation") or "").strip()
    plan = _normalize_plan(parsed.get("plan"))

    return WorkflowDecision(
        intent=intent,
        needs_structured_tool=needs_structured,
        reasoning=reasoning or default.reasoning,
        plan=plan or default.plan,
    )

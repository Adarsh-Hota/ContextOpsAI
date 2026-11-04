from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from contextops.contracts import (
    AssertionResult,
    CaseResult,
    Embedding,
    EmbeddingIndex,
    EvidenceArtifacts,
    Plan,
    ReleasePolicy,
    ResultBundle,
    Snapshot,
    Usage,
    fingerprint,
    text_hash,
)
from contextops.selection import create_plan
from contextops.snapshot import build_snapshot


def write_source(root: Path, source_id: str, text: str, risk: str = "medium") -> None:
    path = root / "knowledge" / f"{source_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nid: {source_id}\ntitle: {source_id}\nrisk: {risk}\n---\n"
        f"<!-- contextops:section=window -->\n{text}\n",
        encoding="utf-8",
    )


def write_case(root: Path, case_id: str, dependency: str, risk: str = "high") -> None:
    path = root / "evals/cases" / f"{case_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 2,
        "id": case_id,
        "risk": risk,
        "prompt": f"Question for {case_id}?",
        "intent": f"Behaviour for {case_id}",
        "dependencies": [dependency],
        "expected_facts": ["Correct fact"],
        "assertions": [
            {
                "id": "structured-output",
                "category": "format",
                "description": "Valid structured output",
                "evaluator_version": "structured-v2",
            },
            {
                "id": "correct",
                "category": "correctness",
                "description": "Correct answer",
                "evaluator_version": "v1",
            },
        ],
        "smoke": case_id == "smoke",
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    write_source(tmp_path, "refund", "Refund window is 30 days.")
    write_source(tmp_path, "shipping", "Shipping is 5 days.")
    write_case(tmp_path, "refund-test", "refund#window")
    write_case(tmp_path, "semantic-test", "shipping#window", risk="medium")
    write_case(tmp_path, "unrelated", "shipping#window", risk="low")
    write_case(tmp_path, "smoke", "shipping#window", risk="critical")
    return tmp_path


def embeddings_for(baseline: Snapshot, candidate: Snapshot) -> EmbeddingIndex:
    entries: dict[str, tuple[float, ...]] = {}
    for snap in (baseline, candidate):
        for doc in snap.documents:
            for section in doc.sections:
                entries[text_hash(section.text)] = (1.0, 0.0) if doc.id == "refund" else (0.0, 1.0)
    for case in candidate.cases:
        entries[text_hash(case.semantic_text)] = (
            (1.0, 0.0) if case.id in {"refund-test", "semantic-test"} else (0.0, 1.0)
        )
    return EmbeddingIndex(
        baseline_id=baseline.snapshot_id,
        candidate_id=candidate.snapshot_id,
        case_embeddings=tuple(
            Embedding(
                text_hash=text_hash(c.semantic_text),
                values=(1.0, 0.0) if c.id in {"refund-test", "semantic-test"} else (0.0, 1.0),
            )
            for c in candidate.cases
        ),
        model="test-only-two-dimensional-vectors",
        dimensions=2,
        embeddings=tuple(Embedding(text_hash=k, values=v) for k, v in sorted(entries.items())),
    )


@pytest.fixture
def planned(project: Path) -> tuple[Plan, ReleasePolicy]:
    baseline = build_snapshot(project, "base")
    write_source(project, "refund", "Refund window is 14 days.")
    candidate = build_snapshot(project, "head")
    policy = ReleasePolicy(id="test-policy")
    plan = create_plan(
        candidate,
        policy,
        "agent-build",
        fingerprint({"model": "test-provider"}),
        baseline=baseline,
        embeddings=embeddings_for(baseline, candidate),
        embedding_configuration_hash=embeddings_for(baseline, candidate).configuration_hash,
        judge_configuration_hash=fingerprint("judge"),
        evaluator_hash=fingerprint("evaluator"),
        baseline_trusted=True,
        evaluator_qualification_hash=fingerprint("test-only-calibration"),
    )
    return plan, policy


@pytest.fixture
def passing_results() -> Callable[[Plan], ResultBundle]:
    def make(plan: Plan) -> ResultBundle:
        return ResultBundle(
            evaluator_qualification_hash=plan.evaluator_qualification_hash,
            run_id=plan.run_id,
            attempt_id=plan.attempt_id,
            candidate_content_id=plan.candidate_content_id,
            embedding_configuration_hash=plan.embedding_configuration_hash,
            judge_configuration_hash=plan.judge_configuration_hash,
            evaluator_hash=plan.evaluator_hash,
            plan_id=plan.plan_id,
            candidate_id=plan.candidate_id,
            candidate_revision=plan.candidate_revision,
            agent_version=plan.agent_version,
            model_configuration_hash=plan.model_configuration_hash,
            cases=tuple(
                CaseResult(
                    case_id=s.case.id,
                    case_hash=s.case.case_hash,
                    expectation_hash=s.case.expectation_hash,
                    assertion_hash=s.case.assertion_hash,
                    evaluator_hash=s.case.evaluator_hash,
                    artifacts=EvidenceArtifacts(
                        answer_hash=fingerprint("test-answer"),
                        retrieval_hash=fingerprint("test-retrieval"),
                        judgment_hash=fingerprint("test-judgment"),
                    ),
                    usage=Usage(input_tokens=0, output_tokens=0, cost_usd=0.0, latency_ms=0.0),
                    assertions=tuple(
                        AssertionResult(id=a.id, status="pass", reason="Verified by test evaluator")
                        for a in s.case.assertions
                    ),
                )
                for s in plan.selected
            ),
        )

    return make


def replace_json(model: ResultBundle, **updates: object) -> ResultBundle:
    return ResultBundle.model_validate_json(
        json.dumps({**model.model_dump(mode="json"), **updates})
    )

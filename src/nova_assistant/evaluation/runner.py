from time import perf_counter
from typing import Protocol

from nova_assistant.decision import (
    DecisionStatus,
    EvidenceGateConfig,
    evaluate_evidence,
    normalize_text,
)
from nova_assistant.evaluation.models import EvaluationCase
from nova_assistant.evaluation.results import (
    EvaluationCaseResult,
    EvaluationChecks,
    EvaluationMode,
)
from nova_assistant.retrieval import (
    RetrievalRequest,
    RetrievalResult,
)

DEFAULT_EVALUATION_TOP_K = 5


class Retriever(Protocol):
    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResult: ...


class EvaluationRunner:
    def __init__(
        self,
        retriever: Retriever,
        gate_config: EvidenceGateConfig | None = None,
        top_k: int = DEFAULT_EVALUATION_TOP_K,
    ) -> None:
        if not 1 <= top_k <= 20:
            raise ValueError("top_k doit être compris entre 1 et 20.")

        self.retriever = retriever
        self.gate_config = gate_config
        self.top_k = top_k

    def evaluate_offline_case(
        self,
        case: EvaluationCase,
    ) -> EvaluationCaseResult:
        total_started_at = perf_counter()
        retrieval_started_at = perf_counter()

        try:
            retrieval = self.retriever.retrieve(
                RetrievalRequest(
                    question=case.question,
                    selection_request=(case.to_selection_request()),
                    top_k=self.top_k,
                )
            )
        except Exception as error:
            total_duration_ms = (perf_counter() - total_started_at) * 1000

            return EvaluationCaseResult(
                case_id=case.case_id,
                category=case.category,
                product=case.product,
                mode=EvaluationMode.OFFLINE,
                expected_status=case.expected_status,
                observed_status="error",
                expected_document_ids=(case.expected_document_ids),
                expected_evidence_terms=(case.expected_evidence_terms),
                forbidden_answer_terms=(case.forbidden_answer_terms),
                checks=EvaluationChecks(
                    status=False,
                    selection=False,
                    retrieval_scope=False,
                    evidence=False,
                ),
                retrieval_duration_ms=total_duration_ms,
                total_duration_ms=total_duration_ms,
                error=(f"{type(error).__name__}: {error}"),
            )

        retrieval_duration_ms = (perf_counter() - retrieval_started_at) * 1000

        decision = evaluate_evidence(
            retrieval,
            config=self.gate_config,
        )

        selected_document_ids = retrieval.selection.allowed_document_ids
        retrieved_document_ids = tuple(
            dict.fromkeys(match.chunk.document_id for match in retrieval.matches)
        )
        retrieved_chunk_ids = tuple(match.chunk.chunk_id for match in retrieval.matches)

        retrieved_text = normalize_text("\n".join(match.chunk.text for match in retrieval.matches))

        found_evidence_terms = tuple(
            term for term in case.expected_evidence_terms if normalize_text(term) in retrieved_text
        )
        missing_evidence_terms = tuple(
            term for term in case.expected_evidence_terms if term not in found_evidence_terms
        )

        if case.expected_status.value == "answered":
            status_passed = decision.status is DecisionStatus.GENERATION_ALLOWED
        else:
            status_passed = decision.status.value == case.expected_status.value

        allowed_document_ids = set(selected_document_ids)
        retrieval_scope_passed = all(
            document_id in allowed_document_ids for document_id in retrieved_document_ids
        )

        checks = EvaluationChecks(
            status=status_passed,
            selection=(selected_document_ids == case.expected_document_ids),
            retrieval_scope=retrieval_scope_passed,
            evidence=not missing_evidence_terms,
        )

        total_duration_ms = (perf_counter() - total_started_at) * 1000

        return EvaluationCaseResult(
            case_id=case.case_id,
            category=case.category,
            product=case.product,
            mode=EvaluationMode.OFFLINE,
            expected_status=case.expected_status,
            observed_status=decision.status.value,
            expected_document_ids=(case.expected_document_ids),
            selected_document_ids=selected_document_ids,
            retrieved_document_ids=retrieved_document_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            expected_evidence_terms=(case.expected_evidence_terms),
            found_evidence_terms=found_evidence_terms,
            missing_evidence_terms=missing_evidence_terms,
            forbidden_answer_terms=(case.forbidden_answer_terms),
            checks=checks,
            retrieval_duration_ms=retrieval_duration_ms,
            total_duration_ms=total_duration_ms,
        )

    def run_offline(
        self,
        cases: tuple[EvaluationCase, ...],
    ) -> tuple[EvaluationCaseResult, ...]:
        return tuple(self.evaluate_offline_case(case) for case in cases)

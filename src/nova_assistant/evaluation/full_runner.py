from time import perf_counter

from nova_assistant.decision import (
    AnswerGenerator,
    AnswerService,
    EvidenceGateConfig,
    normalize_text,
)
from nova_assistant.evaluation.models import EvaluationCase
from nova_assistant.evaluation.results import (
    EvaluationCaseResult,
    EvaluationChecks,
    EvaluationMode,
)
from nova_assistant.evaluation.runner import (
    DEFAULT_EVALUATION_TOP_K,
    EvaluationRunner,
    Retriever,
)
from nova_assistant.retrieval import (
    RetrievalRequest,
    RetrievalResult,
)


class FullEvaluationRunner(EvaluationRunner):
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        gate_config: EvidenceGateConfig | None = None,
        top_k: int = DEFAULT_EVALUATION_TOP_K,
        max_sources: int = 5,
    ) -> None:
        super().__init__(
            retriever=retriever,
            gate_config=gate_config,
            top_k=top_k,
        )
        self.answer_service = AnswerService(
            generator=generator,
            gate_config=gate_config,
            max_sources=max_sources,
        )

    def evaluate_full_case(
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
            duration_ms = (perf_counter() - total_started_at) * 1000

            return self._build_error_result(
                case=case,
                error=error,
                retrieval_duration_ms=duration_ms,
                total_duration_ms=duration_ms,
            )

        retrieval_duration_ms = (perf_counter() - retrieval_started_at) * 1000
        generation_started_at = perf_counter()

        try:
            response = self.answer_service.respond(retrieval)
        except Exception as error:
            generation_duration_ms = (perf_counter() - generation_started_at) * 1000
            total_duration_ms = (perf_counter() - total_started_at) * 1000

            return self._build_error_result(
                case=case,
                error=error,
                retrieval=retrieval,
                retrieval_duration_ms=(retrieval_duration_ms),
                generation_duration_ms=(generation_duration_ms),
                total_duration_ms=total_duration_ms,
            )

        generation_duration_ms = (perf_counter() - generation_started_at) * 1000
        total_duration_ms = (perf_counter() - total_started_at) * 1000

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

        normalized_answer = normalize_text(response.answer)
        found_forbidden_terms = tuple(
            term
            for term in case.forbidden_answer_terms
            if normalize_text(term) in normalized_answer
        )

        citation_source_ids = tuple(citation.source_id for citation in response.citations)
        citation_document_ids = tuple(citation.document_id for citation in response.citations)

        if case.requires_citations:
            citations_passed = (
                bool(response.citations)
                and len(citation_source_ids) == len(set(citation_source_ids))
                and all(
                    document_id in selected_document_ids for document_id in citation_document_ids
                )
            )
        else:
            citations_passed = not response.citations

        allowed_document_ids = set(selected_document_ids)
        retrieval_scope_passed = all(
            document_id in allowed_document_ids for document_id in retrieved_document_ids
        )

        checks = EvaluationChecks(
            status=(response.status is case.expected_status),
            selection=(selected_document_ids == case.expected_document_ids),
            retrieval_scope=retrieval_scope_passed,
            evidence=not missing_evidence_terms,
            citations=citations_passed,
            forbidden_terms=(not found_forbidden_terms),
        )

        return EvaluationCaseResult(
            case_id=case.case_id,
            category=case.category,
            product=case.product,
            mode=EvaluationMode.FULL,
            expected_status=case.expected_status,
            observed_status=response.status.value,
            expected_document_ids=(case.expected_document_ids),
            selected_document_ids=selected_document_ids,
            retrieved_document_ids=retrieved_document_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            expected_evidence_terms=(case.expected_evidence_terms),
            found_evidence_terms=found_evidence_terms,
            missing_evidence_terms=missing_evidence_terms,
            forbidden_answer_terms=(case.forbidden_answer_terms),
            found_forbidden_terms=(found_forbidden_terms),
            citation_source_ids=citation_source_ids,
            citation_document_ids=(citation_document_ids),
            answer=response.answer,
            model_name=response.model_name,
            checks=checks,
            retrieval_duration_ms=(retrieval_duration_ms),
            generation_duration_ms=(generation_duration_ms),
            total_duration_ms=total_duration_ms,
        )

    def run_full(
        self,
        cases: tuple[EvaluationCase, ...],
    ) -> tuple[EvaluationCaseResult, ...]:
        return tuple(self.evaluate_full_case(case) for case in cases)

    def _build_error_result(
        self,
        *,
        case: EvaluationCase,
        error: Exception,
        retrieval_duration_ms: float,
        total_duration_ms: float,
        retrieval: RetrievalResult | None = None,
        generation_duration_ms: float | None = None,
    ) -> EvaluationCaseResult:
        selected_document_ids = (
            retrieval.selection.allowed_document_ids if retrieval is not None else ()
        )
        retrieved_document_ids = (
            tuple(dict.fromkeys(match.chunk.document_id for match in retrieval.matches))
            if retrieval is not None
            else ()
        )
        retrieved_chunk_ids = (
            tuple(match.chunk.chunk_id for match in retrieval.matches)
            if retrieval is not None
            else ()
        )

        return EvaluationCaseResult(
            case_id=case.case_id,
            category=case.category,
            product=case.product,
            mode=EvaluationMode.FULL,
            expected_status=case.expected_status,
            observed_status="error",
            expected_document_ids=(case.expected_document_ids),
            selected_document_ids=selected_document_ids,
            retrieved_document_ids=retrieved_document_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            expected_evidence_terms=(case.expected_evidence_terms),
            forbidden_answer_terms=(case.forbidden_answer_terms),
            checks=EvaluationChecks(
                status=False,
                selection=(selected_document_ids == case.expected_document_ids),
                retrieval_scope=False,
                evidence=False,
                citations=False,
                forbidden_terms=False,
            ),
            retrieval_duration_ms=(retrieval_duration_ms),
            generation_duration_ms=(generation_duration_ms),
            total_duration_ms=total_duration_ms,
            error=f"{type(error).__name__}: {error}",
        )

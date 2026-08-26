import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nova_assistant.decision import AssistantStatus
from nova_assistant.domain import ProductType
from nova_assistant.evaluation import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationCategory,
    EvaluationChecks,
    EvaluationMode,
    EvaluationRunner,
    FullEvaluationRunner,
    build_evaluation_summary,
    save_evaluation_report,
)
from nova_assistant.filtering import (
    SelectionStatus,
    select_corpus,
)
from nova_assistant.generation import (
    EvidenceSource,
    GeneratedAnswer,
    GenerationRequest,
)
from nova_assistant.indexing import (
    IndexedChunk,
    SemanticSearchResult,
)
from nova_assistant.ingestion import load_ingested_pages
from nova_assistant.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)


class FakeRetriever:
    def __init__(
        self,
        *,
        text: str = ("La franchise collision est de 350 € par sinistre."),
        raises: bool = False,
    ) -> None:
        self.text = text
        self.raises = raises
        self.calls = 0

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResult:
        self.calls += 1

        if self.raises:
            raise RuntimeError("Erreur de retrieval simulée.")

        selection = select_corpus(request.selection_request)
        if selection.status is not SelectionStatus.SELECTED:
            return RetrievalResult(
                request=request,
                status=RetrievalStatus(selection.status.value),
                selection=selection,
                matches=(),
            )
        page = next(
            page for page in load_ingested_pages() if page.document_id == "NOVA-AUTO-IPID-2025"
        )
        words = self.text.split()
        chunk = IndexedChunk.from_page(
            page=page,
            chunk_number=1,
            text=self.text,
            word_start=0,
            word_end=len(words),
        )

        return RetrievalResult(
            request=request,
            status=RetrievalStatus.RETRIEVED,
            selection=selection,
            matches=(
                SemanticSearchResult(
                    chunk=chunk,
                    score=0.9,
                ),
            ),
        )


def make_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="auto_2025_franchise_collision",
        category=EvaluationCategory.DIRECT_ANSWER,
        question="Quelle est la franchise collision ?",
        product=ProductType.AUTO,
        version=2025,
        expected_status=AssistantStatus.ANSWERED,
        expected_document_ids=(
            "NOVA-AUTO-IPID-2025",
            "NOVA-AUTO-CG-2025",
        ),
        expected_evidence_terms=(
            "franchise collision",
            "350 €",
        ),
        forbidden_answer_terms=("500 €",),
        requires_citations=True,
    )


def make_missing_context_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="auto_missing_context",
        category=EvaluationCategory.MISSING_CONTEXT,
        question="Quelle est ma franchise collision ?",
        product=ProductType.AUTO,
        expected_status=(AssistantStatus.CLARIFICATION_REQUIRED),
        requires_citations=False,
    )


def make_result(
    *,
    case_id: str = "case_one",
    product: ProductType = ProductType.AUTO,
    passed: bool = True,
    mode: EvaluationMode = EvaluationMode.OFFLINE,
) -> EvaluationCaseResult:
    checks = EvaluationChecks(
        status=passed,
        selection=True,
        retrieval_scope=True,
        evidence=True,
        citations=(True if mode is EvaluationMode.FULL else None),
        forbidden_terms=(True if mode is EvaluationMode.FULL else None),
    )

    return EvaluationCaseResult(
        case_id=case_id,
        category=EvaluationCategory.DIRECT_ANSWER,
        product=product,
        mode=mode,
        expected_status=AssistantStatus.ANSWERED,
        observed_status=("answered" if mode is EvaluationMode.FULL else "generation_allowed"),
        checks=checks,
        retrieval_duration_ms=10.0,
        generation_duration_ms=(20.0 if mode is EvaluationMode.FULL else None),
        total_duration_ms=(30.0 if mode is EvaluationMode.FULL else 10.0),
        answer=("La franchise est de 350 € [S1]." if mode is EvaluationMode.FULL else None),
        model_name=("test-model" if mode is EvaluationMode.FULL else None),
    )


def test_checks_compute_overall_result() -> None:
    successful = EvaluationChecks(
        status=True,
        selection=True,
        retrieval_scope=True,
        evidence=True,
    )
    failed = EvaluationChecks(
        status=False,
        selection=True,
        retrieval_scope=True,
        evidence=True,
    )

    assert successful.overall is True
    assert failed.overall is False


def test_offline_result_rejects_generation_fields() -> None:
    with pytest.raises(ValidationError, match="hors ligne"):
        EvaluationCaseResult(
            case_id="invalid_offline",
            category=EvaluationCategory.DIRECT_ANSWER,
            product=ProductType.AUTO,
            mode=EvaluationMode.OFFLINE,
            expected_status=AssistantStatus.ANSWERED,
            observed_status="generation_allowed",
            answer="Réponse interdite en mode hors ligne.",
            checks=EvaluationChecks(
                status=True,
                selection=True,
                retrieval_scope=True,
                evidence=True,
            ),
            retrieval_duration_ms=1.0,
            total_duration_ms=1.0,
        )


def test_runner_evaluates_supported_case() -> None:
    retriever = FakeRetriever()
    runner = EvaluationRunner(
        retriever=retriever,
        top_k=5,
    )

    result = runner.evaluate_offline_case(make_case())

    assert retriever.calls == 1
    assert result.observed_status == "generation_allowed"
    assert result.found_evidence_terms == (
        "franchise collision",
        "350 €",
    )
    assert result.missing_evidence_terms == ()
    assert result.checks.overall is True
    assert result.error is None


def test_runner_captures_retrieval_error() -> None:
    retriever = FakeRetriever(raises=True)
    runner = EvaluationRunner(retriever=retriever)

    result = runner.evaluate_offline_case(make_case())

    assert result.observed_status == "error"
    assert result.checks.overall is False
    assert result.error is not None
    assert "RuntimeError" in result.error


@pytest.mark.parametrize("top_k", [0, 21])
def test_runner_rejects_invalid_top_k(
    top_k: int,
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        EvaluationRunner(
            retriever=FakeRetriever(),
            top_k=top_k,
        )


def test_summary_aggregates_results() -> None:
    results = (
        make_result(
            case_id="auto_pass",
            product=ProductType.AUTO,
            passed=True,
        ),
        make_result(
            case_id="home_fail",
            product=ProductType.HOME,
            passed=False,
        ),
    )

    summary = build_evaluation_summary(results)

    assert summary.total == 2
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.pass_rate == 0.5
    assert summary.status_checks_passed == 1
    assert summary.by_product["auto"].pass_rate == 1.0
    assert summary.by_product["home"].pass_rate == 0.0


def test_summary_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="vide"):
        build_evaluation_summary(())


def test_summary_rejects_mixed_modes() -> None:
    results = (
        make_result(
            case_id="offline_case",
            mode=EvaluationMode.OFFLINE,
        ),
        make_result(
            case_id="full_case",
            mode=EvaluationMode.FULL,
        ),
    )

    with pytest.raises(ValueError, match="même mode"):
        build_evaluation_summary(results)


def test_report_is_saved_as_json_and_jsonl(
    tmp_path: Path,
) -> None:
    results = (
        make_result(case_id="case_one"),
        make_result(case_id="case_two"),
    )

    results_path, summary_path = save_evaluation_report(
        results=results,
        output_directory=tmp_path,
    )

    assert results_path.name == "offline_results.jsonl"
    assert summary_path.name == "offline_summary.json"

    result_lines = results_path.read_text(encoding="utf-8").splitlines()
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(result_lines) == 2
    assert summary_data["total"] == 2
    assert summary_data["passed"] == 2
    assert summary_data["pass_rate"] == 1.0


class FakeGenerator:
    def __init__(
        self,
        *,
        answer: str = ("La franchise collision est de 350 € par sinistre [S1]."),
        raises: bool = False,
    ) -> None:
        self.answer = answer
        self.raises = raises
        self.calls = 0

    def generate(
        self,
        request: GenerationRequest,
    ) -> GeneratedAnswer:
        self.calls += 1

        if self.raises:
            raise RuntimeError("Erreur de génération simulée.")

        source = EvidenceSource.from_match(
            request.retrieval_result.matches[0],
            source_number=1,
        )

        return GeneratedAnswer(
            answer=self.answer,
            model_name="fake-generator",
            citations=(source,),
        )


def test_full_runner_generates_answered_case() -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator()
    runner = FullEvaluationRunner(
        retriever=retriever,
        generator=generator,
        top_k=5,
        max_sources=5,
    )

    result = runner.evaluate_full_case(make_case())

    assert retriever.calls == 1
    assert generator.calls == 1
    assert result.observed_status == "answered"
    assert result.answer is not None
    assert result.model_name == "fake-generator"
    assert result.citation_source_ids == ("S1",)
    assert result.found_forbidden_terms == ()
    assert result.checks.overall is True
    assert result.error is None


def test_full_runner_does_not_generate_refusal() -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator()
    runner = FullEvaluationRunner(
        retriever=retriever,
        generator=generator,
    )

    result = runner.evaluate_full_case(make_missing_context_case())

    assert retriever.calls == 1
    assert generator.calls == 0
    assert result.observed_status == "clarification_required"
    assert result.citation_source_ids == ()
    assert result.model_name is None
    assert result.checks.overall is True


def test_full_runner_detects_forbidden_answer_term() -> None:
    generator = FakeGenerator(answer=("La franchise collision est de 500 € par sinistre [S1]."))
    runner = FullEvaluationRunner(
        retriever=FakeRetriever(),
        generator=generator,
    )

    result = runner.evaluate_full_case(make_case())

    assert result.found_forbidden_terms == ("500 €",)
    assert result.checks.forbidden_terms is False
    assert result.checks.overall is False


def test_full_runner_captures_generation_error() -> None:
    generator = FakeGenerator(raises=True)
    runner = FullEvaluationRunner(
        retriever=FakeRetriever(),
        generator=generator,
    )

    result = runner.evaluate_full_case(make_case())

    assert generator.calls == 1
    assert result.observed_status == "error"
    assert result.checks.overall is False
    assert result.error is not None
    assert "RuntimeError" in result.error

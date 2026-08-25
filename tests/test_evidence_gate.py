from datetime import date

from nova_assistant.decision import (
    AnswerService,
    AssistantStatus,
    DecisionStatus,
    EvidenceGateConfig,
    evaluate_evidence,
    extract_query_terms,
    normalize_text,
)
from nova_assistant.domain import ProductType
from nova_assistant.filtering import (
    SelectionRequest,
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


def make_retrieved_result(
    question: str,
    texts: tuple[str, ...],
    scores: tuple[float, ...] | None = None,
) -> RetrievalResult:
    request = RetrievalRequest(
        question=question,
        selection_request=SelectionRequest(
            product=ProductType.AUTO,
            version=2025,
        ),
        top_k=len(texts),
    )
    selection = select_corpus(request.selection_request)
    document_ids = (
        "NOVA-AUTO-IPID-2025",
        "NOVA-AUTO-CG-2025",
    )
    selected_scores = scores or tuple(0.9 - index * 0.05 for index in range(len(texts)))
    matches = []

    for index, (text, score) in enumerate(
        zip(texts, selected_scores, strict=True),
        start=1,
    ):
        document_id = document_ids[(index - 1) % len(document_ids)]
        page = next(page for page in load_ingested_pages() if page.document_id == document_id)
        word_count = len(text.split())
        chunk = IndexedChunk.from_page(
            page=page,
            chunk_number=index,
            text=text,
            word_start=0,
            word_end=word_count,
        )
        matches.append(
            SemanticSearchResult(
                chunk=chunk,
                score=score,
            )
        )

    return RetrievalResult(
        request=request,
        status=RetrievalStatus.RETRIEVED,
        selection=selection,
        matches=tuple(matches),
    )


def test_normalization_and_query_terms() -> None:
    assert normalize_text("L’information n’est pas décrite.") == "l information n est pas decrite"

    assert extract_query_terms("Quelle est la franchise collision en 2025 ?") == (
        "franchise",
        "collision",
    )


def test_evidence_gate_allows_supported_question() -> None:
    retrieval = make_retrieved_result(
        question="Quelle est la franchise collision ?",
        texts=("La franchise collision est de 350 € par sinistre.",),
    )

    decision = evaluate_evidence(retrieval)

    assert decision.status is DecisionStatus.GENERATION_ALLOWED
    assert decision.generation_allowed is True
    assert decision.signals.query_term_coverage == 1.0
    assert decision.signals.conflicting_values == ()


def test_evidence_gate_rejects_low_semantic_score() -> None:
    retrieval = make_retrieved_result(
        question="Quelle est la franchise collision ?",
        texts=("La franchise collision est de 350 € par sinistre.",),
        scores=(0.70,),
    )

    decision = evaluate_evidence(
        retrieval,
        EvidenceGateConfig(min_top_score=0.80),
    )

    assert decision.status is DecisionStatus.INSUFFICIENT_EVIDENCE
    assert "score" in decision.reason.lower()


def test_evidence_gate_rejects_low_term_coverage() -> None:
    retrieval = make_retrieved_result(
        question=("Quelle hospitalisation dentaire couvre les implants ?"),
        texts=("La franchise collision est de 350 € par sinistre.",),
    )

    decision = evaluate_evidence(retrieval)

    assert decision.status is DecisionStatus.INSUFFICIENT_EVIDENCE
    assert decision.signals.query_term_coverage == 0.0
    assert "termes" in decision.reason.lower()


def test_evidence_gate_rejects_explicit_absence() -> None:
    retrieval = make_retrieved_result(
        question=("Quels vaccins sont obligatoires pour ma destination ?"),
        texts=(
            "Les vaccins obligatoires ne sont pas décrits dans ce "
            "corpus. Le système ne doit pas inventer une réponse.",
        ),
    )

    decision = evaluate_evidence(retrieval)

    assert decision.status is DecisionStatus.INSUFFICIENT_EVIDENCE
    assert decision.signals.absence_markers
    assert "absente" in decision.reason.lower()


def test_evidence_gate_detects_conflicting_values() -> None:
    retrieval = make_retrieved_result(
        question="Quelle est la franchise collision ?",
        texts=(
            "La franchise collision est de 350 € par sinistre.",
            "La franchise collision est de 500 € par sinistre.",
        ),
    )

    decision = evaluate_evidence(retrieval)

    assert decision.status is DecisionStatus.CONFLICTING_SOURCES
    assert decision.signals.conflicting_values == (
        "350 €",
        "500 €",
    )


def test_evidence_gate_accepts_repeated_identical_value() -> None:
    retrieval = make_retrieved_result(
        question="Quelle est la franchise collision ?",
        texts=(
            "La franchise collision est de 350 € par sinistre.",
            "Une franchise collision de 350 € est appliquée.",
        ),
    )

    decision = evaluate_evidence(retrieval)

    assert decision.status is DecisionStatus.GENERATION_ALLOWED
    assert decision.signals.conflicting_values == ()


def test_evidence_gate_limits_evaluated_passages() -> None:
    retrieval = make_retrieved_result(
        question="Quelle est la franchise collision ?",
        texts=(
            "La franchise collision est de 350 € par sinistre.",
            "La franchise collision est de 350 € par sinistre.",
        ),
    )

    decision = evaluate_evidence(
        retrieval,
        EvidenceGateConfig(max_passages=1),
    )

    assert len(decision.signals.evaluated_chunk_ids) == 1


def test_evidence_gate_propagates_clarification() -> None:
    request = RetrievalRequest(
        question="Quelle est la franchise ?",
        selection_request=SelectionRequest(
            product=ProductType.AUTO,
        ),
    )
    selection = select_corpus(request.selection_request)
    retrieval = RetrievalResult(
        request=request,
        status=RetrievalStatus.CLARIFICATION_REQUIRED,
        selection=selection,
    )

    decision = evaluate_evidence(retrieval)

    assert decision.status is DecisionStatus.CLARIFICATION_REQUIRED
    assert decision.generation_allowed is False


def test_evidence_gate_propagates_conflicting_context() -> None:
    request = RetrievalRequest(
        question="Quelle est la franchise ?",
        selection_request=SelectionRequest(
            product=ProductType.AUTO,
            version=2024,
            contract_date=date(2025, 6, 1),
        ),
    )
    selection = select_corpus(request.selection_request)
    retrieval = RetrievalResult(
        request=request,
        status=RetrievalStatus.CONFLICTING_CONTEXT,
        selection=selection,
    )

    decision = evaluate_evidence(retrieval)

    assert decision.status is DecisionStatus.CONFLICTING_CONTEXT


def test_evidence_gate_propagates_no_matching_corpus() -> None:
    request = RetrievalRequest(
        question="Que couvre Travel 2024 ?",
        selection_request=SelectionRequest(
            product=ProductType.TRAVEL,
            version=2024,
        ),
    )
    selection = select_corpus(request.selection_request)
    retrieval = RetrievalResult(
        request=request,
        status=RetrievalStatus.NO_MATCHING_CORPUS,
        selection=selection,
    )

    decision = evaluate_evidence(retrieval)

    assert decision.status is DecisionStatus.NO_MATCHING_CORPUS


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        request: GenerationRequest,
    ) -> GeneratedAnswer:
        self.calls += 1
        source = EvidenceSource.from_match(
            match=request.retrieval_result.matches[0],
            source_number=1,
        )

        return GeneratedAnswer(
            answer="La franchise est de 350 € [S1].",
            model_name="fake-generator",
            citations=(source,),
        )


def test_answer_service_generates_after_authorization() -> None:
    retrieval = make_retrieved_result(
        question="Quelle est la franchise collision ?",
        texts=("La franchise collision est de 350 € par sinistre.",),
    )
    generator = FakeGenerator()
    service = AnswerService(generator=generator)

    response = service.respond(retrieval)

    assert response.status is AssistantStatus.ANSWERED
    assert response.answer == "La franchise est de 350 € [S1]."
    assert len(response.citations) == 1
    assert response.model_name == "fake-generator"
    assert generator.calls == 1


def test_answer_service_does_not_generate_without_evidence() -> None:
    retrieval = make_retrieved_result(
        question=("Quels vaccins sont obligatoires pour ma destination ?"),
        texts=(
            "Les vaccins obligatoires ne sont pas décrits dans ce "
            "corpus. Le système ne doit pas inventer une réponse.",
        ),
    )
    generator = FakeGenerator()
    service = AnswerService(generator=generator)

    response = service.respond(retrieval)

    assert response.status is AssistantStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == ()
    assert response.model_name is None
    assert generator.calls == 0


def test_answer_service_does_not_generate_on_conflicting_sources() -> None:
    retrieval = make_retrieved_result(
        question="Quelle est la franchise collision ?",
        texts=(
            "La franchise collision est de 350 € par sinistre.",
            "La franchise collision est de 500 € par sinistre.",
        ),
    )
    generator = FakeGenerator()
    service = AnswerService(generator=generator)

    response = service.respond(retrieval)

    assert response.status is AssistantStatus.CONFLICTING_SOURCES
    assert response.citations == ()
    assert generator.calls == 0


def test_answer_service_does_not_generate_for_clarification() -> None:
    request = RetrievalRequest(
        question="Quelle est la franchise ?",
        selection_request=SelectionRequest(
            product=ProductType.AUTO,
        ),
    )
    selection = select_corpus(request.selection_request)
    retrieval = RetrievalResult(
        request=request,
        status=RetrievalStatus.CLARIFICATION_REQUIRED,
        selection=selection,
    )
    generator = FakeGenerator()
    service = AnswerService(generator=generator)

    response = service.respond(retrieval)

    assert response.status is AssistantStatus.CLARIFICATION_REQUIRED
    assert response.citations == ()
    assert generator.calls == 0


def test_answer_service_rejects_invalid_source_limit() -> None:
    generator = FakeGenerator()

    for invalid_value in (0, 11):
        try:
            AnswerService(
                generator=generator,
                max_sources=invalid_value,
            )
        except ValueError as error:
            assert "entre 1 et 10" in str(error)
        else:
            raise AssertionError("Une limite de sources invalide aurait dû être rejetée.")

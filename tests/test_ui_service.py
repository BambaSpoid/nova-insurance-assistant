import pytest

from nova_assistant.filtering import (
    SelectionRequest,
    select_corpus,
)
from nova_assistant.generation import (
    GeneratedAnswer,
    GenerationRequest,
)
from nova_assistant.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)
from nova_assistant.ui import (
    MissingGenerationCredentialError,
    NovaAssistantService,
    UnavailableGenerator,
)


class StubRetriever:
    def __init__(
        self,
        result: RetrievalResult,
    ) -> None:
        self.result = result
        self.received_request: RetrievalRequest | None = None

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResult:
        self.received_request = request
        return self.result


class GeneratorThatMustNotRun:
    def generate(
        self,
        request: GenerationRequest,
    ) -> GeneratedAnswer:
        raise AssertionError("Le générateur ne devait pas être appelé.")


def build_clarification_retrieval(
    question: str,
    selection_request: SelectionRequest,
    top_k: int,
) -> RetrievalResult:
    request = RetrievalRequest(
        question=question,
        selection_request=selection_request,
        top_k=top_k,
    )

    return RetrievalResult(
        request=request,
        status=RetrievalStatus.CLARIFICATION_REQUIRED,
        selection=select_corpus(selection_request),
    )


def test_service_propagates_question_and_context() -> None:
    question = "Quelle est ma franchise ?"
    selection_request = SelectionRequest()
    retrieval_result = build_clarification_retrieval(
        question=question,
        selection_request=selection_request,
        top_k=7,
    )
    retriever = StubRetriever(retrieval_result)

    entry = NovaAssistantService(
        retriever=retriever,
        generator=GeneratorThatMustNotRun(),
        top_k=7,
    ).ask(
        question=question,
        selection_request=selection_request,
    )

    assert retriever.received_request is not None
    assert retriever.received_request.question == question
    assert retriever.received_request.top_k == 7
    assert retriever.received_request.selection_request == selection_request
    assert entry.question == question
    assert entry.selection_request == selection_request
    assert entry.response.status.value == "clarification_required"
    assert entry.response.citations == ()
    assert entry.response.model_name is None


@pytest.mark.parametrize(
    "top_k",
    (0, 21),
)
def test_service_rejects_invalid_top_k(
    top_k: int,
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        NovaAssistantService(
            retriever=object(),
            top_k=top_k,
        )


def test_unavailable_generator_raises_clear_error() -> None:
    with pytest.raises(
        MissingGenerationCredentialError,
        match="clé OpenAI",
    ):
        UnavailableGenerator().generate(
            request=object(),
        )


def test_presentation_labels_are_user_friendly() -> None:
    from nova_assistant.decision import AssistantStatus
    from nova_assistant.domain import ProductType
    from nova_assistant.ui import (
        product_label,
        status_label,
        status_tone,
    )

    assert product_label(ProductType.AUTO) == "Automobile"
    assert status_label(AssistantStatus.ANSWERED) == "Réponse documentée"
    assert status_tone(AssistantStatus.ANSWERED) == "success"


def test_every_product_has_suggested_questions() -> None:
    from nova_assistant.domain import ProductType
    from nova_assistant.ui import suggested_questions

    for product in ProductType:
        questions = suggested_questions(product)

        assert len(questions) == 3
        assert all(question.endswith("?") for question in questions)

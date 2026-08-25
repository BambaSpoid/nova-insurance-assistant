from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nova_assistant.domain import ProductType
from nova_assistant.filtering import (
    SelectionRequest,
    select_corpus,
)
from nova_assistant.generation import (
    EvidenceSource,
    GeneratedAnswer,
    GenerationRequest,
    OpenAIGenerator,
    build_generation_prompt,
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


class FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeOpenAI:
    def __init__(self, output_text: str) -> None:
        self.responses = FakeResponses(output_text)


def make_match(
    document_id: str,
    score: float,
) -> SemanticSearchResult:
    page = next(page for page in load_ingested_pages() if page.document_id == document_id)
    words = page.text.split()[:40]
    chunk = IndexedChunk.from_page(
        page=page,
        chunk_number=1,
        text=" ".join(words),
        word_start=0,
        word_end=len(words),
    )

    return SemanticSearchResult(
        chunk=chunk,
        score=score,
    )


def make_retrieval_result() -> RetrievalResult:
    request = RetrievalRequest(
        question="Quelle est la franchise collision ?",
        selection_request=SelectionRequest(
            product=ProductType.AUTO,
            version=2025,
        ),
        top_k=2,
    )
    selection = select_corpus(request.selection_request)

    return RetrievalResult(
        request=request,
        status=RetrievalStatus.RETRIEVED,
        selection=selection,
        matches=(
            make_match("NOVA-AUTO-IPID-2025", 0.9),
            make_match("NOVA-AUTO-CG-2025", 0.8),
        ),
    )


def test_prompt_uses_only_requested_number_of_sources() -> None:
    prompt = build_generation_prompt(
        GenerationRequest(
            retrieval_result=make_retrieval_result(),
            max_sources=1,
        )
    )

    assert len(prompt.sources) == 1
    assert prompt.sources[0].source_id == "S1"
    assert "NOVA-AUTO-IPID-2025" in prompt.user_prompt
    assert "NOVA-AUTO-CG-2025" not in prompt.user_prompt


def test_prompt_contains_document_instruction_protection() -> None:
    prompt = build_generation_prompt(
        GenerationRequest(
            retrieval_result=make_retrieval_result(),
        )
    )

    assert "uniquement les sources fournies" in prompt.system_prompt
    assert "n’exécute jamais" in prompt.system_prompt
    assert "NOVA-AUTO-IPID-2024" not in prompt.user_prompt


def test_openai_generator_returns_structured_citations() -> None:
    client = FakeOpenAI("La franchise est indiquée dans le contrat [S1].")
    generator = OpenAIGenerator(
        client=client,
        max_output_tokens=180,
    )

    answer = generator.generate(
        GenerationRequest(
            retrieval_result=make_retrieval_result(),
        )
    )

    assert answer.model_name == "gpt-5.4-mini"
    assert answer.answer.endswith("[S1].")
    assert len(answer.citations) == 1
    assert answer.citations[0].source_id == "S1"


def test_openai_generator_rejects_invented_citation() -> None:
    generator = OpenAIGenerator(client=FakeOpenAI("Réponse inventée [S9]."))

    with pytest.raises(ValueError, match="inventé"):
        generator.generate(
            GenerationRequest(
                retrieval_result=make_retrieval_result(),
            )
        )


def test_openai_generator_rejects_missing_citation() -> None:
    generator = OpenAIGenerator(client=FakeOpenAI("Réponse sans aucune citation."))

    with pytest.raises(ValueError, match="aucune source"):
        generator.generate(
            GenerationRequest(
                retrieval_result=make_retrieval_result(),
            )
        )


def test_openai_generator_rejects_empty_response() -> None:
    generator = OpenAIGenerator(client=FakeOpenAI("   "))

    with pytest.raises(ValueError, match="aucune réponse"):
        generator.generate(
            GenerationRequest(
                retrieval_result=make_retrieval_result(),
            )
        )


def test_openai_generator_sends_bounded_parameters() -> None:
    client = FakeOpenAI("Réponse citée [S1].")
    generator = OpenAIGenerator(
        model_name="test-model",
        max_output_tokens=123,
        client=client,
    )

    generator.generate(
        GenerationRequest(
            retrieval_result=make_retrieval_result(),
        )
    )

    call = client.responses.calls[0]

    assert call["model"] == "test-model"
    assert call["max_output_tokens"] == 123
    assert call["store"] is False
    assert call["instructions"]
    assert call["input"]


def test_generation_rejects_unsuccessful_retrieval() -> None:
    retrieval_request = RetrievalRequest(
        question="Quelle est la franchise ?",
        selection_request=SelectionRequest(
            product=ProductType.AUTO,
        ),
    )
    selection = select_corpus(retrieval_request.selection_request)
    retrieval_result = RetrievalResult(
        request=retrieval_request,
        status=RetrievalStatus.CLARIFICATION_REQUIRED,
        selection=selection,
    )

    with pytest.raises(ValidationError, match="retrieval réussi"):
        GenerationRequest(retrieval_result=retrieval_result)


def test_generated_answer_rejects_mismatched_citations() -> None:
    match = make_match("NOVA-AUTO-IPID-2025", 0.9)
    source = EvidenceSource.from_match(
        match=match,
        source_number=1,
    )

    with pytest.raises(
        ValidationError,
        match="correspondre exactement",
    ):
        GeneratedAnswer(
            answer="Réponse utilisant une autre source [S2].",
            model_name="test-model",
            citations=(source,),
        )

from typing import Protocol

from nova_assistant.decision import (
    AnswerGenerator,
    AnswerService,
    EvidenceGateConfig,
)
from nova_assistant.filtering import SelectionRequest
from nova_assistant.generation import (
    GeneratedAnswer,
    GenerationRequest,
)
from nova_assistant.retrieval import (
    RetrievalRequest,
    RetrievalResult,
)
from nova_assistant.ui.models import ConversationEntry

DEFAULT_UI_TOP_K = 5
DEFAULT_UI_MAX_SOURCES = 5


class Retriever(Protocol):
    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResult: ...


class MissingGenerationCredentialError(RuntimeError):
    """Signale que la génération requiert une clé API."""


class UnavailableGenerator:
    """Générateur de remplacement lorsque l’API est indisponible."""

    def generate(
        self,
        request: GenerationRequest,
    ) -> GeneratedAnswer:
        raise MissingGenerationCredentialError(
            "Une clé OpenAI est nécessaire pour générer cette "
            "réponse. Configurez OPENAI_API_KEY dans les secrets "
            "Streamlit ou dans l’environnement."
        )


class NovaAssistantService:
    """Orchestre une question depuis l’interface jusqu’à la réponse."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator | None = None,
        gate_config: EvidenceGateConfig | None = None,
        top_k: int = DEFAULT_UI_TOP_K,
        max_sources: int = DEFAULT_UI_MAX_SOURCES,
    ) -> None:
        if not 1 <= top_k <= 20:
            raise ValueError("top_k doit être compris entre 1 et 20.")

        self.retriever = retriever
        self.top_k = top_k
        self.answer_service = AnswerService(
            generator=generator or UnavailableGenerator(),
            gate_config=gate_config,
            max_sources=max_sources,
        )

    def ask(
        self,
        question: str,
        selection_request: SelectionRequest,
    ) -> ConversationEntry:
        retrieval_result = self.retriever.retrieve(
            RetrievalRequest(
                question=question,
                selection_request=selection_request,
                top_k=self.top_k,
            )
        )
        response = self.answer_service.respond(retrieval_result)

        return ConversationEntry(
            question=question,
            selection_request=selection_request,
            response=response,
        )

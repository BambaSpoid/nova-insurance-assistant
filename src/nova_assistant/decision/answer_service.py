from typing import Protocol

from nova_assistant.decision.evidence_gate import evaluate_evidence
from nova_assistant.decision.models import (
    AssistantResponse,
    AssistantStatus,
    EvidenceGateConfig,
)
from nova_assistant.generation import (
    GeneratedAnswer,
    GenerationRequest,
)
from nova_assistant.retrieval import RetrievalResult


class AnswerGenerator(Protocol):
    def generate(
        self,
        request: GenerationRequest,
    ) -> GeneratedAnswer: ...


class AnswerService:
    """Autorise la génération seulement après validation externe."""

    def __init__(
        self,
        generator: AnswerGenerator,
        gate_config: EvidenceGateConfig | None = None,
        max_sources: int = 5,
    ) -> None:
        if not 1 <= max_sources <= 10:
            raise ValueError("max_sources doit être compris entre 1 et 10.")

        self.generator = generator
        self.gate_config = gate_config
        self.max_sources = max_sources

    def respond(
        self,
        retrieval_result: RetrievalResult,
    ) -> AssistantResponse:
        decision = evaluate_evidence(
            retrieval_result=retrieval_result,
            config=self.gate_config,
        )

        if not decision.generation_allowed:
            return AssistantResponse(
                status=AssistantStatus(decision.status.value),
                answer=decision.reason,
                decision=decision,
            )

        generated_answer = self.generator.generate(
            GenerationRequest(
                retrieval_result=retrieval_result,
                max_sources=self.max_sources,
            )
        )

        return AssistantResponse(
            status=AssistantStatus.ANSWERED,
            answer=generated_answer.answer,
            decision=decision,
            citations=generated_answer.citations,
            model_name=generated_answer.model_name,
        )

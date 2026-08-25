from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from nova_assistant.generation import EvidenceSource
from nova_assistant.retrieval import (
    RetrievalResult,
    RetrievalStatus,
)


class DecisionStatus(StrEnum):
    GENERATION_ALLOWED = "generation_allowed"
    CLARIFICATION_REQUIRED = "clarification_required"
    CONFLICTING_CONTEXT = "conflicting_context"
    NO_MATCHING_CORPUS = "no_matching_corpus"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_SOURCES = "conflicting_sources"


class EvidenceGateConfig(BaseModel):
    """Paramètres externes utilisés pour évaluer les preuves."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_top_score: float = Field(default=0.80, ge=-1.0, le=1.0)
    min_query_term_coverage: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
    )
    max_passages: int = Field(default=5, ge=1, le=20)


class EvidenceSignals(BaseModel):
    """Mesures observables calculées sans demander l’avis du LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    top_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )
    query_terms: tuple[str, ...] = ()
    matched_query_terms: tuple[str, ...] = ()
    query_term_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    absence_markers: tuple[str, ...] = ()
    conflicting_values: tuple[str, ...] = ()
    evaluated_chunk_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def matched_terms_must_come_from_query(self) -> Self:
        if not set(self.matched_query_terms).issubset(self.query_terms):
            raise ValueError("Les termes correspondants doivent venir de la question.")

        return self


class EvidenceDecision(BaseModel):
    """Décision déterministe prise avant toute génération."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_result: RetrievalResult
    status: DecisionStatus
    reason: str = Field(min_length=1)
    signals: EvidenceSignals

    @field_validator("reason", mode="before")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        normalized_reason = value.strip()

        if not normalized_reason:
            raise ValueError("La justification de la décision est obligatoire.")

        return normalized_reason

    @model_validator(mode="after")
    def validate_status_consistency(self) -> Self:
        retrieval_status = self.retrieval_result.status

        propagated_statuses = {
            RetrievalStatus.CLARIFICATION_REQUIRED: (DecisionStatus.CLARIFICATION_REQUIRED),
            RetrievalStatus.CONFLICTING_CONTEXT: (DecisionStatus.CONFLICTING_CONTEXT),
            RetrievalStatus.NO_MATCHING_CORPUS: (DecisionStatus.NO_MATCHING_CORPUS),
        }

        if retrieval_status is not RetrievalStatus.RETRIEVED:
            expected_status = propagated_statuses[retrieval_status]

            if self.status is not expected_status:
                raise ValueError("La décision ne propage pas correctement le statut du retrieval.")

        elif self.status in {
            DecisionStatus.CLARIFICATION_REQUIRED,
            DecisionStatus.CONFLICTING_CONTEXT,
            DecisionStatus.NO_MATCHING_CORPUS,
        }:
            raise ValueError("Un retrieval réussi ne peut pas produire ce statut.")

        if (
            self.status is DecisionStatus.GENERATION_ALLOWED
            and not self.signals.evaluated_chunk_ids
        ):
            raise ValueError("Une génération autorisée exige des passages évalués.")

        return self

    @property
    def generation_allowed(self) -> bool:
        return self.status is DecisionStatus.GENERATION_ALLOWED


class AssistantStatus(StrEnum):
    ANSWERED = "answered"
    CLARIFICATION_REQUIRED = "clarification_required"
    CONFLICTING_CONTEXT = "conflicting_context"
    NO_MATCHING_CORPUS = "no_matching_corpus"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_SOURCES = "conflicting_sources"


class AssistantResponse(BaseModel):
    """Réponse finale, clarification ou refus contrôlé."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AssistantStatus
    answer: str = Field(min_length=1)
    decision: EvidenceDecision
    citations: tuple[EvidenceSource, ...] = ()
    model_name: str | None = None

    @field_validator("answer", mode="before")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        normalized_answer = value.strip()

        if not normalized_answer:
            raise ValueError("La réponse finale ne doit pas être vide.")

        return normalized_answer

    @model_validator(mode="after")
    def validate_final_response(self) -> Self:
        if self.status is AssistantStatus.ANSWERED:
            if not self.decision.generation_allowed:
                raise ValueError("Une réponse générée exige une autorisation.")

            if not self.citations:
                raise ValueError("Une réponse générée exige des citations.")

            if not self.model_name:
                raise ValueError("Une réponse générée exige un modèle.")

            return self

        if self.citations or self.model_name is not None:
            raise ValueError("Une clarification ou un refus ne doit pas contenir de génération.")

        expected_status = AssistantStatus(self.decision.status.value)

        if self.status is not expected_status:
            raise ValueError("Le statut final ne correspond pas à la décision.")

        return self

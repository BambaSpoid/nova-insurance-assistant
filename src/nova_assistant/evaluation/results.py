from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from nova_assistant.decision import AssistantStatus
from nova_assistant.domain import ProductType
from nova_assistant.evaluation.models import EvaluationCategory


class EvaluationMode(StrEnum):
    OFFLINE = "offline"
    FULL = "full"


class EvaluationChecks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: bool
    selection: bool
    retrieval_scope: bool
    evidence: bool
    citations: bool | None = None
    forbidden_terms: bool | None = None

    @computed_field
    @property
    def overall(self) -> bool:
        values = (
            self.status,
            self.selection,
            self.retrieval_scope,
            self.evidence,
            self.citations,
            self.forbidden_terms,
        )

        return all(value for value in values if value is not None)


class EvaluationCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    category: EvaluationCategory
    product: ProductType
    mode: EvaluationMode

    expected_status: AssistantStatus
    observed_status: str

    expected_document_ids: tuple[str, ...] = ()
    selected_document_ids: tuple[str, ...] = ()
    retrieved_document_ids: tuple[str, ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()

    expected_evidence_terms: tuple[str, ...] = ()
    found_evidence_terms: tuple[str, ...] = ()
    missing_evidence_terms: tuple[str, ...] = ()

    forbidden_answer_terms: tuple[str, ...] = ()
    found_forbidden_terms: tuple[str, ...] = ()

    citation_source_ids: tuple[str, ...] = ()
    citation_document_ids: tuple[str, ...] = ()

    answer: str | None = None
    model_name: str | None = None
    checks: EvaluationChecks

    retrieval_duration_ms: float = Field(ge=0)
    generation_duration_ms: float | None = Field(
        default=None,
        ge=0,
    )
    total_duration_ms: float = Field(ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_mode_specific_fields(self) -> Self:
        if self.mode is EvaluationMode.OFFLINE:
            if self.answer is not None:
                raise ValueError("Une évaluation hors ligne ne produit pas de réponse.")

            if self.model_name is not None:
                raise ValueError("Une évaluation hors ligne n’utilise pas de modèle de génération.")

            if self.generation_duration_ms is not None:
                raise ValueError("Une évaluation hors ligne ne mesure pas la génération.")

        return self


class EvaluationGroupSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)


class EvaluationRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: EvaluationMode
    total: int = Field(ge=1)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)

    status_checks_passed: int = Field(ge=0)
    selection_checks_passed: int = Field(ge=0)
    retrieval_scope_checks_passed: int = Field(ge=0)
    evidence_checks_passed: int = Field(ge=0)
    citation_checks_passed: int | None = Field(
        default=None,
        ge=0,
    )
    forbidden_terms_checks_passed: int | None = Field(
        default=None,
        ge=0,
    )
    generated_answers: int = Field(ge=0)

    observed_status_counts: dict[str, int]
    by_product: dict[str, EvaluationGroupSummary]
    by_category: dict[str, EvaluationGroupSummary]

    average_retrieval_duration_ms: float = Field(ge=0)
    average_total_duration_ms: float = Field(ge=0)

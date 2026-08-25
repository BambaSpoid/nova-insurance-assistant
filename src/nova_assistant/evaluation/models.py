from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from nova_assistant.decision import AssistantStatus
from nova_assistant.domain import DocumentType, ProductType
from nova_assistant.filtering import SelectionRequest


class EvaluationCategory(StrEnum):
    DIRECT_ANSWER = "direct_answer"
    VERSION_DIFFERENCE = "version_difference"
    EXCLUSION = "exclusion"
    RESTRICTION = "restriction"
    MISSING_INFORMATION = "missing_information"
    MISSING_CONTEXT = "missing_context"
    CONFLICTING_CONTEXT = "conflicting_context"
    MISSING_CORPUS = "missing_corpus"
    OUT_OF_SCOPE = "out_of_scope"


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(
        min_length=3,
        pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$",
    )
    category: EvaluationCategory
    question: str = Field(min_length=3)

    product: ProductType
    version: int | None = Field(default=None, ge=2020, le=2100)
    contract_date: date | None = None
    language: str = Field(default="fr", min_length=2, max_length=5)
    document_types: tuple[DocumentType, ...] | None = None

    expected_status: AssistantStatus
    expected_document_ids: tuple[str, ...] = ()
    expected_evidence_terms: tuple[str, ...] = ()
    forbidden_answer_terms: tuple[str, ...] = ()
    requires_citations: bool = False
    notes: str | None = None

    @field_validator("question", "notes")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        stripped = value.strip()

        if not stripped:
            raise ValueError("Le texte ne peut pas être vide.")

        return stripped

    @field_validator(
        "expected_document_ids",
        "expected_evidence_terms",
        "forbidden_answer_terms",
    )
    @classmethod
    def validate_unique_values(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        stripped_values = tuple(value.strip() for value in values)

        if any(not value for value in stripped_values):
            raise ValueError("Les valeurs ne peuvent pas être vides.")

        normalized_values = tuple(value.casefold() for value in stripped_values)

        if len(normalized_values) != len(set(normalized_values)):
            raise ValueError("Les valeurs doivent être uniques.")

        return stripped_values

    @field_validator("document_types")
    @classmethod
    def validate_document_types(
        cls,
        values: tuple[DocumentType, ...] | None,
    ) -> tuple[DocumentType, ...] | None:
        if values is None:
            return None

        if not values:
            raise ValueError("document_types ne peut pas être vide.")

        if len(values) != len(set(values)):
            raise ValueError("Les types de documents doivent être uniques.")

        return values

    @model_validator(mode="after")
    def validate_expectations(self) -> Self:
        if self.expected_status is AssistantStatus.ANSWERED:
            if not self.expected_document_ids:
                raise ValueError("Une réponse attendue doit définir ses documents.")

            if not self.expected_evidence_terms:
                raise ValueError("Une réponse attendue doit définir ses preuves.")

            if not self.requires_citations:
                raise ValueError("Une réponse attendue doit exiger des citations.")

        elif self.requires_citations:
            raise ValueError("Un refus ou une clarification ne doit pas exiger de citations.")

        return self

    def to_selection_request(self) -> SelectionRequest:
        return SelectionRequest(
            product=self.product,
            version=self.version,
            contract_date=self.contract_date,
            language=self.language,
            document_types=self.document_types,
        )

from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from nova_assistant.domain import (
    DocumentMetadata,
    DocumentType,
    ProductType,
)


class SelectionStatus(StrEnum):
    SELECTED = "selected"
    CLARIFICATION_REQUIRED = "clarification_required"
    CONFLICTING_CONTEXT = "conflicting_context"
    NO_MATCHING_CORPUS = "no_matching_corpus"


class SelectionRequest(BaseModel):
    """Contexte exact utilisé pour sélectionner le corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    product: ProductType | None = None
    version: int | None = Field(default=None, ge=1900, le=2100)
    contract_date: date | None = None
    language: Literal["fr"] = "fr"
    document_types: tuple[DocumentType, ...] | None = Field(
        default=None,
        min_length=1,
    )

    @field_validator("document_types")
    @classmethod
    def document_types_must_be_unique(
        cls,
        value: tuple[DocumentType, ...] | None,
    ) -> tuple[DocumentType, ...] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("Les types de documents doivent être uniques.")

        return value


class CorpusSelection(BaseModel):
    """Résultat explicite de la sélection exacte du corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: SelectionRequest
    status: SelectionStatus
    documents: tuple[DocumentMetadata, ...] = ()
    reason: str = Field(min_length=1)

    @field_validator("reason", mode="before")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        normalized_reason = value.strip()

        if not normalized_reason:
            raise ValueError("La justification de la sélection est obligatoire.")

        return normalized_reason

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        document_ids = [document.document_id for document in self.documents]

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("La sélection contient des documents dupliqués.")

        if self.status is SelectionStatus.SELECTED:
            if not self.documents:
                raise ValueError("Une sélection réussie doit contenir des documents.")
        elif self.documents:
            raise ValueError("Une décision sans sélection ne doit contenir aucun document.")

        return self

    @property
    def allowed_document_ids(self) -> tuple[str, ...]:
        return tuple(document.document_id for document in self.documents)

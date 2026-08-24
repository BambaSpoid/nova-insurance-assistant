from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from nova_assistant.filtering import (
    CorpusSelection,
    SelectionRequest,
    SelectionStatus,
)
from nova_assistant.indexing import SemanticSearchResult


class RetrievalStatus(StrEnum):
    RETRIEVED = "retrieved"
    CLARIFICATION_REQUIRED = "clarification_required"
    CONFLICTING_CONTEXT = "conflicting_context"
    NO_MATCHING_CORPUS = "no_matching_corpus"


SELECTION_TO_RETRIEVAL_STATUS = {
    SelectionStatus.CLARIFICATION_REQUIRED: (RetrievalStatus.CLARIFICATION_REQUIRED),
    SelectionStatus.CONFLICTING_CONTEXT: (RetrievalStatus.CONFLICTING_CONTEXT),
    SelectionStatus.NO_MATCHING_CORPUS: (RetrievalStatus.NO_MATCHING_CORPUS),
}


class RetrievalRequest(BaseModel):
    """Question accompagnée de son contexte de sélection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1)
    selection_request: SelectionRequest
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question", mode="before")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        normalized_question = value.strip()

        if not normalized_question:
            raise ValueError("La question ne doit pas être vide.")

        return normalized_question


class RetrievalResult(BaseModel):
    """Résultat traçable du filtre et de la recherche vectorielle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: RetrievalRequest
    status: RetrievalStatus
    selection: CorpusSelection
    matches: tuple[SemanticSearchResult, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.selection.request != self.request.selection_request:
            raise ValueError("La sélection ne correspond pas à la demande de retrieval.")

        if self.status is RetrievalStatus.RETRIEVED:
            self._validate_retrieved_matches()
        else:
            self._validate_non_retrieved_result()

        return self

    def _validate_retrieved_matches(self) -> None:
        if self.selection.status is not SelectionStatus.SELECTED:
            raise ValueError("Un retrieval réussi exige une sélection réussie.")

        if not self.matches:
            raise ValueError("Un retrieval réussi doit contenir des passages.")

        if len(self.matches) > self.request.top_k:
            raise ValueError("Le nombre de passages dépasse top_k.")

        chunk_ids = [match.chunk.chunk_id for match in self.matches]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Le résultat contient des passages dupliqués.")

        allowed_ids = set(self.selection.allowed_document_ids)

        if any(match.chunk.document_id not in allowed_ids for match in self.matches):
            raise ValueError("Un passage provient d’un document non autorisé.")

        scores = [match.score for match in self.matches]

        if scores != sorted(scores, reverse=True):
            raise ValueError("Les passages doivent être classés par score décroissant.")

    def _validate_non_retrieved_result(self) -> None:
        if self.matches:
            raise ValueError("Un résultat sans retrieval ne doit contenir aucun passage.")

        expected_status = SELECTION_TO_RETRIEVAL_STATUS.get(self.selection.status)

        if self.status is not expected_status:
            raise ValueError("Le statut du retrieval ne correspond pas à la sélection.")

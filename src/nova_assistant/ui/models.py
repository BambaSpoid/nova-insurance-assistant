from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from nova_assistant.decision import AssistantResponse
from nova_assistant.filtering import SelectionRequest


class ConversationEntry(BaseModel):
    """Échange complet conservé dans l’historique Streamlit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1)
    selection_request: SelectionRequest
    response: AssistantResponse

    @field_validator("question", mode="before")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        normalized_question = value.strip()

        if not normalized_question:
            raise ValueError("La question ne doit pas être vide.")

        return normalized_question

    @model_validator(mode="after")
    def validate_retrieval_context(self) -> Self:
        retrieval_request = self.response.decision.retrieval_result.request

        if retrieval_request.question != self.question:
            raise ValueError("La réponse ne correspond pas à la question.")

        if retrieval_request.selection_request != self.selection_request:
            raise ValueError("La réponse ne correspond pas au contexte.")

        return self

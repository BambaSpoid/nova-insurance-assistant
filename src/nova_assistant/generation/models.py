import re
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from nova_assistant.indexing import SemanticSearchResult
from nova_assistant.retrieval import (
    RetrievalResult,
    RetrievalStatus,
)


class EvidenceSource(BaseModel):
    """Source documentaire fournie au générateur."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    document_id: str
    title: str
    page_number: int = Field(ge=1)
    chunk_id: str
    pdf_path: Path
    score: float = Field(ge=-1.0, le=1.0)
    text: str = Field(min_length=1)

    @field_validator("text", mode="before")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        normalized_text = value.strip()

        if not normalized_text:
            raise ValueError("Le texte de la source ne doit pas être vide.")

        return normalized_text

    @classmethod
    def from_match(
        cls,
        match: SemanticSearchResult,
        source_number: int,
    ) -> Self:
        chunk = match.chunk

        return cls(
            source_id=f"S{source_number}",
            document_id=chunk.document_id,
            title=chunk.title,
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
            pdf_path=chunk.pdf_path,
            score=match.score,
            text=chunk.text,
        )


class GenerationRequest(BaseModel):
    """Retrieval validé destiné à la génération."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_result: RetrievalResult
    max_sources: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def retrieval_must_be_successful(self) -> Self:
        if self.retrieval_result.status is not RetrievalStatus.RETRIEVED:
            raise ValueError("La génération exige un retrieval réussi.")

        return self


class GenerationPrompt(BaseModel):
    """Prompt complet et sources numérotées."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    sources: tuple[EvidenceSource, ...] = Field(min_length=1)

    @field_validator("system_prompt", "user_prompt", mode="before")
    @classmethod
    def prompts_must_not_be_blank(cls, value: str) -> str:
        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("Un prompt ne doit pas être vide.")

        return normalized_value

    @model_validator(mode="after")
    def source_ids_must_be_unique(self) -> Self:
        source_ids = [source.source_id for source in self.sources]

        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Les identifiants de source doivent être uniques.")

        return self


class GeneratedAnswer(BaseModel):
    """Réponse générée accompagnée de citations vérifiables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    citations: tuple[EvidenceSource, ...] = Field(min_length=1)

    @field_validator("answer", "model_name", mode="before")
    @classmethod
    def values_must_not_be_blank(cls, value: str) -> str:
        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("La valeur ne doit pas être vide.")

        return normalized_value

    @model_validator(mode="after")
    def citations_must_match_answer(self) -> Self:
        cited_markers = set(re.findall(r"\[(S[1-9][0-9]*)\]", self.answer))
        citation_ids = {citation.source_id for citation in self.citations}

        if cited_markers != citation_ids:
            raise ValueError(
                "Les citations structurées doivent correspondre "
                "exactement aux marqueurs de la réponse."
            )

        return self

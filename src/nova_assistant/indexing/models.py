from hashlib import sha256
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from nova_assistant.domain import (
    DocumentStatus,
    DocumentType,
    ProductType,
)
from nova_assistant.ingestion import IngestedPage


class IndexedChunk(BaseModel):
    """Passage documentaire destiné à l’index sémantique."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(
        pattern=(
            r"^NOVA-[A-Z]+-[A-Z]+-[0-9]{4}"
            r":page:[1-9][0-9]*:chunk:[1-9][0-9]*$"
        )
    )
    page_id: str = Field(pattern=r"^NOVA-[A-Z]+-[A-Z]+-[0-9]{4}:page:[1-9][0-9]*$")
    document_id: str = Field(pattern=r"^NOVA-[A-Z]+-[A-Z]+-[0-9]{4}$")
    title: str
    product: ProductType
    document_type: DocumentType
    version: int
    language: Literal["fr"]
    status: DocumentStatus
    territories: tuple[str, ...]
    page_number: int = Field(ge=1)
    chunk_number: int = Field(ge=1)
    word_start: int = Field(ge=0)
    word_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    pdf_path: Path
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("text", mode="before")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        normalized_text = value.strip()

        if not normalized_text:
            raise ValueError("Le passage ne doit pas être vide.")

        return normalized_text

    @model_validator(mode="after")
    def validate_traceability(self) -> Self:
        expected_page_id = f"{self.document_id}:page:{self.page_number}"
        expected_chunk_id = f"{expected_page_id}:chunk:{self.chunk_number}"

        if self.page_id != expected_page_id:
            raise ValueError("Le page_id ne correspond pas au document et à la page.")

        if self.chunk_id != expected_chunk_id:
            raise ValueError("Le chunk_id ne correspond pas à la page et au passage.")

        if self.word_end <= self.word_start:
            raise ValueError("word_end doit être strictement supérieur à word_start.")

        expected_word_count = self.word_end - self.word_start
        actual_word_count = len(self.text.split())

        if actual_word_count != expected_word_count:
            raise ValueError("Le nombre de mots ne correspond pas aux positions.")

        return self

    @classmethod
    def from_page(
        cls,
        page: IngestedPage,
        chunk_number: int,
        text: str,
        word_start: int,
        word_end: int,
    ) -> Self:
        normalized_text = text.strip()

        return cls(
            chunk_id=(f"{page.page_id}:chunk:{chunk_number}"),
            page_id=page.page_id,
            document_id=page.document_id,
            title=page.title,
            product=page.product,
            document_type=page.document_type,
            version=page.version,
            language=page.language,
            status=page.status,
            territories=page.territories,
            page_number=page.page_number,
            chunk_number=chunk_number,
            word_start=word_start,
            word_end=word_end,
            text=normalized_text,
            pdf_path=page.pdf_path,
            content_sha256=sha256(normalized_text.encode("utf-8")).hexdigest(),
        )


class SemanticSearchResult(BaseModel):
    """Passage retourné par une recherche sémantique."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: IndexedChunk
    score: float = Field(ge=-1.0, le=1.0)


class IndexManifest(BaseModel):
    """Description vérifiable d’un index sémantique."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    model_name: str
    dimension: int = Field(gt=0)
    chunk_count: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    source_pages_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunks_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    embeddings_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

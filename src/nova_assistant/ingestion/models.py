from hashlib import sha256
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nova_assistant.domain import (
    DocumentMetadata,
    DocumentStatus,
    DocumentType,
    ProductType,
)


class IngestedPage(BaseModel):
    """Page extraite d’un document et enrichie avec ses métadonnées."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_id: str = Field(pattern=r"^NOVA-[A-Z]+-[A-Z]+-[0-9]{4}:page:[1-9][0-9]*$")
    document_id: str
    title: str
    product: ProductType
    document_type: DocumentType
    version: int
    language: str
    status: DocumentStatus
    territories: tuple[str, ...]
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    pdf_path: Path
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("text", mode="before")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        normalized_text = value.strip()

        if not normalized_text:
            raise ValueError("Le texte extrait ne doit pas être vide.")

        return normalized_text

    @field_validator("pdf_path")
    @classmethod
    def pdf_path_must_be_relative(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("Le chemin PDF doit être relatif au projet.")

        if value.suffix.lower() != ".pdf":
            raise ValueError("Le chemin doit désigner un fichier PDF.")

        return value

    @classmethod
    def from_document(
        cls,
        document: DocumentMetadata,
        page_number: int,
        text: str,
    ) -> Self:
        normalized_text = text.strip()

        return cls(
            page_id=f"{document.document_id}:page:{page_number}",
            document_id=document.document_id,
            title=document.title,
            product=document.product,
            document_type=document.document_type,
            version=document.version,
            language=document.language,
            status=document.status,
            territories=document.territories,
            page_number=page_number,
            text=normalized_text,
            pdf_path=document.pdf_path,
            content_sha256=sha256(normalized_text.encode("utf-8")).hexdigest(),
        )

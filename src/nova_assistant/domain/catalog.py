from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CATALOG_PATH = Path("data/catalog/document_catalog.json")


class ProductType(StrEnum):
    """Produits disponibles dans le corpus."""

    HOME = "home"
    AUTO = "auto"
    TRAVEL = "travel"


class DocumentType(StrEnum):
    """Types de documents reconnus."""

    IPID = "ipid"
    GENERAL_CONDITIONS = "conditions_generales"
    FAQ = "faq"


DOCUMENT_TYPE_FILENAME_LABELS: dict[DocumentType, str] = {
    DocumentType.IPID: "ipid",
    DocumentType.GENERAL_CONDITIONS: "conditions",
    DocumentType.FAQ: "faq",
}


class DocumentStatus(StrEnum):
    """Statuts documentaires autorisés."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class DocumentMetadata(BaseModel):
    """Métadonnées structurées d'un document."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    document_id: str = Field(
        min_length=8,
        pattern=r"^NOVA-[A-Z]+-[A-Z]+-[0-9]{4}$",
    )
    title: str = Field(min_length=5)
    product: ProductType
    document_type: DocumentType
    version: int = Field(ge=2024, le=2025)
    effective_from: date
    effective_to: date
    language: Literal["fr"]
    status: DocumentStatus
    territories: tuple[str, ...] = Field(min_length=1)
    source_path: Path
    pdf_path: Path
    searchable: bool = True

    @field_validator("source_path", "pdf_path")
    @classmethod
    def validate_relative_path(cls, value: Path) -> Path:
        """Interdit les chemins absolus et les remontées de répertoire."""

        if value.is_absolute():
            raise ValueError("Le chemin doit être relatif au projet.")

        if ".." in value.parts:
            raise ValueError("Le chemin ne peut pas remonter un répertoire.")

        return value

    @field_validator("source_path")
    @classmethod
    def validate_source_extension(cls, value: Path) -> Path:
        """Vérifie l'extension du fichier source."""

        if value.suffix != ".md":
            raise ValueError("Le fichier source doit être au format Markdown.")

        return value

    @field_validator("pdf_path")
    @classmethod
    def validate_pdf_extension(cls, value: Path) -> Path:
        """Vérifie l'extension du fichier généré."""

        if value.suffix != ".pdf":
            raise ValueError("Le document généré doit être au format PDF.")

        return value

    @model_validator(mode="after")
    def validate_business_consistency(self) -> "DocumentMetadata":
        """Vérifie les invariants métier d'une entrée."""

        if self.effective_to < self.effective_from:
            raise ValueError("La date de fin doit être postérieure ou égale à la date de début.")

        if self.effective_from.year != self.version:
            raise ValueError("L'année de début doit correspondre à la version.")

        if self.effective_to.year != self.version:
            raise ValueError("L'année de fin doit correspondre à la version.")

        expected_status = DocumentStatus.ARCHIVED if self.version == 2024 else DocumentStatus.ACTIVE

        if self.status != expected_status:
            raise ValueError(
                f"La version {self.version} doit avoir le statut {expected_status.value}."
            )

        filename_label = DOCUMENT_TYPE_FILENAME_LABELS[self.document_type]
        expected_source_name = f"nova_{self.product.value}_{filename_label}_{self.version}.md"

        if self.source_path.name != expected_source_name:
            raise ValueError("Le nom du fichier source ne correspond pas aux métadonnées.")

        if self.pdf_path.stem != self.source_path.stem:
            raise ValueError("Les fichiers Markdown et PDF doivent partager le même nom.")

        return self


class DocumentCatalog(BaseModel):
    """Catalogue complet des documents disponibles."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal["1.0"]
    documents: tuple[DocumentMetadata, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_uniqueness(self) -> "DocumentCatalog":
        """Vérifie l'unicité des identifiants et des chemins."""

        document_ids = [document.document_id for document in self.documents]
        source_paths = [document.source_path for document in self.documents]
        pdf_paths = [document.pdf_path for document in self.documents]

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Les identifiants documentaires doivent être uniques.")

        if len(source_paths) != len(set(source_paths)):
            raise ValueError("Les chemins des sources doivent être uniques.")

        if len(pdf_paths) != len(set(pdf_paths)):
            raise ValueError("Les chemins des PDF doivent être uniques.")

        return self

    def validate_files_exist(self, project_root: Path = Path(".")) -> None:
        """Vérifie que tous les fichiers déclarés existent."""

        missing_paths = []

        for document in self.documents:
            for relative_path in (
                document.source_path,
                document.pdf_path,
            ):
                absolute_path = project_root / relative_path

                if not absolute_path.is_file():
                    missing_paths.append(relative_path)

        if missing_paths:
            missing = ", ".join(str(path) for path in sorted(missing_paths))
            raise FileNotFoundError(f"Fichiers déclarés mais absents : {missing}")


def load_catalog(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
) -> DocumentCatalog:
    """Charge et valide le catalogue JSON."""

    content = catalog_path.read_text(encoding="utf-8")
    return DocumentCatalog.model_validate_json(content)

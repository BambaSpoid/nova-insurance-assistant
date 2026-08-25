import json
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from nova_assistant.domain import (
    DEFAULT_CATALOG_PATH,
    DocumentCatalog,
    load_catalog,
)
from nova_assistant.evaluation.models import EvaluationCase

DEFAULT_EVALUATION_DATASET_PATH = Path("data/evaluation/evaluation_cases.jsonl")


def validate_evaluation_cases(
    cases: tuple[EvaluationCase, ...],
    catalog: DocumentCatalog,
) -> None:
    if not cases:
        raise ValueError("Le jeu d’évaluation doit contenir au moins un cas.")

    case_ids = tuple(case.case_id for case in cases)

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Les identifiants des cas d’évaluation doivent être uniques.")

    documents_by_id = {document.document_id: document for document in catalog.documents}

    for case in cases:
        for document_id in case.expected_document_ids:
            document = documents_by_id.get(document_id)

            if document is None:
                raise ValueError(f"Document attendu inconnu pour {case.case_id} : {document_id}.")

            if document.product is not case.product:
                raise ValueError(
                    f"Le document {document_id} ne correspond pas au produit du cas {case.case_id}."
                )


def load_evaluation_cases(
    dataset_path: Path = DEFAULT_EVALUATION_DATASET_PATH,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
) -> tuple[EvaluationCase, ...]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Jeu d’évaluation introuvable : {dataset_path}")

    cases: list[EvaluationCase] = []

    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line_number, raw_line in enumerate(
            dataset_file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                raise ValueError(f"Ligne vide interdite à la ligne {line_number}.")

            try:
                raw_case = json.loads(line)
            except JSONDecodeError as error:
                raise ValueError(f"JSON invalide à la ligne {line_number}.") from error

            try:
                case = EvaluationCase.model_validate(raw_case)
            except ValidationError as error:
                raise ValueError(f"Cas invalide à la ligne {line_number}.") from error

            cases.append(case)

    loaded_cases = tuple(cases)
    catalog = load_catalog(catalog_path)

    validate_evaluation_cases(
        cases=loaded_cases,
        catalog=catalog,
    )

    return loaded_cases

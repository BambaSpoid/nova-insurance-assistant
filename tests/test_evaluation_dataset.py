import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from nova_assistant.decision import (
    AssistantStatus,
    normalize_text,
)
from nova_assistant.domain import ProductType
from nova_assistant.evaluation import (
    EvaluationCase,
    EvaluationCategory,
    load_evaluation_cases,
)
from nova_assistant.filtering import (
    SelectionStatus,
    select_corpus,
)
from nova_assistant.ingestion import load_ingested_pages


def valid_case_data() -> dict:
    return {
        "case_id": "auto_2025_collision_franchise",
        "category": "direct_answer",
        "question": "Quelle est la franchise collision ?",
        "product": "auto",
        "version": 2025,
        "expected_status": "answered",
        "expected_document_ids": [
            "NOVA-AUTO-IPID-2025",
            "NOVA-AUTO-CG-2025",
        ],
        "expected_evidence_terms": [
            "franchise collision",
            "350 €",
        ],
        "forbidden_answer_terms": ["500 €"],
        "requires_citations": True,
    }


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    content = "\n".join(
        json.dumps(
            row,
            ensure_ascii=False,
        )
        for row in rows
    )
    path.write_text(
        f"{content}\n",
        encoding="utf-8",
    )


def test_dataset_contains_expected_case_count() -> None:
    cases = load_evaluation_cases()

    assert len(cases) == 30
    assert len({case.case_id for case in cases}) == 30


def test_dataset_is_balanced_by_product() -> None:
    cases = load_evaluation_cases()

    counts = Counter(case.product for case in cases)

    assert counts == {
        ProductType.HOME: 10,
        ProductType.AUTO: 10,
        ProductType.TRAVEL: 10,
    }


def test_dataset_contains_expected_status_distribution() -> None:
    cases = load_evaluation_cases()

    counts = Counter(case.expected_status for case in cases)

    assert counts == {
        AssistantStatus.ANSWERED: 23,
        AssistantStatus.CLARIFICATION_REQUIRED: 3,
        AssistantStatus.CONFLICTING_CONTEXT: 1,
        AssistantStatus.NO_MATCHING_CORPUS: 1,
        AssistantStatus.INSUFFICIENT_EVIDENCE: 2,
    }


def test_answered_cases_define_evidence_and_citations() -> None:
    cases = load_evaluation_cases()

    answered_cases = tuple(
        case for case in cases if case.expected_status is AssistantStatus.ANSWERED
    )

    assert len(answered_cases) == 23
    assert all(case.expected_document_ids for case in answered_cases)
    assert all(case.expected_evidence_terms for case in answered_cases)
    assert all(case.requires_citations for case in answered_cases)


def test_non_answered_cases_do_not_require_citations() -> None:
    cases = load_evaluation_cases()

    non_answered_cases = tuple(
        case for case in cases if case.expected_status is not AssistantStatus.ANSWERED
    )

    assert len(non_answered_cases) == 7
    assert all(not case.requires_citations for case in non_answered_cases)


def test_dataset_context_selects_expected_corpus() -> None:
    cases = load_evaluation_cases()

    for case in cases:
        selection = select_corpus(case.to_selection_request())

        if case.expected_document_ids:
            assert selection.status is SelectionStatus.SELECTED
            assert selection.allowed_document_ids == case.expected_document_ids
        else:
            assert selection.status.value == case.expected_status.value


def test_model_rejects_answer_without_evidence() -> None:
    data = valid_case_data()
    data["expected_evidence_terms"] = []

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(data)


def test_model_rejects_refusal_requiring_citations() -> None:
    data = valid_case_data()
    data["expected_status"] = "insufficient_evidence"
    data["requires_citations"] = True

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(data)


def test_model_builds_selection_request() -> None:
    case = EvaluationCase(
        case_id="home_date_selection",
        category=EvaluationCategory.DIRECT_ANSWER,
        question="Quelle garantie est applicable ?",
        product=ProductType.HOME,
        contract_date="2025-06-15",
        expected_status=AssistantStatus.ANSWERED,
        expected_document_ids=(
            "NOVA-HOME-IPID-2025",
            "NOVA-HOME-CG-2025",
        ),
        expected_evidence_terms=("Nova Home 2025",),
        requires_citations=True,
    )

    request = case.to_selection_request()

    assert request.product is ProductType.HOME
    assert request.version is None
    assert request.contract_date.isoformat() == "2025-06-15"


def test_loader_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "invalid.jsonl"
    dataset_path.write_text(
        '{"case_id":\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="JSON invalide"):
        load_evaluation_cases(dataset_path=dataset_path)


def test_loader_rejects_blank_line(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "blank.jsonl"
    row = json.dumps(
        valid_case_data(),
        ensure_ascii=False,
    )
    dataset_path.write_text(
        f"{row}\n\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Ligne vide"):
        load_evaluation_cases(dataset_path=dataset_path)


def test_loader_rejects_duplicate_case_ids(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "duplicates.jsonl"
    row = valid_case_data()

    write_jsonl(
        dataset_path,
        [row, row],
    )

    with pytest.raises(ValueError, match="uniques"):
        load_evaluation_cases(dataset_path=dataset_path)


def test_loader_rejects_unknown_document(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "unknown_document.jsonl"
    row = valid_case_data()
    row["expected_document_ids"] = ["NOVA-UNKNOWN"]

    write_jsonl(
        dataset_path,
        [row],
    )

    with pytest.raises(ValueError, match="inconnu"):
        load_evaluation_cases(dataset_path=dataset_path)


def test_loader_rejects_document_from_another_product(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "wrong_product.jsonl"
    row = valid_case_data()
    row["expected_document_ids"] = [
        "NOVA-HOME-IPID-2025",
    ]

    write_jsonl(
        dataset_path,
        [row],
    )

    with pytest.raises(ValueError, match="produit"):
        load_evaluation_cases(dataset_path=dataset_path)


def test_expected_evidence_exists_in_authorized_corpus() -> None:
    pages_by_document = defaultdict(list)

    for page in load_ingested_pages():
        pages_by_document[page.document_id].append(page.text)

    for case in load_evaluation_cases():
        authorized_text = normalize_text(
            "\n".join(
                text
                for document_id in case.expected_document_ids
                for text in pages_by_document[document_id]
            )
        )

        for expected_term in case.expected_evidence_terms:
            assert normalize_text(expected_term) in authorized_text, (
                f"Preuve absente pour {case.case_id} : {expected_term}"
            )

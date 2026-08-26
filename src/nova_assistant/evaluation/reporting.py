import json
from collections import Counter, defaultdict
from pathlib import Path

from nova_assistant.evaluation.results import (
    EvaluationCaseResult,
    EvaluationGroupSummary,
    EvaluationRunSummary,
)

DEFAULT_EVALUATION_RESULTS_DIRECTORY = Path("data/evaluation/results")


def build_group_summary(
    results: tuple[EvaluationCaseResult, ...],
) -> EvaluationGroupSummary:
    total = len(results)
    passed = sum(result.checks.overall for result in results)

    return EvaluationGroupSummary(
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total if total else 0.0,
    )


def build_evaluation_summary(
    results: tuple[EvaluationCaseResult, ...],
) -> EvaluationRunSummary:
    if not results:
        raise ValueError("Impossible de résumer une évaluation vide.")

    modes = {result.mode for result in results}

    if len(modes) != 1:
        raise ValueError("Tous les résultats doivent utiliser le même mode.")

    results_by_product = defaultdict(list)
    results_by_category = defaultdict(list)

    for result in results:
        results_by_product[result.product.value].append(result)
        results_by_category[result.category.value].append(result)

    total = len(results)
    passed = sum(result.checks.overall for result in results)
    citation_checks = tuple(
        result.checks.citations for result in results if result.checks.citations is not None
    )
    forbidden_term_checks = tuple(
        result.checks.forbidden_terms
        for result in results
        if result.checks.forbidden_terms is not None
    )
    return EvaluationRunSummary(
        mode=next(iter(modes)),
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total,
        status_checks_passed=sum(result.checks.status for result in results),
        selection_checks_passed=sum(result.checks.selection for result in results),
        retrieval_scope_checks_passed=sum(result.checks.retrieval_scope for result in results),
        evidence_checks_passed=sum(result.checks.evidence for result in results),
        citation_checks_passed=(sum(citation_checks) if citation_checks else None),
        forbidden_terms_checks_passed=(
            sum(forbidden_term_checks) if forbidden_term_checks else None
        ),
        generated_answers=sum(result.model_name is not None for result in results),
        observed_status_counts=dict(Counter(result.observed_status for result in results)),
        by_product={
            product: build_group_summary(tuple(product_results))
            for product, product_results in sorted(results_by_product.items())
        },
        by_category={
            category: build_group_summary(tuple(category_results))
            for category, category_results in sorted(results_by_category.items())
        },
        average_retrieval_duration_ms=(
            sum(result.retrieval_duration_ms for result in results) / total
        ),
        average_total_duration_ms=(sum(result.total_duration_ms for result in results) / total),
    )


def save_evaluation_report(
    results: tuple[EvaluationCaseResult, ...],
    output_directory: Path = (DEFAULT_EVALUATION_RESULTS_DIRECTORY),
) -> tuple[Path, Path]:
    summary = build_evaluation_summary(results)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = output_directory / f"{summary.mode.value}_results.jsonl"
    summary_path = output_directory / f"{summary.mode.value}_summary.json"

    results_content = "\n".join(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        for result in results
    )

    results_path.write_text(
        f"{results_content}\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            summary.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return results_path, summary_path

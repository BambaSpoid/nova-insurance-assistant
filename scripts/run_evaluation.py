import argparse
import os
from pathlib import Path

from nova_assistant.evaluation import (
    DEFAULT_EVALUATION_RESULTS_DIRECTORY,
    EvaluationRunner,
    FullEvaluationRunner,
    build_evaluation_summary,
    load_evaluation_cases,
    save_evaluation_report,
)
from nova_assistant.generation import (
    DEFAULT_GENERATION_MODEL,
    OpenAIGenerator,
)
from nova_assistant.retrieval import load_default_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Exécute le jeu d’évaluation Nova en mode hors ligne ou complet.")
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "full"),
        default="offline",
        help=("Le mode full appelle OpenAI uniquement pour les réponses autorisées."),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre de passages récupérés par question.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=5,
        help="Nombre maximal de sources envoyées au générateur.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GENERATION_MODEL,
        help="Modèle OpenAI utilisé en mode full.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=180,
        help="Limite de sortie par réponse générée.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_EVALUATION_RESULTS_DIRECTORY,
        help="Dossier de sortie du rapport.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_evaluation_cases()
    retriever = load_default_retriever()

    if args.mode == "full":
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY est absente de l’environnement.")

        runner = FullEvaluationRunner(
            retriever=retriever,
            generator=OpenAIGenerator(
                model_name=args.model,
                max_output_tokens=(args.max_output_tokens),
            ),
            top_k=args.top_k,
            max_sources=args.max_sources,
        )
        results = runner.run_full(cases)
    else:
        runner = EvaluationRunner(
            retriever=retriever,
            top_k=args.top_k,
        )
        results = runner.run_offline(cases)

    summary = build_evaluation_summary(results)
    results_path, summary_path = save_evaluation_report(
        results=results,
        output_directory=args.output_directory,
    )

    print(f"Évaluation {summary.mode.value} terminée.")
    print(f"- Cas : {summary.total}")
    print(f"- Réussites : {summary.passed}")
    print(f"- Échecs : {summary.failed}")
    print(f"- Taux global : {summary.pass_rate:.1%}")
    print(f"- Sélection : {summary.selection_checks_passed}/{summary.total}")
    print(f"- Périmètre retrieval : {summary.retrieval_scope_checks_passed}/{summary.total}")
    print(f"- Preuves : {summary.evidence_checks_passed}/{summary.total}")
    print(f"- Statuts : {summary.status_checks_passed}/{summary.total}")
    print(f"- Réponses générées : {summary.generated_answers}")

    if summary.citation_checks_passed is not None:
        print(f"- Citations : {summary.citation_checks_passed}/{summary.total}")

    if summary.forbidden_terms_checks_passed is not None:
        print(
            f"- Termes interdits absents : {summary.forbidden_terms_checks_passed}/{summary.total}"
        )

    print(f"- Résultats : {results_path}")
    print(f"- Synthèse : {summary_path}")

    if summary.failed:
        print("\nCas en échec :")

        for result in results:
            if result.checks.overall:
                continue

            print(
                f"- {result.case_id} : "
                f"attendu={result.expected_status.value}, "
                f"observé={result.observed_status}, "
                f"erreur={result.error}"
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()

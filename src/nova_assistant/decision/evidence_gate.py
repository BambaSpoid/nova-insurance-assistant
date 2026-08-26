import re
import unicodedata

from nova_assistant.decision.models import (
    DecisionStatus,
    EvidenceDecision,
    EvidenceGateConfig,
    EvidenceSignals,
)
from nova_assistant.retrieval import (
    RetrievalResult,
    RetrievalStatus,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
CLAUSE_PATTERN = re.compile(r"[.;\n]+")
VALUE_PATTERN = re.compile(
    r"\b\d{1,3}(?:[ .]\d{3})*(?:,\d+)?\s*"
    r"(?:€|euros?|jours?|heures?|interventions?)"
    r"(?=\s|$|[.,;:])"
)

NUMBER_WORDS = {
    "zero": "0",
    "un": "1",
    "une": "1",
    "deux": "2",
    "trois": "3",
    "quatre": "4",
    "cinq": "5",
    "six": "6",
    "sept": "7",
    "huit": "8",
    "neuf": "9",
    "dix": "10",
    "onze": "11",
    "douze": "12",
    "treize": "13",
    "quatorze": "14",
    "quinze": "15",
    "seize": "16",
}

QUERY_TERM_EXPANSIONS = {
    "vol": ("transport",),
}

STOPWORDS = {
    "a",
    "ai",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "combien",
    "comment",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "est",
    "et",
    "il",
    "je",
    "la",
    "le",
    "les",
    "ma",
    "mes",
    "mon",
    "ne",
    "ou",
    "par",
    "pas",
    "pour",
    "que",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "qui",
    "sa",
    "se",
    "son",
    "sur",
    "un",
    "une",
    "assurance",
    "contrat",
    "applique",
    "appliquee",
    "apres",
}

ABSENCE_MARKERS = (
    "ne sont pas decrits",
    "n est pas decrit",
    "n est pas decrite",
    "absent du corpus",
    "absente du corpus",
    "ne doit pas inventer",
    "ne contient aucune regle relative",
    "ne fournit aucune information",
    "pas decrit dans le present corpus",
    "pas decrite dans le present corpus",
    "n est pas present",
    "n est pas presente",
)


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(
        r"[^a-z0-9€]+",
        " ",
        without_accents,
    ).strip()


def extract_query_terms(question: str) -> tuple[str, ...]:
    normalized_question = normalize_text(question)
    terms: list[str] = []

    for token in TOKEN_PATTERN.findall(normalized_question):
        if len(token) < 2:
            continue

        if token in STOPWORDS:
            continue

        if re.fullmatch(r"20[0-9]{2}", token):
            continue

        if token not in terms:
            terms.append(token)

    return tuple(terms)


def find_absence_markers(
    query_terms: tuple[str, ...],
    texts: tuple[str, ...],
) -> tuple[str, ...]:
    found_markers: list[str] = []
    query_term_set = set(query_terms)

    for text in texts:
        for raw_clause in CLAUSE_PATTERN.split(text):
            clause = normalize_text(raw_clause)
            clause_tokens = set(TOKEN_PATTERN.findall(clause))

            if query_term_set and not query_term_set.intersection(clause_tokens):
                continue

            for marker in ABSENCE_MARKERS:
                if marker in clause and marker not in found_markers:
                    found_markers.append(marker)

    return tuple(
        marker
        for marker in found_markers
        if not any(
            marker != other_marker and marker in other_marker for other_marker in found_markers
        )
    )


def value_unit(value: str) -> str:
    normalized_value = normalize_text(value)

    if "€" in value or "euro" in normalized_value:
        return "money"

    if "jour" in normalized_value:
        return "day"

    if "heure" in normalized_value:
        return "hour"

    if "intervention" in normalized_value:
        return "intervention"

    return "other"


def find_conflicting_values(
    query_terms: tuple[str, ...],
    texts: tuple[str, ...],
) -> tuple[str, ...]:
    query_term_set = set(query_terms)
    for query_term in query_terms:
        query_term_set.update(
            QUERY_TERM_EXPANSIONS.get(
                query_term,
                (),
            )
        )
    query_numbers = {term for term in query_terms if term.isdigit()}
    query_numbers.update(NUMBER_WORDS[term] for term in query_terms if term in NUMBER_WORDS)

    candidates: list[tuple[int, str]] = []

    for text in texts:
        for raw_clause in CLAUSE_PATTERN.split(text):
            clause = normalize_text(raw_clause)
            clause_tokens = set(TOKEN_PATTERN.findall(clause))
            overlap = len(query_term_set.intersection(clause_tokens))

            if overlap == 0:
                continue

            values = tuple(
                dict.fromkeys(
                    re.sub(r"\s+", " ", value).strip() for value in VALUE_PATTERN.findall(clause)
                )
            )

            # Une clause contenant plusieurs valeurs, comme un
            # tableau aplati, ne permet pas une association fiable.
            if len(values) != 1:
                continue

            value = values[0]
            number_match = re.match(
                r"\d{1,3}(?:[ .]\d{3})*",
                value,
            )
            value_number = (
                number_match.group(0).replace(" ", "").replace(".", "") if number_match else None
            )

            # La valeur décrivant le scénario de la question
            # n’est pas une valeur contractuelle contradictoire.
            if value_number in query_numbers:
                continue

            candidates.append((overlap, value))

    if not candidates:
        return ()

    best_overlap = max(overlap for overlap, _ in candidates)
    best_values = tuple(value for overlap, value in candidates if overlap == best_overlap)

    values_by_unit: dict[str, list[str]] = {}

    for value in best_values:
        unit = value_unit(value)
        unit_values = values_by_unit.setdefault(
            unit,
            [],
        )

        if value not in unit_values:
            unit_values.append(value)

    return tuple(value for values in values_by_unit.values() if len(values) > 1 for value in values)


def evaluate_evidence(
    retrieval_result: RetrievalResult,
    config: EvidenceGateConfig | None = None,
) -> EvidenceDecision:
    """Autorise ou refuse la génération à partir de signaux externes."""

    selected_config = config or EvidenceGateConfig()

    if retrieval_result.status is not RetrievalStatus.RETRIEVED:
        return _propagate_retrieval_status(retrieval_result)

    evaluated_matches = retrieval_result.matches[: selected_config.max_passages]
    question_terms = extract_query_terms(retrieval_result.request.question)
    combined_tokens: set[str] = set()
    texts: list[str] = []

    for match in evaluated_matches:
        normalized_text = normalize_text(match.chunk.text)
        combined_tokens.update(TOKEN_PATTERN.findall(normalized_text))
        texts.append(match.chunk.text)

    matched_query_terms = tuple(term for term in question_terms if term in combined_tokens)
    coverage = len(matched_query_terms) / len(question_terms) if question_terms else 0.0
    absence_markers = find_absence_markers(
        query_terms=question_terms,
        texts=tuple(texts),
    )
    conflicting_values = find_conflicting_values(
        query_terms=question_terms,
        texts=tuple(texts),
    )
    top_score = evaluated_matches[0].score if evaluated_matches else None

    signals = EvidenceSignals(
        top_score=top_score,
        query_terms=question_terms,
        matched_query_terms=matched_query_terms,
        query_term_coverage=coverage,
        absence_markers=absence_markers,
        conflicting_values=conflicting_values,
        evaluated_chunk_ids=tuple(match.chunk.chunk_id for match in evaluated_matches),
    )

    if top_score is None or top_score < selected_config.min_top_score:
        return EvidenceDecision(
            retrieval_result=retrieval_result,
            status=DecisionStatus.INSUFFICIENT_EVIDENCE,
            reason=("Le meilleur score sémantique est inférieur au seuil provisoire."),
            signals=signals,
        )

    if coverage < selected_config.min_query_term_coverage:
        return EvidenceDecision(
            retrieval_result=retrieval_result,
            status=DecisionStatus.INSUFFICIENT_EVIDENCE,
            reason=(
                "Les passages retrouvés couvrent insuffisamment les "
                "termes significatifs de la question."
            ),
            signals=signals,
        )

    if absence_markers:
        return EvidenceDecision(
            retrieval_result=retrieval_result,
            status=DecisionStatus.INSUFFICIENT_EVIDENCE,
            reason=(
                "Les sources indiquent explicitement que l’information "
                "demandée est absente du corpus."
            ),
            signals=signals,
        )

    if conflicting_values:
        return EvidenceDecision(
            retrieval_result=retrieval_result,
            status=DecisionStatus.CONFLICTING_SOURCES,
            reason=(
                "Plusieurs valeurs différentes sont associées aux termes "
                "de la question dans les sources applicables."
            ),
            signals=signals,
        )

    return EvidenceDecision(
        retrieval_result=retrieval_result,
        status=DecisionStatus.GENERATION_ALLOWED,
        reason=(
            "Les signaux externes dépassent les seuils provisoires et "
            "aucune absence ou contradiction n’a été détectée."
        ),
        signals=signals,
    )


def _propagate_retrieval_status(
    retrieval_result: RetrievalResult,
) -> EvidenceDecision:
    status_mapping = {
        RetrievalStatus.CLARIFICATION_REQUIRED: (DecisionStatus.CLARIFICATION_REQUIRED),
        RetrievalStatus.CONFLICTING_CONTEXT: (DecisionStatus.CONFLICTING_CONTEXT),
        RetrievalStatus.NO_MATCHING_CORPUS: (DecisionStatus.NO_MATCHING_CORPUS),
    }

    return EvidenceDecision(
        retrieval_result=retrieval_result,
        status=status_mapping[retrieval_result.status],
        reason=retrieval_result.selection.reason,
        signals=EvidenceSignals(),
    )

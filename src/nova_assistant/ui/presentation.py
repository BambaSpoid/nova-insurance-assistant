from nova_assistant.decision import AssistantStatus
from nova_assistant.domain import ProductType
from nova_assistant.generation import EvidenceSource

PRODUCT_LABELS: dict[ProductType, str] = {
    ProductType.HOME: "Habitation",
    ProductType.AUTO: "Automobile",
    ProductType.TRAVEL: "Voyage",
}

STATUS_LABELS: dict[AssistantStatus, str] = {
    AssistantStatus.ANSWERED: "Réponse documentée",
    AssistantStatus.CLARIFICATION_REQUIRED: ("Contexte à préciser"),
    AssistantStatus.CONFLICTING_CONTEXT: ("Contexte contradictoire"),
    AssistantStatus.NO_MATCHING_CORPUS: ("Corpus introuvable"),
    AssistantStatus.INSUFFICIENT_EVIDENCE: ("Information insuffisante"),
    AssistantStatus.CONFLICTING_SOURCES: ("Sources contradictoires"),
}

STATUS_TONES: dict[AssistantStatus, str] = {
    AssistantStatus.ANSWERED: "success",
    AssistantStatus.CLARIFICATION_REQUIRED: "warning",
    AssistantStatus.CONFLICTING_CONTEXT: "error",
    AssistantStatus.NO_MATCHING_CORPUS: "error",
    AssistantStatus.INSUFFICIENT_EVIDENCE: "warning",
    AssistantStatus.CONFLICTING_SOURCES: "error",
}

SUGGESTED_QUESTIONS: dict[ProductType, tuple[str, ...]] = {
    ProductType.HOME: (
        "Quel est le plafond pour mes objets de valeur ?",
        "Le vol sans effraction est-il couvert ?",
        "Quels dégâts des eaux sont assurés ?",
    ),
    ProductType.AUTO: (
        "Quelle est la franchise après une collision ?",
        "L’assistance intervient-elle devant mon domicile ?",
        "Combien de jours dure le véhicule de remplacement ?",
    ),
    ProductType.TRAVEL: (
        "Un voyage de 120 jours est-il couvert ?",
        "Après combien d’heures le retard est-il indemnisé ?",
        "Quels vaccins sont obligatoires ?",
    ),
}


def product_label(product: ProductType) -> str:
    return PRODUCT_LABELS[product]


def status_label(status: AssistantStatus) -> str:
    return STATUS_LABELS[status]


def status_tone(status: AssistantStatus) -> str:
    return STATUS_TONES[status]


def suggested_questions(
    product: ProductType,
) -> tuple[str, ...]:
    return SUGGESTED_QUESTIONS[product]


def citation_label(source: EvidenceSource) -> str:
    return f"[{source.source_id}] {source.title} — page {source.page_number}"


def citation_metadata(source: EvidenceSource) -> str:
    return f"Document : {source.document_id} · Pertinence : {source.score:.3f}"

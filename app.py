import logging
import os
from datetime import date
from pathlib import Path

import streamlit as st
from openai import OpenAI

from nova_assistant.config import get_settings
from nova_assistant.decision import AssistantResponse
from nova_assistant.domain import ProductType
from nova_assistant.filtering import SelectionRequest
from nova_assistant.generation import OpenAIGenerator
from nova_assistant.retrieval import load_default_retriever
from nova_assistant.ui import (
    ConversationEntry,
    MissingGenerationCredentialError,
    NovaAssistantService,
    citation_label,
    citation_metadata,
    product_label,
    status_label,
    status_tone,
    suggested_questions,
)

settings = get_settings()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=settings.app_title,
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def resolve_openai_api_key() -> str | None:
    environment_key = os.getenv("OPENAI_API_KEY")

    if environment_key:
        return environment_key

    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None

    if not secret_key:
        return None

    return str(secret_key)


@st.cache_resource(show_spinner="Chargement de l’index documentaire…")
def get_retriever():
    return load_default_retriever()


@st.cache_resource
def get_assistant_service() -> NovaAssistantService:
    api_key = resolve_openai_api_key()
    generator = None

    if api_key:
        generator = OpenAIGenerator(
            client=OpenAI(
                api_key=api_key,
                max_retries=1,
                timeout=30.0,
            )
        )

    return NovaAssistantService(
        retriever=get_retriever(),
        generator=generator,
        top_k=5,
        max_sources=5,
    )


def build_selection_request(
    product: ProductType,
    context_mode: str,
    version: int,
    contract_date: date,
) -> SelectionRequest:
    selected_version = None
    selected_date = None

    if context_mode in {"Version", "Version et date"}:
        selected_version = version

    if context_mode in {"Date du contrat", "Version et date"}:
        selected_date = contract_date

    return SelectionRequest(
        product=product,
        version=selected_version,
        contract_date=selected_date,
    )


def render_response(response: AssistantResponse) -> None:
    tone = status_tone(response.status)
    callout = getattr(st, tone)

    callout(status_label(response.status))
    st.markdown(response.answer)

    if response.model_name:
        st.caption(f"Modèle de génération : {response.model_name}")

    if not response.citations:
        return

    with st.expander(f"Consulter les sources ({len(response.citations)})"):
        for source in response.citations:
            st.markdown(f"#### {citation_label(source)}")
            st.caption(citation_metadata(source))
            st.write(source.text)

            pdf_path = Path(source.pdf_path)

            if pdf_path.is_file():
                st.download_button(
                    label="Télécharger le document",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=(f"pdf-{source.source_id}-{source.chunk_id}"),
                    icon=":material/download:",
                )

            st.divider()


def render_history(
    history: list[ConversationEntry],
) -> None:
    for entry in history:
        with st.chat_message("user"):
            st.markdown(entry.question)

        with st.chat_message(
            "assistant",
            avatar=":material/shield:",
        ):
            render_response(entry.response)


if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


with st.sidebar:
    st.header("Contexte du contrat")

    product = st.selectbox(
        "Produit d’assurance",
        options=list(ProductType),
        format_func=product_label,
    )

    context_mode = st.radio(
        "Mode de sélection",
        options=(
            "Version",
            "Date du contrat",
            "Version et date",
            "Sans contexte",
        ),
        help=("Le corpus est toujours sélectionné avant la recherche sémantique."),
    )

    version = st.selectbox(
        "Version documentaire",
        options=(2025, 2024),
        disabled=context_mode
        not in {
            "Version",
            "Version et date",
        },
    )

    contract_date = st.date_input(
        "Date de souscription",
        value=date(2025, 6, 15),
        disabled=context_mode
        not in {
            "Date du contrat",
            "Version et date",
        },
    )

    st.divider()

    if resolve_openai_api_key():
        st.success(
            "Génération OpenAI disponible",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            "Clé OpenAI absente. Les clarifications et refus restent disponibles.",
            icon=":material/key_off:",
        )

    if st.button(
        "Effacer la conversation",
        icon=":material/delete_sweep:",
    ):
        st.session_state.conversation = []
        st.session_state.pending_question = None
        st.rerun()

    st.caption("Les réponses utilisent exclusivement le corpus synthétique Nova autorisé.")


st.title(settings.app_title)
st.caption("Assistant RAG documentaire pour les assurances habitation, automobile et voyage.")

intro_left, intro_right = st.columns([2, 1])

with intro_left:
    st.markdown(
        "Sélectionnez le contexte contractuel, puis posez une "
        "question. Chaque réponse générée est accompagnée de "
        "sources vérifiables."
    )

with intro_right:
    st.info(
        f"Produit sélectionné : **{product_label(product)}**",
        icon=":material/policy:",
    )

st.subheader("Questions suggérées")

suggestion_columns = st.columns(3)

for column, question in zip(
    suggestion_columns,
    suggested_questions(product),
    strict=True,
):
    with column:
        if st.button(
            question,
            key=f"suggestion-{product.value}-{question}",
            icon=":material/chat:",
        ):
            st.session_state.pending_question = question

st.divider()

render_history(st.session_state.conversation)

typed_question = st.chat_input("Posez votre question sur votre contrat…")

question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    selection_request = build_selection_request(
        product=product,
        context_mode=context_mode,
        version=version,
        contract_date=contract_date,
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message(
        "assistant",
        avatar=":material/shield:",
    ):
        try:
            with st.spinner("Recherche des documents applicables…"):
                entry = get_assistant_service().ask(
                    question=question,
                    selection_request=selection_request,
                )

            st.session_state.conversation.append(entry)
            render_response(entry.response)

        except MissingGenerationCredentialError as error:
            st.error(
                str(error),
                icon=":material/key_off:",
            )

        except Exception:
            logger.exception("Échec du traitement d’une question depuis l’interface.")
            st.error(
                "La réponse n’a pas pu être produite. Vérifiez la configuration et réessayez.",
                icon=":material/error:",
            )

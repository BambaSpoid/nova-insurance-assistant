import streamlit as st

from nova_assistant.config import get_settings

settings = get_settings()

st.set_page_config(
    page_title=settings.app_title,
    page_icon="🛡️",
    layout="wide",
)

st.title(settings.app_title)
st.caption("Prototype RAG documentaire — données d'assurance synthétiques")

st.info(
    "L'environnement est prêt. Le corpus, la recherche et la génération "
    "seront ajoutés progressivement."
)

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Choisir avant de chercher")
    st.write(
        "Le système sélectionnera les documents applicables avant "
        "d'effectuer la recherche sémantique."
    )

with right_column:
    st.subheader("Savoir dire non")
    st.write(
        "Le système répondra uniquement lorsque les documents "
        "contiennent des preuves suffisantes."
    )

with st.sidebar:
    st.header("État du projet")
    st.success("Étape 0 — environnement opérationnel")
    st.write(f"Environnement : `{settings.environment}`")
    st.write(f"Dossier des données : `{settings.data_dir}`")
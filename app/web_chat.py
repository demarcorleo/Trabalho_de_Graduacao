import streamlit as st

from src.pipeline.pipeline_rag import PipelineRAG


st.set_page_config(
    page_title="Assistente RAG",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# Configuração
# ============================================================

st.title("🤖 Assistente de Documentos Técnicos")

st.caption(
    "Sistema baseado em RAG para consulta aos documentos técnicos."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ Sistema")

    st.markdown(
        """
        **Arquitetura**

        - LlamaIndex
        - Ollama
        - Qwen3.5 4B BF16
        - FAISS
        - BGE Reranker
        - Multilingual E5

        **Recuperação**

        - Busca vetorial
        - Reranking
        - Contexto documental
        """
    )

    st.divider()

    if st.button(
        "🗑️ Nova conversa",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# Pipeline
# ============================================================

@st.cache_resource
def carregar_pipeline():
    return PipelineRAG()


if "pipeline" not in st.session_state:
    with st.spinner("Carregando sistema RAG..."):
        st.session_state.pipeline = carregar_pipeline()

pipeline = st.session_state.pipeline


# ============================================================
# Histórico
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("fontes")
        ):
            with st.expander("📚 Fontes consultadas"):

                for i, fonte in enumerate(
                    message["fontes"],
                    start=1,
                ):

                    st.markdown(
                        f"**{i}. {fonte['source']}**"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        if fonte.get("score") is not None:
                            st.caption(
                                f"FAISS: {fonte['score']:.4f}"
                            )

                    with col2:
                        if fonte.get("reranker_score") is not None:
                            st.caption(
                                f"Reranker: "
                                f"{fonte['reranker_score']:.4f}"
                            )

                    if fonte.get("texto"):
                        with st.expander(
                            "Ver trecho recuperado",
                            expanded=False,
                        ):
                            st.markdown(fonte["texto"])

                    if i < len(message["fontes"]):
                        st.divider()


# ============================================================
# Entrada do usuário
# ============================================================

if prompt := st.chat_input(
    "Digite sua pergunta..."
):

    # --------------------------------------------------------
    # Pergunta
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # --------------------------------------------------------
    # Resposta
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Consultando documentos e gerando resposta..."
        ):

            resultado = pipeline.processar_query_detalhado(
                prompt
            )

        resposta = resultado["resposta"]
        fontes = resultado["fontes"]

        st.markdown(resposta)

        # ----------------------------------------------------
        # Fontes
        # ----------------------------------------------------

        if fontes:

            with st.expander(
                f"📚 Fontes consultadas ({len(fontes)})"
            ):

                for i, fonte in enumerate(
                    fontes,
                    start=1,
                ):

                    st.markdown(
                        f"**{i}. {fonte['source']}**"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        if fonte.get("score") is not None:
                            st.caption(
                                f"FAISS: {fonte['score']:.4f}"
                            )

                    with col2:
                        if fonte.get("reranker_score") is not None:
                            st.caption(
                                f"Reranker: "
                                f"{fonte['reranker_score']:.4f}"
                            )

                    if fonte.get("texto"):
                        with st.expander(
                            "Ver trecho recuperado",
                            expanded=False,
                        ):
                            st.markdown(fonte["texto"])

                    if i < len(fontes):
                        st.divider()

    # --------------------------------------------------------
    # Salva histórico
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": resposta,
            "fontes": fontes,
        }
    )
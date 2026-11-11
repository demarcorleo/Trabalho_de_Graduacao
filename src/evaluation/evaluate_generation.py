import json
import re
from pathlib import Path

import numpy as np

from config.settings import (
    LLM_MODEL_NAME, N_CTX, MAX_TOKENS, TEMPERATURE, TOP_P,
    EMBEDDING_MODEL_NAME, TOP_K, RERANKER_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, RERANKER_TOP_N, RERANKER_MIN_SCORE,
    PROMPT_TEMPLATE, PROMPT_VERSION,
)
from src.pipeline.pipeline_rag import PipelineRAG
from src.retrieval.generate_embedding import (
    gerar_embedding_query,
)


QUESTIONS_PATH = Path(
    "src/evaluation/questions.json"
)

OUTPUT_DIR = Path(
    "results/generation"
)


def carregar_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalizar_texto(texto):
    if texto is None:
        return ""

    texto = str(texto).lower()

    texto = re.sub(
        r"\s+",
        " ",
        texto,
    )

    return texto.strip()


def similaridade_coseno(
    vetor_a,
    vetor_b,
):
    vetor_a = np.asarray(
        vetor_a
    ).reshape(-1)

    vetor_b = np.asarray(
        vetor_b
    ).reshape(-1)

    norma_a = np.linalg.norm(
        vetor_a
    )

    norma_b = np.linalg.norm(
        vetor_b
    )

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return float(
        np.dot(
            vetor_a,
            vetor_b,
        )
        / (
            norma_a
            * norma_b
        )
    )


def calcular_overlap_lexical(
    resposta,
    gold,
):
    """
    Calcula uma medida simples de sobreposição
    entre as palavras da resposta gerada e do gold.

    Essa métrica é complementar à similaridade
    semântica e não é utilizada isoladamente
    para definir correção.
    """

    resposta_tokens = set(
        normalizar_texto(
            resposta
        ).split()
    )

    gold_tokens = set(
        normalizar_texto(
            gold
        ).split()
    )

    if not gold_tokens:
        return 0.0

    intersecao = (
        resposta_tokens
        & gold_tokens
    )

    return len(
        intersecao
    ) / len(
        gold_tokens
    )


def avaliar_generation():
    print("=" * 100)
    print(
        "EXPERIMENTO DE GENERATION"
    )
    print("=" * 100)

    questoes = carregar_json(
        QUESTIONS_PATH
    )

    print(
        f"Questões carregadas: "
        f"{len(questoes)}"
    )

    print(
        "\nCarregando pipeline RAG..."
    )

    pipeline = PipelineRAG()

    resultados = []

    # --------------------------------------------------------
    # AVALIAÇÃO
    # --------------------------------------------------------

    for numero, question in enumerate(
        questoes,
        start=1,
    ):
        question_id = question[
            "id"
        ]

        pergunta = question[
            "question"
        ]

        gold = question[
            "gold"
        ]

        print("\n")
        print("-" * 100)

        print(
            f"[{numero}/{len(questoes)}] "
            f"{question_id}"
        )

        print(
            f"Pergunta: {pergunta}"
        )

        print(
            f"Gold: {gold}"
        )

        # ----------------------------------------------------
        # EXECUÇÃO DO PIPELINE
        # ----------------------------------------------------

        resultado_pipeline = (
            pipeline.processar_query_detalhado(
                pergunta
            )
        )

        resposta = resultado_pipeline[
            "resposta"
        ]

        fontes = resultado_pipeline[
            "fontes"
        ]

        # ----------------------------------------------------
        # EMBEDDINGS DA RESPOSTA E DO GOLD
        # ----------------------------------------------------

        embedding_resposta = (
            gerar_embedding_query(
                resposta,
                pipeline.tokenizer,
                pipeline.embedding_model,
            )
        )

        embedding_gold = (
            gerar_embedding_query(
                gold,
                pipeline.tokenizer,
                pipeline.embedding_model,
            )
        )

        similaridade = (
            similaridade_coseno(
                embedding_resposta,
                embedding_gold,
            )
        )

        overlap = (
            calcular_overlap_lexical(
                resposta,
                gold,
            )
        )

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        resultado = {
            "id": question_id,

            "question": pergunta,

            "gold": gold,

            "gold_source":
                question.get(
                    "gold_source"
                ),

            "source":
                question.get(
                    "source"
                ),

            "response":
                resposta,

            "semantic_similarity":
                similaridade,

            "lexical_overlap":
                overlap,

            "retrieved_sources":
                fontes,
        }

        resultados.append(
            resultado
        )

        print(
            "\nResposta gerada:"
        )

        print(
            resposta
        )

        print(
            "\nSimilaridade semântica: "
            f"{similaridade:.4f}"
        )

        print(
            "Overlap lexical: "
            f"{overlap:.4f}"
        )

    # --------------------------------------------------------
    # MÉTRICAS AGREGADAS
    # --------------------------------------------------------

    similaridades = [
        resultado[
            "semantic_similarity"
        ]
        for resultado in resultados
    ]

    overlaps = [
        resultado[
            "lexical_overlap"
        ]
        for resultado in resultados
    ]

    media_similaridade = (
        np.mean(
            similaridades
        )
        if similaridades
        else 0.0
    )

    media_overlap = (
        np.mean(
            overlaps
        )
        if overlaps
        else 0.0
    )

    # --------------------------------------------------------
    # RESULTADO FINAL
    # --------------------------------------------------------

    resultado_final = {
        "metadata": {
            "questions":
                len(questoes),

            "llm":
                LLM_MODEL_NAME,

            "context_window":
                N_CTX,

            "max_tokens":
                MAX_TOKENS,

            "temperature":
                TEMPERATURE,

            "top_p":
                TOP_P,

            "embedding_for_evaluation":
                EMBEDDING_MODEL_NAME,

            "retrieval_top_k":
                TOP_K,

            "reranker":
                RERANKER_MODEL_NAME,

            "chunk_size":
                CHUNK_SIZE,

            "chunk_overlap":
                CHUNK_OVERLAP,

            "reranker_top_n": RERANKER_TOP_N,
            "reranker_min_score": RERANKER_MIN_SCORE,
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_version": PROMPT_VERSION,
        },

        "metrics": {
            "mean_semantic_similarity":
                float(
                    media_similaridade
                ),

            "mean_lexical_overlap":
                float(
                    media_overlap
                ),
        },

        "questions":
            resultados,
    }

    # --------------------------------------------------------
    # SALVAMENTO
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / "generation_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            resultado_final,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # RESUMO
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)
    print(
        "COMPARAÇÃO FINAL — GENERATION"
    )
    print("=" * 100)

    print(
        f"Questões avaliadas: "
        f"{len(questoes)}"
    )

    print(
        f"Similaridade semântica média: "
        f"{media_similaridade:.4f}"
    )

    print(
        f"Overlap lexical médio: "
        f"{media_overlap:.4f}"
    )

    print(
        "\nResultado salvo em: "
        f"{output_file}"
    )

    print("=" * 100)


if __name__ == "__main__":
    avaliar_generation()
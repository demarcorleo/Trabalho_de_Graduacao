import json
from pathlib import Path

from config.settings import TOP_K

from src.retrieval.vector_store import (
    carregar_indice,
    busca,
)

from src.retrieval.reranker import (
    carregar_reranker,
    rerankear,
)

from src.retrieval.generate_embedding import (
    carregar_modelo_embedding,
    gerar_embedding_query,
)


QUESTIONS_PATH = Path(
    "src/evaluation/questions.json"
)

VARIANTS_PATH = Path(
    "src/evaluation/query_variants.json"
)

OUTPUT_PATH = Path(
    "results/query_processing"
)


def carregar_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# Critério unificado e sem correspondência parcial.
from src.evaluation.gold_matching import (
    GOLD_MATCH_VERSION,
    normalizar_source,
    normalizar_texto,
    localizar_gold_chunks,
)


def localizar_gold_passage(question, chunks):
    return localizar_gold_chunks(
        chunks,
        question["source"]["doc"],
        question.get("gold_source"),
    )


def encontrar_posicao_documento(
    resultados,
    documento,
):
    documento = normalizar_source(
        documento
    )

    for posicao, resultado in enumerate(
        resultados,
        start=1,
    ):
        source = normalizar_source(
            resultado.get("source")
        )

        if source == documento:
            return posicao

    return None


def encontrar_posicao_passage(
    resultados,
    gold_indices,
):
    if not gold_indices:
        return None

    gold_indices = set(
        gold_indices
    )

    for posicao, resultado in enumerate(
        resultados,
        start=1,
    ):
        indice = resultado.get(
            "indice"
        )

        if indice in gold_indices:
            return posicao

    return None


def recall_documento(
    resultados,
    documento,
    k,
):
    return (
        1.0
        if encontrar_posicao_documento(
            resultados[:k],
            documento,
        )
        is not None
        else 0.0
    )


def recall_passage(
    resultados,
    gold_indices,
    k,
):
    return (
        1.0
        if encontrar_posicao_passage(
            resultados[:k],
            gold_indices,
        )
        is not None
        else 0.0
    )


def reciprocal_rank_documento(
    resultados,
    documento,
):
    posicao = (
        encontrar_posicao_documento(
            resultados,
            documento,
        )
    )

    if posicao is None:
        return 0.0

    return 1.0 / posicao


def reciprocal_rank_passage(
    resultados,
    gold_indices,
):
    posicao = (
        encontrar_posicao_passage(
            resultados,
            gold_indices,
        )
    )

    if posicao is None:
        return 0.0

    return 1.0 / posicao


def calcular_media(
    resultados,
    chave,
):
    if not resultados:
        return 0.0

    valores = [
        resultado[chave]
        for resultado in resultados
    ]

    return sum(valores) / len(valores)


def avaliar_variante(
    nome_variante,
    questoes,
    variantes,
    index,
    chunks,
    tokenizer,
    embedding_model,
    reranker_model,
):
    print("\n")
    print("=" * 100)
    print(
        f"QUERY PROCESSING — "
        f"{nome_variante}"
    )
    print("=" * 100)

    resultados_avaliacao = []

    for numero, question in enumerate(
        questoes,
        start=1,
    ):
        question_id = question[
            "id"
        ]

        source = question[
            "source"
        ]["doc"]

        # ----------------------------------------------------
        # LOCALIZA GOLD PASSAGE NO C4
        # ----------------------------------------------------

        gold_indices = (
            localizar_gold_passage(
                question,
                chunks,
            )
        )

        # ----------------------------------------------------
        # CONSULTA
        # ----------------------------------------------------

        if nome_variante == "portugues":

            query = question[
                "question"
            ]

        else:

            query = variantes[
                question_id
            ]["english"]

        print(
            f"\n[{numero}/{len(questoes)}] "
            f"{question_id}"
        )

        print(
            f"Consulta: {query}"
        )

        print(
            "Gold passage(s): "
            + (
                str(gold_indices)
                if gold_indices
                else "NÃO ENCONTRADO"
            )
        )

        # ----------------------------------------------------
        # EMBEDDING DA QUERY
        # ----------------------------------------------------

        query_vector = (
            gerar_embedding_query(
                query,
                tokenizer,
                embedding_model,
            )
        )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        resultados_faiss = busca(
            query_vector=query_vector,
            index=index,
            chunks=chunks,
            k=TOP_K,
        )

        document_position = (
            encontrar_posicao_documento(
                resultados_faiss,
                source,
            )
        )

        passage_position = (
            encontrar_posicao_passage(
                resultados_faiss,
                gold_indices,
            )
        )

        # ----------------------------------------------------
        # RERANKER
        # ----------------------------------------------------

        resultados_rerankeados = (
            rerankear(
                resultados=resultados_faiss,
                query=query,
                reranker_model=reranker_model,
            )
        )

        reranker_document_position = (
            encontrar_posicao_documento(
                resultados_rerankeados,
                source,
            )
        )

        reranker_passage_position = (
            encontrar_posicao_passage(
                resultados_rerankeados,
                gold_indices,
            )
        )

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        resultado = {
            "id": question_id,
            "query": query,
            "source": source,
            "gold_source":
                question.get(
                    "gold_source"
                ),
            "gold_chunk_indices":
                gold_indices,

            "faiss_document_position":
                document_position,

            "faiss_passage_position":
                passage_position,

            "faiss_document_recall_at_1":
                recall_documento(
                    resultados_faiss,
                    source,
                    1,
                ),

            "faiss_document_recall_at_3":
                recall_documento(
                    resultados_faiss,
                    source,
                    3,
                ),

            "faiss_document_recall_at_5":
                recall_documento(
                    resultados_faiss,
                    source,
                    5,
                ),

            "faiss_document_recall_at_10":
                recall_documento(
                    resultados_faiss,
                    source,
                    10,
                ),

            "faiss_document_mrr":
                reciprocal_rank_documento(
                    resultados_faiss,
                    source,
                ),

            "faiss_passage_recall_at_1":
                recall_passage(
                    resultados_faiss,
                    gold_indices,
                    1,
                ),

            "faiss_passage_recall_at_3":
                recall_passage(
                    resultados_faiss,
                    gold_indices,
                    3,
                ),

            "faiss_passage_recall_at_5":
                recall_passage(
                    resultados_faiss,
                    gold_indices,
                    5,
                ),

            "faiss_passage_recall_at_10":
                recall_passage(
                    resultados_faiss,
                    gold_indices,
                    10,
                ),

            "faiss_passage_mrr":
                reciprocal_rank_passage(
                    resultados_faiss,
                    gold_indices,
                ),

            "reranker_document_position":
                reranker_document_position,

            "reranker_passage_position":
                reranker_passage_position,

            "reranker_document_recall_at_1":
                recall_documento(
                    resultados_rerankeados,
                    source,
                    1,
                ),

            "reranker_document_recall_at_3":
                recall_documento(
                    resultados_rerankeados,
                    source,
                    3,
                ),

            "reranker_document_mrr":
                reciprocal_rank_documento(
                    resultados_rerankeados,
                    source,
                ),

            "reranker_passage_recall_at_1":
                recall_passage(
                    resultados_rerankeados,
                    gold_indices,
                    1,
                ),

            "reranker_passage_recall_at_3":
                recall_passage(
                    resultados_rerankeados,
                    gold_indices,
                    3,
                ),

            "reranker_passage_mrr":
                reciprocal_rank_passage(
                    resultados_rerankeados,
                    gold_indices,
                ),
        }

        resultados_avaliacao.append(
            resultado
        )

        print(
            "  FAISS:"
        )

        print(
            "    Documento: "
            + (
                f"posição {document_position}"
                if document_position
                else "MISS"
            )
        )

        print(
            "    Passage: "
            + (
                f"posição {passage_position}"
                if passage_position
                else "MISS"
            )
        )

        print(
            "  Reranker:"
        )

        print(
            "    Documento: "
            + (
                f"posição "
                f"{reranker_document_position}"
                if reranker_document_position
                else "MISS"
            )
        )

        print(
            "    Passage: "
            + (
                f"posição "
                f"{reranker_passage_position}"
                if reranker_passage_position
                else "MISS"
            )
        )

    # --------------------------------------------------------
    # MÉTRICAS AGREGADAS
    # --------------------------------------------------------

    metrics = {
        "faiss_document_recall_at_1":
            calcular_media(
                resultados_avaliacao,
                "faiss_document_recall_at_1",
            ),

        "faiss_document_recall_at_3":
            calcular_media(
                resultados_avaliacao,
                "faiss_document_recall_at_3",
            ),

        "faiss_document_recall_at_5":
            calcular_media(
                resultados_avaliacao,
                "faiss_document_recall_at_5",
            ),

        "faiss_document_recall_at_10":
            calcular_media(
                resultados_avaliacao,
                "faiss_document_recall_at_10",
            ),

        "faiss_document_mrr":
            calcular_media(
                resultados_avaliacao,
                "faiss_document_mrr",
            ),

        "faiss_passage_recall_at_1":
            calcular_media(
                resultados_avaliacao,
                "faiss_passage_recall_at_1",
            ),

        "faiss_passage_recall_at_3":
            calcular_media(
                resultados_avaliacao,
                "faiss_passage_recall_at_3",
            ),

        "faiss_passage_recall_at_5":
            calcular_media(
                resultados_avaliacao,
                "faiss_passage_recall_at_5",
            ),

        "faiss_passage_recall_at_10":
            calcular_media(
                resultados_avaliacao,
                "faiss_passage_recall_at_10",
            ),

        "faiss_passage_mrr":
            calcular_media(
                resultados_avaliacao,
                "faiss_passage_mrr",
            ),

        "reranker_document_recall_at_1":
            calcular_media(
                resultados_avaliacao,
                "reranker_document_recall_at_1",
            ),

        "reranker_document_recall_at_3":
            calcular_media(
                resultados_avaliacao,
                "reranker_document_recall_at_3",
            ),

        "reranker_document_mrr":
            calcular_media(
                resultados_avaliacao,
                "reranker_document_mrr",
            ),

        "reranker_passage_recall_at_1":
            calcular_media(
                resultados_avaliacao,
                "reranker_passage_recall_at_1",
            ),

        "reranker_passage_recall_at_3":
            calcular_media(
                resultados_avaliacao,
                "reranker_passage_recall_at_3",
            ),

        "reranker_passage_mrr":
            calcular_media(
                resultados_avaliacao,
                "reranker_passage_mrr",
            ),
    }

    return {
        "variant": nome_variante,
        "metrics": metrics,
        "questions": resultados_avaliacao,
    }


def main():

    print("=" * 100)
    print(
        "EXPERIMENTO DE QUERY PROCESSING"
    )

    print(
        "CHUNKING: C4 (1500 / 300)"
    )

    print(
        "EMBEDDING: "
        "intfloat/multilingual-e5-base"
    )

    print(
        f"TOP_K: {TOP_K}"
    )

    print(
        "RERANKER: "
        "BAAI/bge-reranker-v2-m3"
    )

    print("=" * 100)

    # --------------------------------------------------------
    # QUESTÕES
    # --------------------------------------------------------

    questoes = carregar_json(
        QUESTIONS_PATH
    )

    variantes_lista = carregar_json(
        VARIANTS_PATH
    )

    variantes = {
        item["id"]: item
        for item in variantes_lista
    }

    print(
        f"\nQuestões carregadas: "
        f"{len(questoes)}"
    )

    print(
        f"Variantes carregadas: "
        f"{len(variantes)}"
    )

    # --------------------------------------------------------
    # ÍNDICE
    # --------------------------------------------------------

    print(
        "\nCarregando índice FAISS..."
    )

    index, chunks = carregar_indice()

    print(
        f"Chunks no índice: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    print(
        "\nCarregando modelo de embedding..."
    )

    tokenizer, embedding_model = (
        carregar_modelo_embedding()
    )

    # --------------------------------------------------------
    # RERANKER
    # --------------------------------------------------------

    print(
        "\nCarregando reranker..."
    )

    reranker_model = (
        carregar_reranker()
    )

    # --------------------------------------------------------
    # AVALIAÇÕES
    # --------------------------------------------------------

    resultados_portugues = (
        avaliar_variante(
            nome_variante="portugues",
            questoes=questoes,
            variantes=variantes,
            index=index,
            chunks=chunks,
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
    )

    resultados_ingles = (
        avaliar_variante(
            nome_variante="ingles",
            questoes=questoes,
            variantes=variantes,
            index=index,
            chunks=chunks,
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
    )

    # --------------------------------------------------------
    # COMPARAÇÃO
    # --------------------------------------------------------

    comparacao = {
        "portugues":
            resultados_portugues[
                "metrics"
            ],

        "ingles":
            resultados_ingles[
                "metrics"
            ],
    }

    resultado_final = {
        "metadata": {
            "questions":
                len(questoes),

            "top_k":
                TOP_K,

            "chunk_size":
                1500,

            "chunk_overlap":
                300,

            "embedding":
                "intfloat/multilingual-e5-base",

            "reranker":
                "BAAI/bge-reranker-v2-m3",

            "index":
                "FAISS IndexFlatIP",

            "gold_passage_method":
                GOLD_MATCH_VERSION,
        },

        "comparison":
            comparacao,

        "portugues":
            resultados_portugues,

        "ingles":
            resultados_ingles,
    }

    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_PATH
        / "comparison_checkpoint03.json"
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
    print("=" * 120)
    print(
        "COMPARAÇÃO FINAL — QUERY PROCESSING"
    )
    print("=" * 120)

    print(
        f"{'Métrica':<40}"
        f"{'Português':<15}"
        f"{'Inglês':<15}"
    )

    print("-" * 120)

    metricas_para_exibir = [
        (
            "FAISS Document Recall@1",
            "faiss_document_recall_at_1",
        ),
        (
            "FAISS Document Recall@10",
            "faiss_document_recall_at_10",
        ),
        (
            "FAISS Document MRR",
            "faiss_document_mrr",
        ),
        (
            "FAISS Passage Recall@1",
            "faiss_passage_recall_at_1",
        ),
        (
            "FAISS Passage Recall@10",
            "faiss_passage_recall_at_10",
        ),
        (
            "FAISS Passage MRR",
            "faiss_passage_mrr",
        ),
        (
            "Reranker Document Recall@1",
            "reranker_document_recall_at_1",
        ),
        (
            "Reranker Passage Recall@1",
            "reranker_passage_recall_at_1",
        ),
        (
            "Reranker Passage MRR",
            "reranker_passage_mrr",
        ),
    ]

    for nome, chave in (
        metricas_para_exibir
    ):

        valor_pt = comparacao[
            "portugues"
        ][chave]

        valor_en = comparacao[
            "ingles"
        ][chave]

        if "mrr" in chave.lower():

            print(
                f"{nome:<40}"
                f"{valor_pt:<15.4f}"
                f"{valor_en:<15.4f}"
            )

        else:

            print(
                f"{nome:<40}"
                f"{valor_pt:<15.2%}"
                f"{valor_en:<15.2%}"
            )

    print("\n")

    print(
        f"Resultado salvo em: "
        f"{output_file}"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
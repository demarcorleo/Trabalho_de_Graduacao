import json
import re
from pathlib import Path

from config.settings import DOCUMENTS_PATH, TOP_K

from src.ingestion.ingestion import carregar_documentos
from src.retrieval.generate_embedding import (
    carregar_modelo_embedding,
    gerar_embeddings,
    gerar_embedding_query,
)
from src.retrieval.vector_store import criar_indice, busca
from src.retrieval.reranker import carregar_reranker, rerankear


QUESTIONS_PATH = Path("src/evaluation/questions.json")
OUTPUT_PATH = Path("results/chunking")


CONFIGURACOES = [
    {
        "id": "C1",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    {
        "id": "C2",
        "chunk_size": 750,
        "chunk_overlap": 100,
    },
    {
        "id": "C3",
        "chunk_size": 1000,
        "chunk_overlap": 150,
    },
    {
        "id": "C4",
        "chunk_size": 1500,
        "chunk_overlap": 300,
    },
    {
        "id": "C5",
        "chunk_size": 2000,
        "chunk_overlap": 400,
    },
]


def carregar_questoes():
    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# Critério unificado e sem correspondência parcial.
from src.evaluation.gold_matching import (
    normalizar_source,
    normalizar_texto,
    gold_esta_no_chunk,
    localizar_gold_chunks,
)


def encontrar_gold_chunks(chunks, source, gold_source):
    return localizar_gold_chunks(chunks, source, gold_source)


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
    gold_chunk_indices,
):
    """
    Procura a posição do primeiro chunk-gold
    entre os resultados recuperados.
    """
    gold_indices = set(
        gold_chunk_indices
    )

    for posicao, resultado in enumerate(
        resultados,
        start=1,
    ):
        indice = resultado.get("indice")

        if indice in gold_indices:
            return posicao

    return None


def recall_at_k_documento(
    resultados,
    documento,
    k,
):
    return float(
        encontrar_posicao_documento(
            resultados[:k],
            documento,
        )
        is not None
    )


def recall_at_k_passage(
    resultados,
    gold_chunk_indices,
    k,
):
    return float(
        encontrar_posicao_passage(
            resultados[:k],
            gold_chunk_indices,
        )
        is not None
    )


def reciprocal_rank_documento(
    resultados,
    documento,
):
    posicao = encontrar_posicao_documento(
        resultados,
        documento,
    )

    if posicao is None:
        return 0.0

    return 1.0 / posicao


def reciprocal_rank_passage(
    resultados,
    gold_chunk_indices,
):
    posicao = encontrar_posicao_passage(
        resultados,
        gold_chunk_indices,
    )

    if posicao is None:
        return 0.0

    return 1.0 / posicao


def calcular_metricas(
    resultados,
    prefixo,
    tipo,
):
    """
    Calcula métricas agregadas.

    prefixo:
        faiss
        reranker

    tipo:
        document
        passage
    """
    total = len(resultados)

    if total == 0:
        return {}

    if prefixo == "faiss":
        ks = [1, 3, 5, 10]
    else:
        ks = [1, 3]

    metricas = {}

    for k in ks:
        chave = (
            f"{tipo}_recall_at_{k}_{prefixo}"
        )

        valores = [
            item[chave]
            for item in resultados
        ]

        metricas[
            f"recall_at_{k}"
        ] = sum(valores) / total

    chave_rr = f"{tipo}_rr_{prefixo}"

    valores_rr = [
        item[chave_rr]
        for item in resultados
    ]

    metricas["mrr"] = (
        sum(valores_rr) / total
    )

    return metricas


def gerar_chunks_configuracao(
    documentos,
    chunk_size,
    chunk_overlap,
):
    """
    Mantém a mesma lógica do baseline,
    alterando somente tamanho e sobreposição.
    """
    from llama_index.core.schema import TextNode
    from langchain_text_splitters import (
        MarkdownTextSplitter,
    )

    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for documento in documentos:
        texto = documento.get_content()
        metadata = dict(
            documento.metadata
        )

        documento_temp = type(
            "DocumentoTemporario",
            (),
            {
                "page_content": texto,
                "metadata": metadata,
            },
        )()

        documentos_divididos = (
            splitter.split_documents(
                [documento_temp]
            )
        )

        for chunk in documentos_divididos:
            node = TextNode(
                text=chunk.page_content,
                metadata=dict(
                    chunk.metadata
                ),
            )

            chunks.append(node)

    return chunks


def avaliar_configuracao(
    configuracao,
    documentos,
    questions,
    tokenizer,
    embedding_model,
    reranker_model,
):
    config_id = configuracao["id"]

    chunk_size = configuracao[
        "chunk_size"
    ]

    chunk_overlap = configuracao[
        "chunk_overlap"
    ]

    print("\n")
    print("=" * 80)
    print(
        f"{config_id} | "
        f"chunk_size={chunk_size} | "
        f"chunk_overlap={chunk_overlap}"
    )
    print("=" * 80)

    # --------------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------------

    chunks = gerar_chunks_configuracao(
        documentos=documentos,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print(
        f"Chunks gerados: {len(chunks)}"
    )

    # --------------------------------------------------------------
    # LOCALIZAÇÃO DOS PASSAGES-GOLD
    # --------------------------------------------------------------

    gold_info = {}

    passages_encontrados = 0

    for question in questions:
        question_id = question["id"]

        source = question[
            "source"
        ]["doc"]

        gold_source = question[
            "gold_source"
        ]

        gold_chunks = (
            encontrar_gold_chunks(
                chunks=chunks,
                source=source,
                gold_source=gold_source,
            )
        )

        gold_info[
            question_id
        ] = gold_chunks

        if gold_chunks:
            passages_encontrados += 1

    print(
        "Passages-gold encontrados: "
        f"{passages_encontrados}/"
        f"{len(questions)}"
    )

    # --------------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------------

    embeddings = gerar_embeddings(
        chunks,
        tokenizer,
        embedding_model,
    )

    # --------------------------------------------------------------
    # FAISS
    # --------------------------------------------------------------

    index = criar_indice(
        embeddings
    )

    resultados_questoes = []

    # --------------------------------------------------------------
    # PERGUNTAS
    # --------------------------------------------------------------

    for numero, question in enumerate(
        questions,
        start=1,
    ):
        question_id = question["id"]

        query = question[
            "question"
        ]

        gold = question["gold"]

        gold_source = question[
            "gold_source"
        ]

        source = question[
            "source"
        ]["doc"]

        gold_chunk_indices = (
            gold_info[
                question_id
            ]
        )

        print(
            f"\n[{numero}/{len(questions)}] "
            f"{question_id}: {query}"
        )

        if gold_chunk_indices:
            print(
                "  Gold passage(s): "
                f"{gold_chunk_indices}"
            )
        else:
            print(
                "  Gold passage: "
                "NÃO ENCONTRADO"
            )

        query_vector = (
            gerar_embedding_query(
                query,
                tokenizer,
                embedding_model,
            )
        )

        # ----------------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------------

        resultados_faiss = busca(
            query_vector=query_vector,
            index=index,
            chunks=chunks,
            k=TOP_K,
        )

        document_position_faiss = (
            encontrar_posicao_documento(
                resultados_faiss,
                source,
            )
        )

        passage_position_faiss = (
            encontrar_posicao_passage(
                resultados_faiss,
                gold_chunk_indices,
            )
        )

        # ----------------------------------------------------------
        # RERANKING
        # ----------------------------------------------------------

        resultados_rerankeados = (
            rerankear(
                resultados=resultados_faiss,
                query=query,
                reranker_model=reranker_model,
            )
        )

        document_position_reranker = (
            encontrar_posicao_documento(
                resultados_rerankeados,
                source,
            )
        )

        passage_position_reranker = (
            encontrar_posicao_passage(
                resultados_rerankeados,
                gold_chunk_indices,
            )
        )

        print(
            "  FAISS:"
        )

        print(
            "    Documento: "
            + (
                f"posição "
                f"{document_position_faiss}"
                if document_position_faiss
                else "MISS"
            )
        )

        print(
            "    Passage: "
            + (
                f"posição "
                f"{passage_position_faiss}"
                if passage_position_faiss
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
                f"{document_position_reranker}"
                if document_position_reranker
                else "MISS"
            )
        )

        print(
            "    Passage: "
            + (
                f"posição "
                f"{passage_position_reranker}"
                if passage_position_reranker
                else "MISS"
            )
        )

        resultado = {
            "id": question_id,
            "question": query,
            "gold": gold,
            "gold_source": gold_source,
            "source": source,

            "gold_chunk_indices": (
                gold_chunk_indices
            ),

            # --------------------------------------------------
            # FAISS — DOCUMENTO
            # --------------------------------------------------

            "document_position_faiss":
                document_position_faiss,

            "document_recall_at_1_faiss":
                recall_at_k_documento(
                    resultados_faiss,
                    source,
                    1,
                ),

            "document_recall_at_3_faiss":
                recall_at_k_documento(
                    resultados_faiss,
                    source,
                    3,
                ),

            "document_recall_at_5_faiss":
                recall_at_k_documento(
                    resultados_faiss,
                    source,
                    5,
                ),

            "document_recall_at_10_faiss":
                recall_at_k_documento(
                    resultados_faiss,
                    source,
                    10,
                ),

            "document_rr_faiss":
                reciprocal_rank_documento(
                    resultados_faiss,
                    source,
                ),

            # --------------------------------------------------
            # FAISS — PASSAGE
            # --------------------------------------------------

            "passage_position_faiss":
                passage_position_faiss,

            "passage_recall_at_1_faiss":
                recall_at_k_passage(
                    resultados_faiss,
                    gold_chunk_indices,
                    1,
                ),

            "passage_recall_at_3_faiss":
                recall_at_k_passage(
                    resultados_faiss,
                    gold_chunk_indices,
                    3,
                ),

            "passage_recall_at_5_faiss":
                recall_at_k_passage(
                    resultados_faiss,
                    gold_chunk_indices,
                    5,
                ),

            "passage_recall_at_10_faiss":
                recall_at_k_passage(
                    resultados_faiss,
                    gold_chunk_indices,
                    10,
                ),

            "passage_rr_faiss":
                reciprocal_rank_passage(
                    resultados_faiss,
                    gold_chunk_indices,
                ),

            # --------------------------------------------------
            # RERANKER — DOCUMENTO
            # --------------------------------------------------

            "document_position_reranker":
                document_position_reranker,

            "document_recall_at_1_reranker":
                recall_at_k_documento(
                    resultados_rerankeados,
                    source,
                    1,
                ),

            "document_recall_at_3_reranker":
                recall_at_k_documento(
                    resultados_rerankeados,
                    source,
                    3,
                ),

            "document_rr_reranker":
                reciprocal_rank_documento(
                    resultados_rerankeados,
                    source,
                ),

            # --------------------------------------------------
            # RERANKER — PASSAGE
            # --------------------------------------------------

            "passage_position_reranker":
                passage_position_reranker,

            "passage_recall_at_1_reranker":
                recall_at_k_passage(
                    resultados_rerankeados,
                    gold_chunk_indices,
                    1,
                ),

            "passage_recall_at_3_reranker":
                recall_at_k_passage(
                    resultados_rerankeados,
                    gold_chunk_indices,
                    3,
                ),

            "passage_rr_reranker":
                reciprocal_rank_passage(
                    resultados_rerankeados,
                    gold_chunk_indices,
                ),
        }

        resultados_questoes.append(
            resultado
        )

    # --------------------------------------------------------------
    # MÉTRICAS AGREGADAS
    # --------------------------------------------------------------

    metricas = {
        "faiss_document": calcular_metricas(
            resultados_questoes,
            "faiss",
            "document",
        ),

        "faiss_passage": calcular_metricas(
            resultados_questoes,
            "faiss",
            "passage",
        ),

        "reranker_document": calcular_metricas(
            resultados_questoes,
            "reranker",
            "document",
        ),

        "reranker_passage": calcular_metricas(
            resultados_questoes,
            "reranker",
            "passage",
        ),
    }

    resultado_final = {
        "configuration": {
            "id": config_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": len(chunks),
        },

        "metrics": metricas,

        "questions": resultados_questoes,
    }

    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_PATH
        / f"{config_id.lower()}.json"
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

    print(
        f"\nResultado salvo em: "
        f"{output_file}"
    )

    return resultado_final


def main():
    print("=" * 80)
    print(
        "EXPERIMENTO DE CHUNKING"
    )
    print(
        "DOCUMENT RECALL + "
        "PASSAGE RECALL"
    )
    print("=" * 80)

    print(
        "\nCarregando documentos..."
    )

    documentos = carregar_documentos(
        DOCUMENTS_PATH
    )

    print(
        f"Documentos carregados: "
        f"{len(documentos)}"
    )

    print(
        "\nCarregando questões..."
    )

    questions = carregar_questoes()

    print(
        f"Questões carregadas: "
        f"{len(questions)}"
    )

    print(
        "\nCarregando modelo de "
        "embedding..."
    )

    tokenizer, embedding_model = (
        carregar_modelo_embedding()
    )

    print(
        "\nCarregando reranker..."
    )

    reranker_model = (
        carregar_reranker()
    )

    resultados_experimentos = []

    for configuracao in CONFIGURACOES:
        resultado = (
            avaliar_configuracao(
                configuracao=configuracao,
                documentos=documentos,
                questions=questions,
                tokenizer=tokenizer,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
            )
        )

        resultados_experimentos.append(
            resultado
        )

    # --------------------------------------------------------------
    # COMPARAÇÃO FINAL
    # --------------------------------------------------------------

    comparacao = []

    for resultado in resultados_experimentos:
        config = resultado[
            "configuration"
        ]

        metrics = resultado[
            "metrics"
        ]

        faiss_document = metrics[
            "faiss_document"
        ]

        faiss_passage = metrics[
            "faiss_passage"
        ]

        reranker_document = metrics[
            "reranker_document"
        ]

        reranker_passage = metrics[
            "reranker_passage"
        ]

        comparacao.append(
            {
                "configuration":
                    config["id"],

                "chunk_size":
                    config["chunk_size"],

                "chunk_overlap":
                    config["chunk_overlap"],

                "total_chunks":
                    config["total_chunks"],

                # FAISS — documento
                "faiss_document_recall_at_1":
                    faiss_document[
                        "recall_at_1"
                    ],

                "faiss_document_recall_at_3":
                    faiss_document[
                        "recall_at_3"
                    ],

                "faiss_document_recall_at_5":
                    faiss_document[
                        "recall_at_5"
                    ],

                "faiss_document_recall_at_10":
                    faiss_document[
                        "recall_at_10"
                    ],

                "faiss_document_mrr":
                    faiss_document[
                        "mrr"
                    ],

                # FAISS — passage
                "faiss_passage_recall_at_1":
                    faiss_passage[
                        "recall_at_1"
                    ],

                "faiss_passage_recall_at_3":
                    faiss_passage[
                        "recall_at_3"
                    ],

                "faiss_passage_recall_at_5":
                    faiss_passage[
                        "recall_at_5"
                    ],

                "faiss_passage_recall_at_10":
                    faiss_passage[
                        "recall_at_10"
                    ],

                "faiss_passage_mrr":
                    faiss_passage[
                        "mrr"
                    ],

                # Reranker — documento
                "reranker_document_recall_at_1":
                    reranker_document[
                        "recall_at_1"
                    ],

                "reranker_document_recall_at_3":
                    reranker_document[
                        "recall_at_3"
                    ],

                "reranker_document_mrr":
                    reranker_document[
                        "mrr"
                    ],

                # Reranker — passage
                "reranker_passage_recall_at_1":
                    reranker_passage[
                        "recall_at_1"
                    ],

                "reranker_passage_recall_at_3":
                    reranker_passage[
                        "recall_at_3"
                    ],

                "reranker_passage_mrr":
                    reranker_passage[
                        "mrr"
                    ],
            }
        )

    comparison_file = (
        OUTPUT_PATH
        / "comparison.json"
    )

    with open(
        comparison_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparacao,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------------
    # TABELA FINAL
    # --------------------------------------------------------------

    print("\n")
    print("=" * 140)
    print(
        "COMPARAÇÃO FINAL — CHUNKING"
    )
    print("=" * 140)

    print(
        f"{'Cfg':<6}"
        f"{'Size':<7}"
        f"{'Chunks':<8}"
        f"{'Doc R@1':<10}"
        f"{'Doc R@10':<11}"
        f"{'Doc MRR':<10}"
        f"{'Pass R@1':<11}"
        f"{'Pass R@10':<12}"
        f"{'Pass MRR':<10}"
        f"{'Rer Doc@1':<11}"
        f"{'Rer Pass@1':<12}"
    )

    print("-" * 140)

    for item in comparacao:
        print(
            f"{item['configuration']:<6}"
            f"{item['chunk_size']:<7}"
            f"{item['total_chunks']:<8}"
            f"{item['faiss_document_recall_at_1']:<10.2%}"
            f"{item['faiss_document_recall_at_10']:<11.2%}"
            f"{item['faiss_document_mrr']:<10.4f}"
            f"{item['faiss_passage_recall_at_1']:<11.2%}"
            f"{item['faiss_passage_recall_at_10']:<12.2%}"
            f"{item['faiss_passage_mrr']:<10.4f}"
            f"{item['reranker_document_recall_at_1']:<11.2%}"
            f"{item['reranker_passage_recall_at_1']:<12.2%}"
        )

    print("\n")
    print(
        f"Comparação salva em: "
        f"{comparison_file}"
    )

    print("=" * 140)


if __name__ == "__main__":
    main()
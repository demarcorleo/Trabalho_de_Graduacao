import json
from pathlib import Path

from config.settings import (
    DOCUMENTS_PATH,
    TOP_K,
)

from src.ingestion.ingestion import carregar_documentos
from src.evaluation.evaluate_chunking import (
    carregar_questoes,
    gerar_chunks_configuracao,
    recall_at_k_documento,
    recall_at_k_passage,
    reciprocal_rank_documento,
    reciprocal_rank_passage,
    encontrar_posicao_documento,
    encontrar_posicao_passage,
    encontrar_gold_chunks,
)

from src.retrieval.generate_embedding import (
    gerar_embeddings,
    gerar_embedding_query,
)

from src.retrieval.vector_store import (
    criar_indice,
    busca,
)

from src.retrieval.reranker import (
    carregar_reranker,
    rerankear,
)


OUTPUT_PATH = Path(
    "results/embedding"
)


CONFIGURACOES = [
    {
        "id": "E1",
        "model": "intfloat/multilingual-e5-base",
    },
    {
        "id": "E2",
        "model": "BAAI/bge-m3",
    },
    {
        "id": "E3",
        "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    },
]


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


def carregar_modelo_embedding_modelo(
    model_name
):
    from transformers import (
        AutoTokenizer,
        AutoModel,
    )

    print(
        f"\nCarregando embedding: "
        f"{model_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModel.from_pretrained(
        model_name
    )

    model.eval()

    return tokenizer, model


def gerar_embedding_query_modelo(
    query,
    tokenizer,
    model,
):
    import torch
    import torch.nn.functional as F

    from src.retrieval.generate_embedding import (
        mean_pooling,
    )

    encoded_input = tokenizer(
        [query],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        model_output = model(
            **encoded_input
        )

    embedding = mean_pooling(
        model_output,
        encoded_input[
            "attention_mask"
        ],
    )

    embedding = F.normalize(
        embedding,
        p=2,
        dim=1,
    )

    return (
        embedding
        .cpu()
        .numpy()
        .astype("float32")
    )


def calcular_metricas(
    resultados,
    prefixo,
    tipo,
):
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

    chave_rr = (
        f"{tipo}_rr_{prefixo}"
    )

    valores_rr = [
        item[chave_rr]
        for item in resultados
    ]

    metricas["mrr"] = (
        sum(valores_rr) / total
    )

    return metricas


def avaliar_configuracao(
    configuracao,
    documentos,
    questions,
    reranker_model,
):
    config_id = configuracao["id"]
    model_name = configuracao["model"]

    print("\n")
    print("=" * 90)
    print(
        f"{config_id} | "
        f"{model_name}"
    )
    print("=" * 90)

    # ------------------------------------------------------------
    # CHUNKING FIXO — C4
    # ------------------------------------------------------------

    print(
        "\nGerando chunks C4 "
        "(1500 / 300)..."
    )

    chunks = gerar_chunks_configuracao(
        documentos=documentos,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(
        f"Chunks gerados: "
        f"{len(chunks)}"
    )

    # ------------------------------------------------------------
    # GOLD PASSAGES
    # ------------------------------------------------------------

    gold_info = {}

    for question in questions:

        question_id = question[
            "id"
        ]

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

    # ------------------------------------------------------------
    # MODELO
    # ------------------------------------------------------------

    tokenizer, embedding_model = (
        carregar_modelo_embedding_modelo(
            model_name
        )
    )

    # ------------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------------

    print(
        "\nGerando embeddings..."
    )

    embeddings = gerar_embeddings(
        chunks,
        tokenizer,
        embedding_model,
    )

    print(
        f"Dimensão dos embeddings: "
        f"{embeddings.shape}"
    )

    # ------------------------------------------------------------
    # FAISS
    # ------------------------------------------------------------

    index = criar_indice(
        embeddings
    )

    resultados_questoes = []

    # ------------------------------------------------------------
    # PERGUNTAS
    # ------------------------------------------------------------

    for numero, question in enumerate(
        questions,
        start=1,
    ):
        question_id = question[
            "id"
        ]

        query = question[
            "question"
        ]

        gold = question[
            "gold"
        ]

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

        # --------------------------------------------------------
        # QUERY EMBEDDING
        # --------------------------------------------------------

        query_vector = (
            gerar_embedding_query_modelo(
                query,
                tokenizer,
                embedding_model,
            )
        )

        # --------------------------------------------------------
        # RETRIEVAL
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # RERANKER
        # --------------------------------------------------------

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

            "gold_chunk_indices":
                gold_chunk_indices,

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

    # ------------------------------------------------------------
    # MÉTRICAS
    # ------------------------------------------------------------

    metricas = {
        "faiss_document":
            calcular_metricas(
                resultados_questoes,
                "faiss",
                "document",
            ),

        "faiss_passage":
            calcular_metricas(
                resultados_questoes,
                "faiss",
                "passage",
            ),

        "reranker_document":
            calcular_metricas(
                resultados_questoes,
                "reranker",
                "document",
            ),

        "reranker_passage":
            calcular_metricas(
                resultados_questoes,
                "reranker",
                "passage",
            ),
    }

    resultado_final = {
        "configuration": {
            "id": config_id,
            "model": model_name,
            "chunk_size":
                CHUNK_SIZE,
            "chunk_overlap":
                CHUNK_OVERLAP,
            "total_chunks":
                len(chunks),
            "embedding_dimension":
                int(
                    embeddings.shape[1]
                ),
        },

        "metrics": metricas,

        "questions":
            resultados_questoes,
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

    print("=" * 90)
    print(
        "EXPERIMENTO DE EMBEDDING"
    )
    print(
        "CHUNKING FIXO: C4 "
        "(1500 / 300)"
    )
    print("=" * 90)

    # ------------------------------------------------------------
    # DOCUMENTOS
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # QUESTÕES
    # ------------------------------------------------------------

    print(
        "\nCarregando questões..."
    )

    questions = carregar_questoes()

    print(
        f"Questões carregadas: "
        f"{len(questions)}"
    )

    # ------------------------------------------------------------
    # RERANKER
    # ------------------------------------------------------------

    print(
        "\nCarregando reranker..."
    )

    reranker_model = (
        carregar_reranker()
    )

    resultados_experimentos = []

    # ------------------------------------------------------------
    # EXPERIMENTOS
    # ------------------------------------------------------------

    for configuracao in CONFIGURACOES:

        resultado = (
            avaliar_configuracao(
                configuracao=configuracao,
                documentos=documentos,
                questions=questions,
                reranker_model=reranker_model,
            )
        )

        resultados_experimentos.append(
            resultado
        )

    # ------------------------------------------------------------
    # COMPARAÇÃO
    # ------------------------------------------------------------

    comparacao = []

    for resultado in (
        resultados_experimentos
    ):

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

                "model":
                    config["model"],

                "chunk_size":
                    config["chunk_size"],

                "chunk_overlap":
                    config["chunk_overlap"],

                "total_chunks":
                    config["total_chunks"],

                "embedding_dimension":
                    config[
                        "embedding_dimension"
                    ],

                "faiss_document_recall_at_1":
                    faiss_document[
                        "recall_at_1"
                    ],

                "faiss_document_recall_at_10":
                    faiss_document[
                        "recall_at_10"
                    ],

                "faiss_document_mrr":
                    faiss_document[
                        "mrr"
                    ],

                "faiss_passage_recall_at_1":
                    faiss_passage[
                        "recall_at_1"
                    ],

                "faiss_passage_recall_at_10":
                    faiss_passage[
                        "recall_at_10"
                    ],

                "faiss_passage_mrr":
                    faiss_passage[
                        "mrr"
                    ],

                "reranker_document_recall_at_1":
                    reranker_document[
                        "recall_at_1"
                    ],

                "reranker_passage_recall_at_1":
                    reranker_passage[
                        "recall_at_1"
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

    # ------------------------------------------------------------
    # TABELA
    # ------------------------------------------------------------

    print("\n")
    print("=" * 150)
    print(
        "COMPARAÇÃO FINAL — EMBEDDING"
    )
    print("=" * 150)

    print(
        f"{'Cfg':<6}"
        f"{'Modelo':<55}"
        f"{'Dim':<7}"
        f"{'Doc R@1':<10}"
        f"{'Doc R@10':<11}"
        f"{'Doc MRR':<10}"
        f"{'Pass R@1':<11}"
        f"{'Pass R@10':<12}"
        f"{'Pass MRR':<10}"
        f"{'Rer Pass@1':<12}"
    )

    print("-" * 150)

    for item in comparacao:

        print(
            f"{item['configuration']:<6}"
            f"{item['model']:<55}"
            f"{item['embedding_dimension']:<7}"
            f"{item['faiss_document_recall_at_1']:<10.2%}"
            f"{item['faiss_document_recall_at_10']:<11.2%}"
            f"{item['faiss_document_mrr']:<10.4f}"
            f"{item['faiss_passage_recall_at_1']:<11.2%}"
            f"{item['faiss_passage_recall_at_10']:<12.2%}"
            f"{item['faiss_passage_mrr']:<10.4f}"
            f"{item['reranker_passage_recall_at_1']:<12.2%}"
        )

    print("\n")
    print(
        f"Comparação salva em: "
        f"{comparison_file}"
    )

    print("=" * 150)


if __name__ == "__main__":
    main()
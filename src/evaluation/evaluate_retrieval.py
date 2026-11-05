import argparse
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


DEFAULT_QUESTIONS = Path("src/evaluation/questions.json")
DEFAULT_OUTPUT = Path("results/retrieval_baseline.json")


def carregar_questoes(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalizar_source(source):
    """
    Normaliza o nome da fonte para facilitar a comparação.
    """
    if source is None:
        return ""

    return Path(str(source)).name


def encontrar_posicao(resultados, documento):
    """
    Retorna a posição 1-based do documento correto.
    Retorna None caso o documento não esteja presente.
    """
    documento = normalizar_source(documento)

    for posicao, resultado in enumerate(resultados, start=1):
        source = normalizar_source(resultado.get("source"))

        if source == documento:
            return posicao

    return None


def recall_at_k(resultados, documento, k):
    """
    Retorna 1 se o documento correto estiver entre os
    primeiros k resultados.
    """
    posicao = encontrar_posicao(resultados[:k], documento)

    return 1.0 if posicao is not None else 0.0


def reciprocal_rank(resultados, documento):
    """
    Calcula Reciprocal Rank.

    RR = 1 / posição da primeira ocorrência correta.
    Caso não encontre, retorna 0.
    """
    posicao = encontrar_posicao(resultados, documento)

    if posicao is None:
        return 0.0

    return 1.0 / posicao


def calcular_metricas(resultados_por_questao, prefixo):
    """
    Calcula métricas agregadas.
    """
    total = len(resultados_por_questao)

    if total == 0:
        return {}

    if prefixo == "faiss":
        ks = [1, 3, 5, 10]
    else:
        ks = [1, 3]

    metricas = {}

    for k in ks:
        valores = [
            item[f"{prefixo}_recall_at_{k}"]
            for item in resultados_por_questao
        ]

        metricas[f"recall_at_{k}"] = sum(valores) / total

    valores_rr = [
        item[f"{prefixo}_rr"]
        for item in resultados_por_questao
    ]

    metricas["mrr"] = sum(valores_rr) / total

    return metricas


def obter_texto_chunk(chunk):
    """
    Compatibilidade com diferentes tipos de Document.
    """
    if hasattr(chunk, "page_content"):
        return chunk.page_content

    if hasattr(chunk, "get_content"):
        return chunk.get_content()

    return str(chunk)


def avaliar(questions_path, output_path):
    print("=" * 80)
    print("AVALIAÇÃO DO BASELINE DE RETRIEVAL")
    print("=" * 80)

    print(f"\nQuestões: {questions_path}")

    questions = carregar_questoes(questions_path)

    print(f"Total de questões: {len(questions)}")

    print("\nCarregando índice...")
    index, chunks = carregar_indice()

    print(f"Total de chunks no índice: {len(chunks)}")

    print("\nCarregando modelo de embedding...")
    tokenizer, embedding_model = carregar_modelo_embedding()

    print("\nCarregando reranker...")
    reranker_model = carregar_reranker()

    resultados_avaliacao = []

    for numero, question in enumerate(questions, start=1):
        question_id = question["id"]
        query = question["question"]
        gold = question["gold"]
        source = question["source"]["doc"]

        print("\n" + "-" * 80)
        print(f"[{numero}/{len(questions)}] {question_id}")
        print(f"Pergunta: {query}")
        print(f"Fonte esperada: {source}")

        # ------------------------------------------------------------------
        # EMBEDDING DA QUERY
        # ------------------------------------------------------------------

        query_vector = gerar_embedding_query(
            query,
            tokenizer,
            embedding_model,
        )

        # ------------------------------------------------------------------
        # RETRIEVAL / FAISS
        # ------------------------------------------------------------------

        resultados_faiss = busca(
            query_vector=query_vector,
            index=index,
            chunks=chunks,
            k=TOP_K,
        )

        posicao_faiss = encontrar_posicao(
            resultados_faiss,
            source,
        )

        print("\nFAISS:")

        if posicao_faiss is None:
            print("  Fonte correta: MISS")
        else:
            print(f"  Fonte correta: posição {posicao_faiss}")

        for posicao, resultado in enumerate(
            resultados_faiss,
            start=1,
        ):
            marcador = " <-- GOLD" if (
                normalizar_source(resultado.get("source"))
                == normalizar_source(source)
            ) else ""

            print(
                f"  {posicao:02d}. "
                f"score={resultado['score']:.6f} | "
                f"{resultado['source']}"
                f"{marcador}"
            )

        # ------------------------------------------------------------------
        # RERANKING
        # ------------------------------------------------------------------

        resultados_rerankeados = rerankear(
            resultados=resultados_faiss,
            query=query,
            reranker_model=reranker_model,
        )

        posicao_reranker = encontrar_posicao(
            resultados_rerankeados,
            source,
        )

        print("\nRERANKER:")

        if posicao_reranker is None:
            print("  Fonte correta: MISS")
        else:
            print(
                f"  Fonte correta: posição {posicao_reranker}"
            )

        for posicao, resultado in enumerate(
            resultados_rerankeados,
            start=1,
        ):
            marcador = " <-- GOLD" if (
                normalizar_source(resultado.get("source"))
                == normalizar_source(source)
            ) else ""

            print(
                f"  {posicao:02d}. "
                f"reranker={resultado['reranker_score']:.6f} | "
                f"FAISS={resultado['score']:.6f} | "
                f"{resultado['source']}"
                f"{marcador}"
            )

        # ------------------------------------------------------------------
        # MÉTRICAS DA QUESTÃO
        # ------------------------------------------------------------------

        resultado = {
            "id": question_id,
            "question": query,
            "gold": gold,
            "source": source,

            "faiss_position": posicao_faiss,

            "faiss_recall_at_1": recall_at_k(
                resultados_faiss,
                source,
                1,
            ),

            "faiss_recall_at_3": recall_at_k(
                resultados_faiss,
                source,
                3,
            ),

            "faiss_recall_at_5": recall_at_k(
                resultados_faiss,
                source,
                5,
            ),

            "faiss_recall_at_10": recall_at_k(
                resultados_faiss,
                source,
                10,
            ),

            "faiss_rr": reciprocal_rank(
                resultados_faiss,
                source,
            ),

            "reranker_position": posicao_reranker,

            "reranker_recall_at_1": recall_at_k(
                resultados_rerankeados,
                source,
                1,
            ),

            "reranker_recall_at_3": recall_at_k(
                resultados_rerankeados,
                source,
                3,
            ),

            "reranker_rr": reciprocal_rank(
                resultados_rerankeados,
                source,
            ),

            "faiss_results": [
                {
                    "rank": posicao,
                    "source": resultado.get("source"),
                    "score": resultado.get("score"),
                    "indice": resultado.get("indice"),
                }
                for posicao, resultado in enumerate(
                    resultados_faiss,
                    start=1,
                )
            ],

            "reranker_results": [
                {
                    "rank": posicao,
                    "source": resultado.get("source"),
                    "score": resultado.get("score"),
                    "reranker_score": resultado.get(
                        "reranker_score"
                    ),
                    "indice": resultado.get("indice"),
                }
                for posicao, resultado in enumerate(
                    resultados_rerankeados,
                    start=1,
                )
            ],
        }

        resultados_avaliacao.append(resultado)

    # ----------------------------------------------------------------------
    # MÉTRICAS AGREGADAS
    # ----------------------------------------------------------------------

    metricas_faiss = calcular_metricas(
        resultados_avaliacao,
        "faiss",
    )

    metricas_reranker = calcular_metricas(
        resultados_avaliacao,
        "reranker",
    )

    resultado_final = {
        "metadata": {
            "questions": len(questions),
            "top_k": TOP_K,
            "retrieval": "FAISS IndexFlatIP",
            "reranker": True,
        },

        "metrics": {
            "faiss": metricas_faiss,
            "reranker": metricas_reranker,
        },

        "questions": resultados_avaliacao,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            resultado_final,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ----------------------------------------------------------------------
    # RESUMO
    # ----------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RESULTADO FINAL — BASELINE")
    print("=" * 80)

    print("\nFAISS")
    print(
        f"Recall@1  : "
        f"{metricas_faiss['recall_at_1']:.2%}"
    )
    print(
        f"Recall@3  : "
        f"{metricas_faiss['recall_at_3']:.2%}"
    )
    print(
        f"Recall@5  : "
        f"{metricas_faiss['recall_at_5']:.2%}"
    )
    print(
        f"Recall@10 : "
        f"{metricas_faiss['recall_at_10']:.2%}"
    )
    print(
        f"MRR       : "
        f"{metricas_faiss['mrr']:.4f}"
    )

    print("\nRERANKER")
    print(
        f"Recall@1  : "
        f"{metricas_reranker['recall_at_1']:.2%}"
    )
    print(
        f"Recall@3  : "
        f"{metricas_reranker['recall_at_3']:.2%}"
    )
    print(
        f"MRR       : "
        f"{metricas_reranker['mrr']:.4f}"
    )

    print("\nResultado salvo em:")
    print(output_path)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Avalia o baseline de Retrieval e Reranking "
            "do pipeline RAG."
        )
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help=(
            "Caminho para o questions.json."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Arquivo JSON de saída."
        ),
    )

    args = parser.parse_args()

    avaliar(
        questions_path=args.questions,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
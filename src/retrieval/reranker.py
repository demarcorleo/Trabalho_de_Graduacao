from sentence_transformers import CrossEncoder

from config.settings import RERANKER_MODEL_NAME, RERANKER_TOP_N, RERANKER_MIN_SCORE


def carregar_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


def rerankear(resultados, query, reranker_model):
    if not resultados:
        return []

    pares = [(query, resultado["texto"]) for resultado in resultados]

    scores = reranker_model.predict(pares)

    for resultado, score in zip(resultados, scores):
        resultado["reranker_score"] = float(score)

    resultados_ordenados = sorted(
        resultados,
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    resultados_filtrados = [
        resultado
        for resultado in resultados_ordenados
        if resultado["reranker_score"] >= RERANKER_MIN_SCORE
    ]

    return resultados_filtrados[:RERANKER_TOP_N]
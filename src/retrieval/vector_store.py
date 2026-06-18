import os
import pickle
import faiss

from config.settings import INDEX_PATH, CHUNKS_PATH


def criar_indice(embeddings):
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


def salvar_indice(index, chunks):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("Índice e chunks salvos com sucesso!")


def carregar_indice():
    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def busca(query_vector, index, chunks, k):
    scores, indices = index.search(query_vector, k)

    resultados = []

    for pos, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(chunks):
            continue

        resultados.append({
            "texto": chunks[idx].page_content,
            "source": chunks[idx].metadata.get("source"),
            "score": float(scores[0][pos]),
            "indice": int(idx)
        })

    return resultados


def reset():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)

    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)

    print("Índice resetado com sucesso!")
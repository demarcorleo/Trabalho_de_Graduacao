import os
import pickle
import json

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

import faiss

from config.settings import (
    INDEX_PATH,
    CHUNKS_PATH,
)


def caminho_manifesto():
    return os.path.join(os.path.dirname(str(INDEX_PATH)), "index_manifest.json")


def configuracao_indice():
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }


def validar_manifesto():
    caminho = caminho_manifesto()
    if not os.path.exists(caminho):
        raise RuntimeError(
            "Índice sem manifesto de configuração (possivelmente criado antes do checkpoint). "
            "Execute 'python -m src.pipeline.pipeline_indexing' para reconstruí-lo com a configuração ativa."
        )
    with open(caminho, "r", encoding="utf-8") as arquivo:
        gravada = json.load(arquivo)
    esperada = configuracao_indice()
    if any(gravada.get(chave) != valor for chave, valor in esperada.items()):
        raise RuntimeError(
            f"Configuração do índice difere da atual. Índice={gravada}; atual={esperada}. "
            "Reconstrua o índice antes de avaliar ou consultar."
        )


def criar_indice(embeddings):
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


def salvar_indice(index, chunks):
    os.makedirs(
        os.path.dirname(INDEX_PATH),
        exist_ok=True,
    )

    os.makedirs(
        os.path.dirname(CHUNKS_PATH),
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    # Registrar a configuração associada ao índice para impedir reuso incompatível.
    with open(caminho_manifesto(), "w", encoding="utf-8") as arquivo:
        json.dump(configuracao_indice(), arquivo, indent=2, ensure_ascii=False)

    print(
        "Índice, chunks e manifesto salvos com sucesso!"
    )


def carregar_indice():
    validar_manifesto()
    index = faiss.read_index(
        str(INDEX_PATH)
    )

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def busca(
    query_vector,
    index,
    chunks,
    k,
):
    scores, indices = index.search(
        query_vector,
        k,
    )

    resultados = []

    for pos, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        resultados.append(
            {
                "texto": chunk.get_content(),
                "source": chunk.metadata.get(
                    "source"
                ),
                "score": float(
                    scores[0][pos]
                ),
                "indice": int(idx),
            }
        )

    return resultados


def reset():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)

    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)

    if os.path.exists(caminho_manifesto()):
        os.remove(caminho_manifesto())

    print(
        "Índice resetado com sucesso!"
    )
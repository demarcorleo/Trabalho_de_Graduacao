from config.settings import DOCUMENTS_PATH

from src.ingestion.ingestion import carregar_documentos
from src.ingestion.generate_chunking import gerar_chunks
from src.retrieval.generate_embedding import (
    carregar_modelo_embedding,
    gerar_embeddings,
)
from src.retrieval.vector_store import (
    criar_indice,
    salvar_indice,
    reset,
)


def indexar_documentos(resetar_indice: bool = True):
    if resetar_indice:
        reset()

    documentos = carregar_documentos(DOCUMENTS_PATH)

    if not documentos:
        raise ValueError("Nenhum documento encontrado para indexação.")

    chunks = gerar_chunks(documentos)

    if not chunks:
        raise ValueError("Nenhum chunk foi gerado.")

    tokenizer, model = carregar_modelo_embedding()

    embeddings = gerar_embeddings(
        chunks=chunks,
        tokenizer=tokenizer,
        model=model
    )

    index = criar_indice(embeddings)

    salvar_indice(index, chunks)

    print("Indexação concluída com sucesso!")


if __name__ == "__main__":
    indexar_documentos(resetar_indice=True)
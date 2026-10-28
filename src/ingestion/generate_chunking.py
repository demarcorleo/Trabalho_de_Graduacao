from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import MarkdownTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def gerar_chunks(documentos):
    """
    Gera chunks utilizando o MarkdownTextSplitter atual.

    Os documentos recebidos pelo pipeline são Documents do
    LlamaIndex. Antes da divisão, eles são convertidos para
    LangChain Documents para preservar o comportamento original
    do chunking.
    """

    documentos_langchain = []

    for documento in documentos:
        documentos_langchain.append(
            LangChainDocument(
                page_content=documento.get_content(),
                metadata=dict(documento.metadata),
            )
        )

    splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documentos_langchain)

    print(f"Total de chunks gerados: {len(chunks)}")

    return chunks
from llama_index.core.schema import TextNode
from langchain_text_splitters import MarkdownTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def gerar_chunks(documentos):
    """
    Gera os chunks utilizando o mesmo MarkdownTextSplitter
    utilizado anteriormente.

    A diferença é que a saída passa a ser composta por
    TextNode do LlamaIndex, eliminando a dependência de
    LangChain Document como representação dos chunks.
    """

    splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []

    for documento in documentos:
        texto = documento.get_content()
        metadata = dict(documento.metadata)

        # Mantém exatamente a lógica de divisão do
        # MarkdownTextSplitter utilizada no baseline.
        documento_temp = type(
            "DocumentoTemporario",
            (),
            {
                "page_content": texto,
                "metadata": metadata,
            },
        )()

        documentos_divididos = splitter.split_documents(
            [documento_temp]
        )

        for chunk in documentos_divididos:
            node = TextNode(
                text=chunk.page_content,
                metadata=dict(chunk.metadata),
            )

            chunks.append(node)

    print(f"Total de chunks gerados: {len(chunks)}")

    return chunks
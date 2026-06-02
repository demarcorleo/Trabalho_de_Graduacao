from langchain_text_splitters import MarkdownTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def gerar_chunks(documentos):
    splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documentos)

    print(f"Total de chunks gerados: {len(chunks)}")

    return chunks
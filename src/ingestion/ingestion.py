import os
import re
import unicodedata

import pymupdf4llm
from langchain_core.documents import Document


def limpar_texto(texto: str) -> str:
    """
    Limpa o texto extraído dos documentos sem alterar seu significado.
    """

    # Remove caracteres invisíveis/problemáticos
    texto = texto.replace("\x00", " ")
    texto = texto.replace("\ufeff", " ")

    # Normaliza caracteres unicode
    texto = unicodedata.normalize("NFKC", texto)

    # Corrige palavras quebradas por hifenização em PDFs
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)

    # Reduz múltiplas quebras de linha
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    # Reduz múltiplos espaços e tabs
    texto = re.sub(r"[ \t]+", " ", texto)

    # Remove espaço antes de pontuação
    texto = re.sub(r"\s+([,.;:!?])", r"\1", texto)

    return texto.strip()


def carregar_documentos(pasta: str):
    """
    Carrega arquivos PDF e TXT de uma pasta, extrai o texto,
    aplica limpeza e retorna uma lista de Documents.
    """

    documentos = []

    if not os.path.exists(pasta):
        raise FileNotFoundError(f"Pasta '{pasta}' não encontrada.")

    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)

        if not arquivo.lower().endswith((".pdf", ".txt")):
            continue

        print(f"Carregando arquivo: {arquivo}")

        if arquivo.lower().endswith(".pdf"):
            texto = pymupdf4llm.to_markdown(caminho)
        else:
            with open(caminho, "r", encoding="utf-8") as f:
                texto = f.read()

        texto = limpar_texto(texto)

        documento = Document(
            page_content=texto,
            metadata={"source": arquivo}
        )

        documentos.append(documento)

    print(f"Total de documentos carregados: {len(documentos)}")

    return documentos
"""Correspondência integral de evidências gold após normalização textual."""
import re
import unicodedata
from pathlib import Path

GOLD_MATCH_VERSION = "exact_normalized_v1"


def normalizar_source(source):
    if source is None:
        return ""
    return Path(str(source).replace("\\", "/")).name


def normalizar_texto(texto):
    if texto is None:
        return ""
    texto = unicodedata.normalize("NFKC", str(texto)).casefold()
    texto = texto.replace("…", "...").replace("⋯", "...")
    texto = texto.replace("−", "-").replace("–", "-").replace("—", "-")
    texto = texto.replace("\u200b", "").replace("\ufeff", "")
    texto = re.sub(r"<br\s*/?>", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"[\*_`#]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def gold_esta_no_chunk(gold_source, texto):
    referencia = normalizar_texto(gold_source)
    return bool(referencia) and referencia in normalizar_texto(texto)


def localizar_gold_chunks(chunks, source, gold_source):
    """Retorna índices com evidência integral, no documento indicado."""
    arquivo = normalizar_source(source)
    if not arquivo or not normalizar_texto(gold_source):
        return []

    indices = []
    for indice, chunk in enumerate(chunks):
        metadata = getattr(chunk, "metadata", {}) or {}

        if normalizar_source(metadata.get("source")) != arquivo:
            continue

        texto = (
            chunk.get_content()
            if hasattr(chunk, "get_content")
            else chunk.text
        )

        if gold_esta_no_chunk(gold_source, texto):
            indices.append(indice)

    return indices

"""Indexação SWaT; exige corpus PDF esperado e preserva índice existente em falha anterior à construção."""
from pathlib import Path
from config.settings import DOCUMENTS_PATH, INDEX_PATH, CHUNKS_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from src.ingestion.ingestion import carregar_documentos
from src.ingestion.generate_chunking import gerar_chunks
from src.retrieval.generate_embedding import carregar_modelo_embedding, gerar_embeddings
from src.retrieval.vector_store import criar_indice, salvar_indice


def conferir_corpus(pasta=DOCUMENTS_PATH):
    pasta = Path(pasta)
    if not pasta.is_dir():
        raise FileNotFoundError(f"Corpus não localizado: {pasta}")
    arquivos = sorted(p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() in {'.pdf', '.txt'})
    pdfs = [p for p in arquivos if p.suffix.lower() == '.pdf']
    extras = [p for p in arquivos if p.suffix.lower() != '.pdf']
    if len(pdfs) != 14 or extras:
        raise ValueError(f"Esperados 14 PDFs SWaT sem TXT extra; encontrados {len(pdfs)} PDFs e {len(extras)} TXT em {pasta}. Confira o corpus antes de indexar.")
    suspeitos = [p.name for p in pdfs if any(k in p.name.lower() for k in ('manual_vpn_c3', 'regimento_c3', 'ppc_', 'furg'))]
    if suspeitos:
        raise ValueError(f"Possíveis documentos institucionais no corpus SWaT: {suspeitos}")
    print(f"Corpus: {len(pdfs)} PDFs. Confira os nomes com a lista de 14 documentos do TCC:")
    for p in pdfs:
        print('  ', p.name)
    return arquivos


def indexar_documentos(resetar_indice=False):
    # Não apagar o índice antigo antes de testar documentos e gerar os novos artefatos.
    conferir_corpus()
    print(f"Configuração ativa: chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    documentos = carregar_documentos(DOCUMENTS_PATH)
    if len(documentos) != 14:
        raise ValueError(f"Esperados 14 documentos processados, recebidos {len(documentos)}")
    chunks = gerar_chunks(documentos)
    if not chunks:
        raise ValueError("Nenhum bloco gerado.")
    tokenizer, model = carregar_modelo_embedding()
    embeddings = gerar_embeddings(chunks=chunks, tokenizer=tokenizer, model=model)
    index = criar_indice(embeddings)
    salvar_indice(index, chunks)
    print(f"Índice atualizado: {index.ntotal} vetores. Arquivos: {INDEX_PATH}, {CHUNKS_PATH}")
    print('Recomenda-se backup prévio e guardar resultados históricos separadamente.')


if __name__ == '__main__':
    indexar_documentos()
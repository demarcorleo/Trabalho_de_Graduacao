from langchain_community.llms import Llamacpp
from ingestion.ingestion import carregar_documentos
from retrieval.vector_store import carregar_indice, busca
from retrieval.reranker import carregar_reranker, rerankear
from config.settings import MODEL_NAME1, EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE
from config.prompts import PROMPT_VERSION, PROMPTS_DIR


class PipelineRAG:
    def __init__(self):
        self.llm = Llamacpp(model_path=MODEL_NAME1)
        self.embedding_model_name = EMBEDDING_MODEL_NAME
        self.embedding_batch_size = EMBEDDING_BATCH_SIZE
        self.reranker_model = carregar_reranker()
        self.prompts = self.carregar_prompts()

    def carregar_prompts(self):
        prompts_path = PROMPTS_DIR / PROMPT_VERSION
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts para a versão '{PROMPT_VERSION}' não encontrados em '{prompts_path}'.")

        prompts = {}
        for arquivo in prompts_path.glob("*.txt"):
            with open(arquivo, "r", encoding="utf-8") as f:
                prompts[arquivo.stem] = f.read()

        return prompts

    def processar_query(self, query):
        index, chunks = carregar_indice()
        query_vector = self.gerar_embedding(query)
        resultados = busca(query_vector, index, chunks, k=10)
        resultados_rerankeados = rerankear(resultados, query, self.reranker_model)

        resposta = self.gerar_resposta(query, resultados_rerankeados)
        return resposta

    def gerar_embedding(self, texto):
        # Implementação para gerar embedding usando o modelo especificado
        pass

    def gerar_resposta(self, query, resultados):
        # Implementação para gerar resposta usando o LLM e os resultados relevantes
        pass
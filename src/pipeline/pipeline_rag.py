import yaml
from langchain_community.llms import LlamaCpp

from config.settings import (
    LLM_MODEL_PATH,
    N_CTX,
    N_THREADS,
    N_GPU_LAYERS,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    PROMPTS_DIR,
    PROMPT_VERSION,
)

from src.retrieval.vector_store import carregar_indice, busca
from src.retrieval.reranker import carregar_reranker, rerankear
from src.retrieval.generate_embedding import (
    carregar_modelo_embedding,
    gerar_embedding_query,
)
from config.settings import PROMPT_TEMPLATE



class PipelineRAG:
    def __init__(self):
        self.index, self.chunks = carregar_indice()

        self.tokenizer, self.embedding_model = carregar_modelo_embedding()

        self.reranker_model = carregar_reranker()

        self.llm = LlamaCpp(
            model_path=str(LLM_MODEL_PATH),
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        self.prompts = self.carregar_prompts()

    def carregar_prompts(self):
        prompt_path = PROMPTS_DIR / PROMPT_VERSION

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Arquivo de prompt não encontrado: {prompt_path}"
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def processar_query(self, query: str) -> str:
        query_vector = gerar_embedding_query(
            query,
            self.tokenizer,
            self.embedding_model
        )

        resultados = busca(
            query_vector=query_vector,
            index=self.index,
            chunks=self.chunks,
            k=TOP_K
        )

        resultados_rerankeados = rerankear(
            resultados=resultados,
            query=query,
            reranker_model=self.reranker_model
        )

        return self.gerar_resposta(query, resultados_rerankeados)

    def gerar_resposta(self, query: str, resultados: list) -> str:
        if not resultados:
            return "Não encontrei informações relevantes nos documentos."

        contexto = "\n\n".join(
            resultado["texto"]
            for resultado in resultados
        )

        template_config = self.prompts["templates"][PROMPT_TEMPLATE]

        system_prompt = template_config["system"]
        prompt_template = template_config["template"]

        prompt_usuario = prompt_template.format(
            context=contexto,
            query=query
        )

        prompt_final = f"{system_prompt}\n\n{prompt_usuario}"

        return self.llm.invoke(prompt_final)
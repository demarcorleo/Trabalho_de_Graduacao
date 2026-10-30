import yaml

from llama_index.llms.ollama import Ollama

from config.settings import (
    LLM_MODEL_NAME,
    N_CTX,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    OLLAMA_REQUEST_TIMEOUT,
    TOP_K,
    PROMPTS_DIR,
    PROMPT_VERSION,
    PROMPT_TEMPLATE,
)

from src.retrieval.vector_store import carregar_indice, busca
from src.retrieval.reranker import carregar_reranker, rerankear
from src.retrieval.generate_embedding import (
    carregar_modelo_embedding,
    gerar_embedding_query,
)


class PipelineRAG:
    def __init__(self):
        self.index, self.chunks = carregar_indice()

        self.tokenizer, self.embedding_model = carregar_modelo_embedding()

        self.reranker_model = carregar_reranker()

        self.llm = Ollama(
            model=LLM_MODEL_NAME,
            request_timeout=OLLAMA_REQUEST_TIMEOUT,
            context_window=N_CTX,
            temperature=TEMPERATURE,
            additional_kwargs={
                "num_predict": MAX_TOKENS,
                "top_p": TOP_P,
            },
            thinking=False,
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

    def recuperar_resultados(self, query: str) -> list:
        query_vector = gerar_embedding_query(
            query,
            self.tokenizer,
            self.embedding_model,
        )

        resultados = busca(
            query_vector=query_vector,
            index=self.index,
            chunks=self.chunks,
            k=TOP_K,
        )

        resultados_rerankeados = rerankear(
            resultados=resultados,
            query=query,
            reranker_model=self.reranker_model,
        )

        return resultados_rerankeados

    def processar_query(self, query: str) -> str:
        resultados_rerankeados = self.recuperar_resultados(query)

        return self.gerar_resposta(
            query,
            resultados_rerankeados,
        )

    def processar_query_detalhado(self, query: str) -> dict:
        resultados_rerankeados = self.recuperar_resultados(query)

        resposta = self.gerar_resposta(
            query,
            resultados_rerankeados,
        )

        fontes = []

        for resultado in resultados_rerankeados:
            fontes.append(
                {
                    "source": resultado.get("source"),
                    "score": resultado.get("score"),
                    "reranker_score": resultado.get("reranker_score"),
                    "indice": resultado.get("indice"),
                    "texto": resultado.get("texto"),
                }
            )

        return {
            "resposta": resposta,
            "fontes": fontes,
        }

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
            query=query,
        )

        prompt_final = f"{system_prompt}\n\n{prompt_usuario}"

        resposta = self.llm.complete(prompt_final)

        return resposta.text
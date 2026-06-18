from src.pipeline.pipeline_indexing import indexar_documentos
from src.pipeline.pipeline_rag import PipelineRAG


def main():
    print("Preparando base de conhecimento...")
    indexar_documentos(resetar_indice=True)

    pipeline = PipelineRAG()

    print("\nChatbot RAG iniciado.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Você: ").strip()

        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando chatbot...")
            break

        if not pergunta:
            continue

        resposta = pipeline.processar_query(pergunta)

        print("\nAssistente:")
        print(resposta)
        print()


if __name__ == "__main__":
    main()
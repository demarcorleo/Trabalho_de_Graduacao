# Ambiente Computacional

Este documento apresenta as características do ambiente computacional utilizado na execução atual do pipeline RAG desenvolvido para consulta à documentação técnica da planta Secure Water Treatment (SWaT).

As informações foram coletadas em 11 de novembro de 2026.

## 1. Hardware

| Componente | Especificação |
|---|---|
| Processador | Intel Core i7-13700K (13ª geração) |
| Processadores lógicos | 24 |
| Memória RAM | 62 GiB |
| Placa gráfica dedicada | NVIDIA GeForce GTX 1050 Ti |
| Memória de vídeo | 4 GB |
| Placa gráfica integrada | Intel UHD Graphics (identificada pelo controlador integrado) |

## 2. Sistema operacional

| Componente | Versão |
|---|---|
| Distribuição Linux | Ubuntu 22.04.5 LTS |
| Kernel Linux | 6.8.0-138-generic |
| Python | 3.10.12 |
| Gerenciamento de ambiente | Python virtual environment (`venv`) |

## 3. Ambiente de aprendizado de máquina

| Componente | Versão |
|---|---|
| PyTorch | 2.13.0+cu130 |
| CUDA associada ao PyTorch | 13.0 |
| Driver NVIDIA | 535.309.01 |
| CUDA indicada pelo driver | 12.2 |
| Transformers | 5.12.1 |
| Sentence Transformers | 5.6.0 |
| FAISS CPU | 1.14.3 |
| ONNX Runtime | 1.23.2 |
| NumPy | 2.2.6 |

Durante a execução atual, o PyTorch apresentou incompatibilidade entre a versão CUDA utilizada em sua instalação e o driver NVIDIA disponível.

A verificação `torch.cuda.is_available()` retornou `False`, indicando indisponibilidade de aceleração CUDA para operações executadas pelo PyTorch nesse ambiente.

Essa condição não permite concluir, isoladamente, qual dispositivo foi utilizado pelo Ollama.

## 4. Processamento documental e recuperação de informações

| Biblioteca | Versão |
|---|---|
| PyMuPDF | 1.27.2.3 |
| PyMuPDF4LLM | 1.27.2.3 |
| LlamaIndex | 0.14.25 |
| LlamaIndex Core | 0.14.25 |
| LangChain Text Splitters | 1.1.2 |
| FAISS CPU | 1.14.3 |
| Scikit-learn | 1.7.2 |
| Transformers | 5.12.1 |

## 5. Geração de respostas e interface

| Componente | Versão |
|---|---|
| Ollama (servidor) | 0.40.12 |
| Biblioteca Python `ollama` | 0.6.3 |
| LlamaIndex LLMs Ollama | 0.11.0 |
| Streamlit | 1.65.0 |
| Modelo de linguagem | `qwen3.5:4b-bf16` |
| Identificador local do modelo | `530cc2ae9e85` |

O Streamlit é utilizado para disponibilizar uma interface interativa de consulta ao sistema RAG, permitindo que o usuário formule perguntas e receba respostas produzidas pelo modelo de linguagem com base no contexto recuperado da documentação técnica.

## 6. Configuração do pipeline

| Parâmetro | Configuração |
|---|---|
| Extração documental | PyMuPDF4LLM |
| Segmentação | MarkdownTextSplitter |
| Tamanho dos chunks | 1500 caracteres |
| Sobreposição | 300 caracteres |
| Modelo de embeddings | `intfloat/multilingual-e5-base` |
| Índice vetorial | FAISS IndexFlatIP |
| Quantidade de chunks indexados | 996 |
| Recuperação inicial (top-k) | 10 |
| Modelo de reranking | `BAAI/bge-reranker-v2-m3` |
| Quantidade final de trechos (top-n) | 3 |
| Limiar do reranker | 0,2 |
| Modelo generativo | `qwen3.5:4b-bf16` |
| Janela de contexto | 4096 |
| Máximo de tokens gerados | 1024 |
| Temperatura | 0,2 |
| Top-p | 0,9 |

## 7. Considerações sobre reprodutibilidade

As especificações apresentadas documentam o ambiente identificado durante a etapa de validação e execução atual do pipeline.

A reprodução dos experimentos requer a utilização de um conjunto documental correspondente, versões compatíveis das bibliotecas, modelos e parâmetros de configuração.

Os documentos PDF originais não são redistribuídos neste repositório. Sua identificação e as condições necessárias para a preparação do corpus devem ser consultadas na documentação do projeto.

As especificações deste documento não devem ser atribuídas automaticamente a execuções históricas realizadas sob outras configurações de software ou hardware.
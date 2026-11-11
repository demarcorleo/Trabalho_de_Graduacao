# Sistema de Perguntas e Respostas com RAG para Documentação Técnica Industrial

Este repositório apresenta o desenvolvimento de uma aplicação de inteligência artificial generativa baseada em **Retrieval-Augmented Generation (RAG)**, destinada à consulta de documentação técnica de equipamentos industriais associados à planta **Secure Water Treatment (SWaT)**.

A aplicação possui uma interface interativa desenvolvida com **Streamlit**, permitindo que usuários formulem perguntas em linguagem natural e recebam respostas produzidas por um modelo de linguagem a partir de informações recuperadas dos documentos técnicos.

O projeto também contempla uma avaliação experimental de diferentes técnicas e configurações de processamento documental, segmentação textual, representação vetorial e recuperação de informações. Esses experimentos subsidiaram a seleção dos parâmetros utilizados na aplicação final.

## 1. Objetivo do projeto

Desenvolver uma aplicação de perguntas e respostas baseada em RAG para facilitar o acesso a informações presentes na documentação técnica de equipamentos industriais, utilizando um modelo de linguagem generativa integrado a mecanismos de recuperação de informações.

Especificamente, o projeto contempla:

- Implementação de um pipeline para ingestão e processamento de documentos PDF.
- Comparação de técnicas de extração documental.
- Avaliação de estratégias de segmentação textual (*chunking*).
- Comparação de modelos de embeddings e configurações de recuperação.
- Aplicação de reranqueamento para selecionar trechos relevantes.
- Integração de um modelo de linguagem executado localmente.
- Desenvolvimento de uma interface interativa com Streamlit.
- Avaliação da recuperação de informações e da qualidade das respostas geradas.

## 2. Aplicação desenvolvida

O produto final consiste em uma aplicação de consulta documental que utiliza inteligência artificial generativa para responder perguntas relacionadas às especificações de equipamentos industriais presentes no corpus SWaT.

Por meio da interface Streamlit, o usuário pode inserir perguntas em linguagem natural. O sistema recupera informações potencialmente relevantes da documentação indexada, seleciona os trechos mais adequados e os encaminha ao modelo de linguagem para elaboração da resposta.

O funcionamento da aplicação está organizado nas seguintes etapas:

1. **Consulta do usuário:** uma pergunta é inserida na interface Streamlit.
2. **Representação vetorial:** a pergunta é convertida em embedding.
3. **Recuperação:** o FAISS identifica os trechos semanticamente mais próximos.
4. **Reranqueamento:** um modelo CrossEncoder reordena os candidatos recuperados.
5. **Construção do contexto:** os trechos selecionados são organizados para compor o prompt.
6. **Geração:** o modelo Qwen3.5 produz uma resposta fundamentada no contexto recebido.
7. **Apresentação:** a resposta é exibida na interface da aplicação.

A geração utiliza um modelo executado localmente por meio do Ollama, sem depender de uma API externa para a inferência do modelo de linguagem.

## 3. Arquitetura do pipeline RAG

O pipeline desenvolvido possui duas etapas principais.

### 3.1. Preparação e indexação documental

Nesta etapa, os documentos técnicos são processados para construção da base de conhecimento utilizada pela aplicação.

O processamento compreende:

- Extração textual de documentos PDF com PyMuPDF4LLM.
- Normalização do conteúdo extraído.
- Segmentação do texto com MarkdownTextSplitter.
- Geração de embeddings utilizando Hugging Face Transformers.
- Armazenamento das representações vetoriais em um índice FAISS.

A indexação é realizada previamente à utilização da aplicação.

### 3.2. Consulta e geração de respostas

Durante a utilização da aplicação, uma pergunta é submetida ao pipeline de recuperação e geração.

A recuperação inicial utiliza FAISS, seguida de um modelo de reranqueamento responsável por reordenar os candidatos.

Os trechos selecionados são incorporados ao prompt enviado ao modelo generativo, que produz uma resposta a partir do contexto disponibilizado.

Essa arquitetura permite separar a preparação da base documental do processamento das consultas realizadas pelo usuário.

## 4. Configuração selecionada

A configuração final do pipeline foi definida a partir dos experimentos realizados durante o desenvolvimento do projeto.

| Componente | Configuração |
|---|---|
| Extração documental | PyMuPDF4LLM |
| Segmentação textual | MarkdownTextSplitter |
| Chunk size | 1500 caracteres |
| Chunk overlap | 300 caracteres |
| Modelo de embeddings | `intfloat/multilingual-e5-base` |
| Índice vetorial | FAISS IndexFlatIP |
| Recuperação inicial | Top-k = 10 |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Trechos após reranking | Top-n = 3 |
| Limiar do reranker | 0,2 |
| Modelo generativo | `qwen3.5:4b-bf16` |
| Janela de contexto | 4096 |
| Máximo de tokens | 1024 |
| Temperatura | 0,2 |
| Top-p | 0,9 |
| Interface | Streamlit |
| Execução do LLM | Ollama |

Na indexação final realizada com essa configuração, foram gerados **996 chunks a partir de 14 documentos PDF**.

As configurações são centralizadas em `config/settings.py` e podem ser sobrescritas pelas variáveis de ambiente correspondentes.

## 5. Conjunto documental

O corpus experimental é composto por 14 documentos PDF relacionados à planta Secure Water Treatment (SWaT) e a equipamentos industriais utilizados em processos de tratamento de água.

O conjunto inclui documentos técnicos, catálogos e folhas de dados de equipamentos como:

- Transmissores e instrumentos de medição.
- Sensores e chaves de pressão.
- Bombas industriais.
- Válvulas.
- Fluxômetros.
- Sistemas de filtração.
- Membranas de osmose reversa.

**Os documentos PDF originais não são disponibilizados neste repositório.**

Essa decisão busca respeitar as condições de distribuição da documentação técnica dos respectivos fabricantes.

Para reproduzir os experimentos, é necessário obter os documentos correspondentes e organizá-los no diretório configurado para a ingestão documental.

## 6. Avaliação experimental

Antes da consolidação da aplicação, foram realizados experimentos para avaliar diferentes componentes do pipeline RAG.

### 6.1. Extração documental

Foram consideradas diferentes ferramentas para extração de conteúdo de documentos PDF, observando aspectos de desempenho computacional e recuperação de informações.

### 6.2. Segmentação textual

Foram avaliadas diferentes configurações de tamanho e sobreposição dos blocos de texto.

A configuração C4, com tamanho de 1500 caracteres e sobreposição de 300 caracteres, foi selecionada considerando os critérios de recuperação priorizados no experimento.

### 6.3. Modelos de embeddings

Foram comparados modelos de representação vetorial para identificar uma configuração adequada à recuperação semântica de informações técnicas.

O modelo selecionado foi `intfloat/multilingual-e5-base`.

### 6.4. Recuperação e reranqueamento

A avaliação incluiu consultas em português e inglês, considerando a posição de documentos e trechos relevantes no conjunto recuperado.

Foram utilizadas métricas como Recall@k e Mean Reciprocal Rank (MRR).

### 6.5. Geração de respostas

A aplicação foi avaliada utilizando um conjunto de 15 perguntas relacionadas às especificações técnicas dos equipamentos documentados.

Na execução final registrada, foram obtidos:

| Métrica | Resultado |
|---|---:|
| Similaridade semântica média | 0,8638 |
| Sobreposição lexical média | 0,2464 |

Essas métricas representam a proximidade entre as respostas geradas e as respostas de referência, não devendo ser interpretadas diretamente como taxas de acerto factual.

A análise das respostas também identificou limitações relacionadas à recuperação de informações específicas, especialmente valores numéricos, códigos de equipamentos e dados apresentados em tabelas técnicas.

Esses resultados são considerados parte da avaliação experimental do sistema.

## 7. Organização do repositório

O código do projeto está organizado principalmente nos seguintes diretórios:

| Diretório | Responsabilidade |
|---|---|
| `config/` | Configuração do pipeline e templates de prompts |
| `src/benchmark/` | Procedimentos de benchmark de extração |
| `src/ingestion/` | Ingestão documental e segmentação |
| `src/retrieval/` | Embeddings, índice FAISS e reranking |
| `src/pipeline/` | Construção do índice e execução do RAG |
| `src/evaluation/` | Avaliação experimental dos componentes |
| `docs/` | Documentação técnica e ambiente computacional, quando adicionados |

A aplicação também inclui uma interface desenvolvida com Streamlit, integrada ao pipeline de consultas.

## 8. Execução do projeto

A execução requer Python, dependências compatíveis, Ollama e o modelo de linguagem utilizado pelo sistema.

### 8.1. Preparação do ambiente

Recomenda-se utilizar um ambiente virtual Python para isolar as dependências do projeto.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

As bibliotecas necessárias devem ser instaladas de acordo com as versões documentadas para o ambiente experimental.

### 8.2. Preparação dos documentos

Os documentos PDF devem ser obtidos separadamente e colocados no diretório correspondente à variável `DOCUMENTS_PATH`.

Na configuração utilizada nos experimentos finais, o corpus foi organizado em:

```text
data/documents/Swat/
```

### 8.3. Construção do índice vetorial

Com as dependências instaladas e o corpus disponível:

```bash
python -m src.pipeline.pipeline_indexing
```

O procedimento realiza extração, segmentação, geração de embeddings e construção do índice FAISS.

### 8.4. Avaliação de consultas

```bash
python -m src.evaluation.evaluate_query_processing
```

### 8.5. Avaliação da geração de respostas

```bash
python -m src.evaluation.evaluate_generation
```

### 8.6. Execução da aplicação Streamlit

A aplicação é iniciada por meio do Streamlit, utilizando o arquivo responsável pela interface.

O comando específico de inicialização será documentado após a identificação do arquivo de entrada da aplicação no repositório.

O serviço Ollama e o modelo correspondente devem estar disponíveis para que a aplicação realize a geração das respostas.

## 9. Ambiente computacional

O ambiente utilizado na execução atual do pipeline apresenta as seguintes características:

| Componente | Especificação |
|---|---|
| Sistema operacional | Ubuntu 22.04.5 LTS |
| Processador | Intel Core i7-13700K |
| Memória RAM | 62 GiB |
| GPU | NVIDIA GeForce GTX 1050 Ti — 4 GB |
| Python | 3.10.12 |
| PyTorch | 2.13.0+cu130 |
| Ollama | 0.40.12 |
| Streamlit | 1.65.0 |

As especificações completas do ambiente de execução, incluindo hardware, sistema operacional, versões das bibliotecas, configuração do pipeline e limitações relacionadas à aceleração CUDA, estão disponíveis no documento [Ambiente Computacional](config/ambiente_computacional.md).

Essas informações correspondem ao ambiente identificado durante a etapa final de execução e validação do sistema.

## 10. Reprodutibilidade

O repositório disponibiliza o código-fonte dos componentes desenvolvidos e os scripts empregados nas avaliações experimentais.

A documentação de reprodutibilidade contempla o ambiente computacional, as dependências, as configurações e os procedimentos necessários para executar as etapas do pipeline.

A reprodução dos resultados depende da disponibilidade dos documentos técnicos correspondentes, das versões dos modelos e das condições de execução adotadas.

Os resultados experimentais históricos podem apresentar diferenças em relação a novas execuções devido a alterações de versões, condições de processamento e características dos modelos utilizados.

## 11. Limitações e trabalhos futuros

Os experimentos demonstraram a viabilidade de utilização do pipeline RAG para consultas sobre documentação técnica industrial, mas também evidenciaram limitações.

Entre elas, destacam-se:

- Dificuldades de recuperação de informações numéricas específicas.
- Sensibilidade à estrutura de tabelas presentes nos documentos PDF.
- Recuperação de documentos relevantes sem identificação do trecho necessário.
- Possibilidade de eliminação de candidatos relevantes durante o reranking.
- Dependência da qualidade do contexto recuperado para a geração de respostas.
- Limitações das métricas automáticas na identificação de erros factuais.

Como possibilidades de aprimoramento, destacam-se a investigação de recuperação híbrida, a ampliação do conjunto de candidatos, o ajuste de estratégias de reranqueamento e a incorporação de métodos específicos para recuperação de informações estruturadas.

Essas possibilidades representam trabalhos futuros e não fazem parte da configuração experimental consolidada.

## 12. Contexto acadêmico

Projeto desenvolvido como parte de um Trabalho de Graduação em Engenharia de Automação da Universidade Federal do Rio Grande (FURG).

O trabalho integra o desenvolvimento de uma aplicação de inteligência artificial generativa com recuperação aumentada por documentos, acompanhado de uma avaliação experimental das técnicas e configurações utilizadas.
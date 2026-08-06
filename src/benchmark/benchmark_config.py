from pathlib import Path


# ============================================================
# CAMINHOS
# ============================================================

# Estrutura esperada:
#
# projeto/
# ├── data/
# │   └── documents/
# └── src/
#     └── benchmark/
#         ├── benchmark_config.py
#         ├── base_extractor.py
#         ├── extractor_factory.py
#         ├── extractors.py
#         └── main.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "documents"

RESULTS_DIR = PROJECT_ROOT / "results" / "extractor_benchmark"

EXTRACTED_TEXT_DIR = RESULTS_DIR / "extracted_texts"

RESULTS_CSV_PATH = RESULTS_DIR / "benchmark_results.csv"

SUMMARY_CSV_PATH = RESULTS_DIR / "benchmark_summary.csv"

ERROR_LOG_PATH = RESULTS_DIR / "benchmark_errors.txt"


# ============================================================
# DATASET
# ============================================================

PDF_PATTERN = "*.pdf"

# Caso existam PDFs dentro de subpastas:
SEARCH_RECURSIVELY = False


# ============================================================
# BIBLIOTECAS DO BENCHMARK PRINCIPAL
# ============================================================

# Apenas ferramentas que podem participar do protocolo:
#
# PDF pesquisável -> extração textual -> TXT UTF-8
#
# Você pode comentar uma biblioteca temporariamente sem alterar
# nenhuma outra parte do benchmark.

ENABLED_EXTRACTORS = [
    "pymupdf",
    "docling",
]


# Ferramentas registradas, mas não incluídas no benchmark textual
# principal porque utilizam OCR ou dependem de uma implementação
# externa ainda não especificada.

EXCLUDED_FROM_TEXT_BENCHMARK = [
    "ocrmypdf",
    "pytesseract",
    "test.sh",
]


# ============================================================
# FORMATO COMUM DE SAÍDA
# ============================================================

OUTPUT_EXTENSION = ".txt"

OUTPUT_ENCODING = "utf-8"

PAGE_SEPARATOR = "\n\n"

# Normalização mínima aplicada igualmente a todas as bibliotecas.
NORMALIZE_LINE_ENDINGS = True

REMOVE_TRAILING_SPACES = True

# Não remove cabeçalhos, rodapés, listas, tabelas ou marcadores.
# Isso faz parte do resultado produzido por cada biblioteca.
COLLAPSE_EXCESSIVE_BLANK_LINES = False


# ============================================================
# EXECUÇÃO DO BENCHMARK
# ============================================================

# Execuções realizadas antes da medição.
# Servem para aquecer caches e inicializações internas.
WARMUP_RUNS = 1

# Execuções que serão realmente registradas.
MEASURED_RUNS = 5

# Continua o benchmark mesmo se uma biblioteca falhar.
FAIL_FAST = False

# Salva o texto extraído por cada biblioteca.
SAVE_EXTRACTED_TEXT = True

# Salva somente o texto da primeira repetição medida.
# Evita criar cinco cópias idênticas.
SAVE_ONLY_FIRST_MEASURED_RUN = True


# ============================================================
# PARÂMETROS COMUNS
# ============================================================

# Nenhum extrator do benchmark principal deve realizar OCR.
ENABLE_OCR = False

# Nenhum extrator deve usar LLM.
ENABLE_LLM = False

# Nenhum extrator deve executar uma rotina exclusiva de tabelas.
ENABLE_TABLE_EXTRACTION = False

# Nenhum extrator deve salvar imagens.
ENABLE_IMAGE_EXTRACTION = False


# ============================================================
# PYMUPDF
# ============================================================

PYMUPDF_SORT_TEXT = False


# ============================================================
# PYPDF
# ============================================================

PYPDF_EXTRACTION_MODE = "plain"


# ============================================================
# DOCLING
# ============================================================

DOCLING_ENABLE_OCR = False

DOCLING_ENABLE_TABLE_STRUCTURE = False


# ============================================================
# MARKER
# ============================================================

MARKER_MODE = "fast"

MARKER_DISABLE_OCR = True

MARKER_DISABLE_IMAGE_EXTRACTION = True

MARKER_USE_LLM = False

MARKER_OUTPUT_FORMAT = "markdown"


# ============================================================
# MARKITDOWN
# ============================================================

MARKITDOWN_ENABLE_PLUGINS = False


# ============================================================
# PDFPLUMBER
# ============================================================

PDFPLUMBER_X_TOLERANCE = 3

PDFPLUMBER_Y_TOLERANCE = 3

PDFPLUMBER_USE_TEXT_FLOW = False


# ============================================================
# PDFMINER
# ============================================================

PDFMINER_LINE_MARGIN = 0.5

PDFMINER_WORD_MARGIN = 0.1

PDFMINER_CHAR_MARGIN = 2.0


# ============================================================
# UNSTRUCTURED
# ============================================================

# "fast" utiliza a camada textual existente no PDF.
UNSTRUCTURED_STRATEGY = "fast"


# ============================================================
# TEST.SH
# ============================================================

# O script deverá imprimir o texto extraído em stdout.
# Deixe como None enquanto ele não estiver definido.

TEST_SHELL_SCRIPT_PATH = None

TEST_SHELL_TIMEOUT_SECONDS = 300
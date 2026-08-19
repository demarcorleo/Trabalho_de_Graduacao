from pathlib import Path


# ============================================================
# CAMINHOS GERAIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "documents"

RESULTS_DIR = PROJECT_ROOT / "results" / "extractor_benchmark"

EXTRACTED_TEXT_DIR = RESULTS_DIR / "extracted_texts"

RESULTS_CSV_PATH = RESULTS_DIR / "benchmark_results.csv"

SUMMARY_CSV_PATH = RESULTS_DIR / "benchmark_summary.csv"

ERROR_LOG_PATH = RESULTS_DIR / "benchmark_errors.txt"

MINERU_SERVER_LOG_PATH = RESULTS_DIR / "mineru_server.log"


# ============================================================
# DATASET
# ============================================================

PDF_PATTERN = "*.pdf"

SEARCH_RECURSIVELY = False


# ============================================================
# BIBLIOTECAS HABILITADAS
# ============================================================

ENABLED_EXTRACTORS = [
    "pymupdf",
    "pypdf",
    "pdfminer",
    "pdfplumber",
    "pymupdf4llm",
    "docling",
    "marker",
    "markitdown",
    "mineru",
    "unstructured",
]


# ============================================================
# FORMATO COMUM DE SAÍDA
# ============================================================

OUTPUT_EXTENSION = ".txt"

OUTPUT_ENCODING = "utf-8"

NORMALIZE_LINE_ENDINGS = True

REMOVE_TRAILING_SPACES = True

COLLAPSE_EXCESSIVE_BLANK_LINES = False


# ============================================================
# EXECUÇÃO DO BENCHMARK
# ============================================================

WARMUP_RUNS = 1

MEASURED_RUNS = 5

FAIL_FAST = False

SAVE_EXTRACTED_TEXT = True

SAVE_ONLY_FIRST_MEASURED_RUN = True


# ============================================================
# PyMuPDF
# ============================================================

PYMUPDF_SORT_TEXT = False


# ============================================================
# pypdf
# ============================================================

# "plain" = comportamento padrão de page.extract_text()
PYPDF_EXTRACTION_MODE = "plain"


# ============================================================
# PDFMiner
# ============================================================

PDFMINER_LINE_MARGIN = 0.5

PDFMINER_WORD_MARGIN = 0.1

PDFMINER_CHAR_MARGIN = 2.0


# ============================================================
# pdfplumber
# ============================================================

PDFPLUMBER_X_TOLERANCE = 3

PDFPLUMBER_Y_TOLERANCE = 3

PDFPLUMBER_USE_TEXT_FLOW = False


# ============================================================
# Docling
# ============================================================

DOCLING_ENABLE_OCR = False

DOCLING_ENABLE_TABLE_STRUCTURE = False


# ============================================================
# Marker
# ============================================================

MARKER_MODE = "fast"

MARKER_DISABLE_OCR = True

MARKER_DISABLE_IMAGE_EXTRACTION = True

MARKER_USE_LLM = False

MARKER_OUTPUT_FORMAT = "markdown"


# ============================================================
# MarkItDown
# ============================================================

MARKITDOWN_ENABLE_PLUGINS = False


# ============================================================
# Unstructured
# ============================================================

UNSTRUCTURED_STRATEGY = "fast"

# Idioma predominante do dataset.
# "por" corresponde ao português no Unstructured.
UNSTRUCTURED_LANGUAGES = ["por"]

# ============================================================
# MinerU
# ============================================================

MINERU_METHOD = "txt"

MINERU_BACKEND = "pipeline"

MINERU_ENABLE_FORMULA = False

MINERU_ENABLE_TABLE = False

MINERU_ENABLE_IMAGE_ANALYSIS = False

MINERU_API_HOST = "127.0.0.1"

MINERU_API_PORT = 8765

MINERU_API_STARTUP_TIMEOUT_SECONDS = 300

MINERU_PROCESS_TIMEOUT_SECONDS = 3600
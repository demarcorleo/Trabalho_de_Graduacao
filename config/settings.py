import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# =========================
# Prompts
# =========================

PROMPTS_DIR = Path(
    os.getenv(
        "PROMPTS_DIR",
        BASE_DIR / "config/prompts"
    )
)

PROMPT_VERSION = os.getenv(
    "PROMPT_VERSION",
    "v1.yml"
)

# =========================
# Documentos
# =========================

DOCUMENTS_PATH = Path(
    os.getenv(
        "DOCUMENTS_PATH",
        BASE_DIR / "data/documents"
    )
)

# =========================
# Chunking
# =========================

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", 1000)
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", 200)
)

# =========================
# Embeddings
# =========================

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "intfloat/multilingual-e5-base"
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", 16)
)

# =========================
# Vetor Store
# =========================

INDEX_PATH = Path(
    os.getenv(
        "INDEX_PATH",
        BASE_DIR / "data/indexes/faiss_index.bin"
    )
)

CHUNKS_PATH = Path(
    os.getenv(
        "CHUNKS_PATH",
        BASE_DIR / "data/indexes/chunks.pkl"
    )
)

# =========================
# Retrieval
# =========================

TOP_K = int(
    os.getenv("TOP_K", 10)
)

# =========================
# Reranker
# =========================

RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "BAAI/bge-reranker-v2-m3"
)

RERANKER_TOP_N = int(
    os.getenv("RERANKER_TOP_N", 3)
)

RERANKER_MIN_SCORE = float(
    os.getenv("RERANKER_MIN_SCORE", 0.2)
)

# =========================
# LLM
# =========================

LLM_MODEL_PATH = Path(
    os.getenv(
        "LLM_MODEL_PATH",
        BASE_DIR / "models/Qwen3.5-4B-q5_k_m.gguf"
    )
)

N_CTX = int(
    os.getenv("N_CTX", 4096)
)

N_THREADS = int(
    os.getenv("N_THREADS", 8)
)

N_GPU_LAYERS = int(
    os.getenv("N_GPU_LAYERS", 40)
)

MAX_TOKENS = int(
    os.getenv("MAX_TOKENS", 1024)
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", 0.2)
)

TOP_P = float(
    os.getenv("TOP_P", 0.9)
)

DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", BASE_DIR / "data/documents"))
PROMPT_TEMPLATE = os.getenv(
    "PROMPT_TEMPLATE",
    "qwen3.5-4B"
)
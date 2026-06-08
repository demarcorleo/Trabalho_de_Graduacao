import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()

PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", BASE_DIR / "config/prompts"))
PROMPT_VERSION  =os.getenv("PROMPT_VERSION", "v1")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
MODEL_NAME = os.getenv("MODEL_NAME", "models/Qwen3.5-4B-q5_k_m.gguf")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
INDEX_PATH = Path(os.getenv("INDEX_PATH", BASE_DIR / "data/indexes/index.faiss"))
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", BASE_DIR / "data/indexes/chunks.pkl"))
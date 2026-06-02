import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()

PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", BASE_DIR / "config/prompts"))
PROMPT_VERSION  =os.getenv("PROMPT_VERSION", "v1")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
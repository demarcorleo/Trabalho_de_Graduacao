import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()

PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", BASE_DIR / "config/prompts"))
PROMPT_VERSION  =os.getenv("PROMPT_VERSION", "v1")

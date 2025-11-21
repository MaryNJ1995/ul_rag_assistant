import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present
load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gen_model: str = os.getenv("GEN_MODEL", "gpt-4o")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    index_path: str = os.getenv("INDEX_PATH", "/home/maryam_najafi/ul_bot/ul_rag_assistant/storage/index/ul_index.pkl")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    eval_path: str = os.getenv("EVAL_PATH", "/home/maryam_najafi/ul_bot/ul_rag_assistant/data/eval/ul_eval.jsonl")

def get_settings() -> Settings:
    settings = Settings()
    idx_path = Path(settings.index_path)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    return settings

# ul_rag/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present
load_dotenv()


def _project_root() -> Path:
    # ul_rag/config.py -> ul_rag -> project root
    return Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # LLM / retrieval models
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gen_model: str = os.getenv("GEN_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Project directories
    project_root: Path = _project_root()
    data_dir: Path = _project_root() / "data"
    storage_dir: Path = _project_root() / "storage"
    log_dir: Path = _project_root() / "logs"

    # Index / eval paths (override via env if needed)
    index_path: Path = Path(os.getenv(
        "INDEX_PATH",
        str(_project_root() / "storage" / "index" / "ul_index.pkl"),
    ))
    eval_path: Path = Path(os.getenv(
        "EVAL_PATH",
        str(_project_root() / "data" / "eval" / "ul_eval.jsonl"),
    ))

    # Misc
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def ensure_dirs(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings

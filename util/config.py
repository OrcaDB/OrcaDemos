import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from dotenv import load_dotenv

BASE_PATH = Path(__file__).parent.parent  # points to the research folder file path

# Load .env file
load_dotenv(BASE_PATH / ".env")


class _Config:
    @property
    def BASE_PATH(self) -> Path:
        return BASE_PATH

    @property
    def CACHE_DIR(self) -> Path:
        return Path(os.getenv("CACHE_DIR", BASE_PATH / ".cache"))

    @property
    def MODEL_CACHE_DIR(self) -> Path:
        return Path(os.getenv("MODEL_CACHE_DIR", self.CACHE_DIR / "models"))

    @property
    def DATASET_CACHE_DIR(self) -> Path:
        return Path(os.getenv("DATASET_CACHE_DIR", self.CACHE_DIR / "datasets"))

    @property
    def SAMPLE_DB_NAME(self) -> str:
        return os.getenv("SAMPLE_DB_NAME", "wiki_sample")

    @property
    def PREFERRED_PYTORCH_DEVICE(self) -> str:
        return os.getenv("PREFERRED_PYTORCH_DEVICE", "cpu")

    @property
    def HF_ACCESS_TOKEN(self) -> str:
        return os.getenv("HF_ACCESS_TOKEN", "")

    @property
    def SAMPLE_DB_BACKUP_NAME(self) -> Optional[str]:
        return os.getenv("SAMPLE_DB_BACKUP_NAME", None)

    @property
    def ORCA_DB_ENDPOINT(self) -> str:
        return os.getenv("ORCA_DB_ENDPOINT", "http://localhost:1583")

    @property
    def ORCADB_API_KEY(self) -> str:
        return os.getenv("ORCADB_API_KEY", "")

    @property
    def ORCADB_SECRET_KEY(self) -> str:
        return os.getenv("ORCADB_SECRET_KEY", "")

    @property
    def PREFERRED_DEVICE(self) -> str:
        device = os.getenv("PREFERRED_DEVICE")
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"


config = _Config()

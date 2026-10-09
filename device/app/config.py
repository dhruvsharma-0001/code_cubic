import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    DEVICE_ID: str = "device_a"
    DEVICE_PORT: int = 8001
    SITE_ID: str = "plant_north"
    HUB_URL: str = "http://127.0.0.1:8765"
    DATA_DIR: str = "data"
    SHARD_DIR: Optional[str] = None
    DENSE_MODEL: str = "BAAI/bge-small-en-v1.5"
    MODEL_CACHE_DIR: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Backwards compatibility for existing imports
DEVICE_ID = settings.DEVICE_ID
DEVICE_PORT = settings.DEVICE_PORT
SITE_ID = settings.SITE_ID
HUB_URL = settings.HUB_URL
DATA_DIR = settings.DATA_DIR
SHARD_DIR = settings.SHARD_DIR or os.path.join(DATA_DIR, f"shard_{DEVICE_ID}")
DENSE_MODEL = settings.DENSE_MODEL
MODEL_CACHE_DIR = settings.MODEL_CACHE_DIR or os.path.join(DATA_DIR, "models")

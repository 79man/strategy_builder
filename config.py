# config.py
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    STORAGE_PATH: str = "./strategy_data"
    DATA_FEED_BASE_URL: str = "http://localhost:8000"
    CALLBACK_BASE_URL: str = "http://localhost:8000/callback"

    class Config:
        env_file = ".env"   # load from .env automatically

# create a singleton config instance
config = Config()

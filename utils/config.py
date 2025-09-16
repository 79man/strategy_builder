# config.py
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    STORAGE_PATH: str = "./strategy_data"
    DATA_FEED_BASE_URL: str = "http://localhost:9000" # URL of the data feed service
    CALLBACK_BASE_URL: str = "http://localhost:8000" # BaseURL use for candle feed callback to strategy builder
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    RELOAD:bool = True

    class Config:
        env_file = ".env"   # load from .env automatically


# create a singleton config instance
config = Config()

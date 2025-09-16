from utils.config import config

def get_callback_url(ticker:str, interval:str) -> str:
    return f"{config.CALLBACK_BASE_URL}/strategies/{ticker}/{interval}/candle"
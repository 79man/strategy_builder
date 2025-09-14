from pydantic import BaseModel
from typing import Optional, Dict, Any, Union

# -------------------------------
# Pydantic models
# -------------------------------


class StrategyCreateRequest(BaseModel):
    strategy: str
    ticker: str
    interval: str
    restart: bool = False
    params: Optional[Dict[str, Any]] = None


class CandleRequest(BaseModel):
    Datetime: str
    Open: float
    High: float
    Low: float
    Close: float


class SignalResponse(BaseModel):
    ticker: str
    interval: str
    strategy: str
    datetime: Optional[str] = None
    signal: Optional[str] = None  # BUY, SELL, HOLD
    indicators: Optional[Dict[str, Union[float, bool, str]]] = None
    message: Optional[str] = None

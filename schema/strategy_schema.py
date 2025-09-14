from pydantic import BaseModel
from typing import Optional, Dict, Any, Union, List

# -------------------------------
# Pydantic models
# -------------------------------


class CreateStrategyRequest(BaseModel):
    strategy_name: str
    params: Optional[Dict[str, Any]] = None
    tickers: Dict[str, List[str]]  # e.g. { "AAPL": ["1h", "5m"], "MSFT": ["15m"] }


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

class StrategyDeleteRequest(BaseModel):
    strategy_name: str
    params: dict  # must match how the instance was created
    ticker: Optional[str] = None
    interval: Optional[str] = None
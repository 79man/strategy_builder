from pydantic import BaseModel
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
# -------------------------------
# Pydantic models
# -------------------------------


class CreateStrategyRequest(BaseModel):
    strategy_name: str
    params: Optional[Dict[str, Any]] = None
    tickers: Dict[str, List[str]]  # e.g. { "AAPL": ["1h", "5m"], "MSFT": ["15m"] }


class CandleRequest(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float

class StrategyDeleteRequest(BaseModel):
    strategy_name: str
    params: dict  # must match how the instance was created
    ticker: Optional[str] = None
    interval: Optional[str] = None

class StrategiesPauseRequest(BaseModel):
    strategy_ids: List[str] # List of Strategy IDs

class StrategiesResumeRequest(BaseModel):
    strategy_ids: List[str] # List of Strategy IDs

class StrategyWithStatus(BaseModel):
    strategy_id: str
    status : str

class StrategiesPauseResponse(BaseModel):
    detail: Optional[Union[List[StrategyWithStatus], str]] = "-" # List of Strategy IDs with reumption status
    message: str
    
class StrategiesResumeResponse(StrategiesPauseResponse):
    pass

## Remote Data Service related
class DataSubscriptionRequest(BaseModel):  
    start_date_time: datetime  
    end_date_time: Optional[datetime] = None  
    interval: str  
    ticker: str  
    callback_url: Optional[str] = None 

class SignalResponse(BaseModel):
    ticker: str
    interval: str
    strategy_id: str
    datetime: Optional[str] = None
    signal: Optional[str] = None  # BUY, SELL, HOLD
    indicators: Optional[Dict[str, Union[float, bool, str]]] = None
    message: Optional[str] = None

class HistoricalDataResponse(BaseModel):
    message: str
    historical_candles: List[Dict[str, Any]]
    candle_count: int
class DataSubscriptionDetails(BaseModel):
    start_date_time: Optional[datetime]
    end_date_time: Optional[datetime]
    interval: str
    ticker: str
    callback_url: Optional[str] = None

class DataSubscriptionResponse(BaseModel):
    message: str
    subscription_key: Optional[str] = None
    subscription: Optional[DataSubscriptionDetails] = None
    historical_data: Optional[HistoricalDataResponse] = None

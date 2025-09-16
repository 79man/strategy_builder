from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
from typing import Optional
from utils.config import config


class DataSourceError(Exception):
    """Base exception for data source errors"""
    pass


class DataSource(ABC):
    @abstractmethod
    def fetch_data(
        self, ticker: str,
        interval: str,
        start_datetime: Optional[datetime] = None,
        callback_url: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLC dataframe for given ticker/interval

        Returns:
            DataFrame with DatetimeIndex and columns: ['Open', 'High', 'Low', 'Close']

        Raises:
            DataSourceError: For data retrieval failures
            ValueError: For invalid parameters
        """
        pass

    def validate_params(self, ticker: str, interval: str) -> None:
        """Validate input parameters"""
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        if not interval or not isinstance(interval, str):
            raise ValueError("Interval must be a non-empty string")

    def supports_callback(self) -> bool:
        """Return True if this data source supports callback URLs for live data"""
        return False

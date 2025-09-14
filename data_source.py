from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
from typing import Optional


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
        If start_date is specified, historical candles (if any) will be fetched
        if callback_url is specified, then feed of candles expected in callback 
        """
        pass

import yfinance as yf
import requests
from datetime import datetime, timezone
from data_source import DataSource
import pandas as pd
from typing import Optional


class YFinanceSource(DataSource):
    def fetch_data(
        self,
        ticker: str,
        interval: str,
        start_datetime: Optional[datetime] = None,
        callback_url: Optional[str] = None
    ) -> pd.DataFrame:
        # does not support subscription

        df = yf.download(
            ticker,
            interval=interval,
            start=start_datetime,
            end=datetime.now(timezone.utc)
        )
        return df[['Open', 'High', 'Low', 'Close']].dropna() if df is not None else pd.DataFrame()

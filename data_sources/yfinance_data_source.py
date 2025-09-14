import yfinance as yf
import requests
from datetime import datetime, timezone
from data_source import DataSource
import pandas as pd


class YFinanceSource(DataSource):
    def fetch_data(
        self, 
        ticker: str, 
        interval: str, 
        start_datetime: datetime, 
        callback_url: str|None = None
    ) -> pd.DataFrame:
        # does not support subscription

        df = yf.download(
            ticker,
            interval=interval,
            start=start_datetime,
            end=datetime.now(timezone.utc)
        )
        return df[['Open', 'High', 'Low', 'Close']].dropna()


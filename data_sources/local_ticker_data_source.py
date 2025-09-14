# import yfinance as yf
import requests
from datetime import datetime, timezone
from data_source import DataSource
import pandas as pd
from config import config
from typing import Optional

DATA_FEED_BASE_URL = config.DATA_FEED_BASE_URL
CALLBACK_BASE_URL = config.CALLBACK_BASE_URL


class LocalTickerDataSource(DataSource):
    def fetch_data(
        self, 
        ticker: str, 
        interval: str, 
        start_datetime: Optional[datetime] = None, 
        callback_url: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from the external data feed service.
        If callback_url is provided, it can also be registered for live updates.
        """

         # Prepare request
        params = {
            "ticker": ticker,
            "interval": interval,
            "start": start_datetime.isoformat() if start_datetime else None,
            "end": datetime.now(timezone.utc).isoformat(),
        }
        if callback_url:
            params["callback_url"] = callback_url

        url = f"{DATA_FEED_BASE_URL}/candles"
        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            raise RuntimeError(
                f"❌ Failed to fetch data from {url}: {response.status_code} {response.text}"
            )

        data = response.json()
        if not data:
            raise ValueError("No candle data returned from feed service")

        # Convert JSON to DataFrame
        df = pd.DataFrame(data)
        # Expecting feed service returns columns: datetime, open, high, low, close
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)

        return df[["Open", "High", "Low", "Close"]].dropna()


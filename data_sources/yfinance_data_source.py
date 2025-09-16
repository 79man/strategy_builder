import yfinance as yf
import requests
from datetime import datetime, timezone
from data_source import DataSource, DataSourceError
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class YFinanceSource(DataSource):
    def supports_callback(self) -> bool:
        return False

    def fetch_data(
        self,
        ticker: str,
        interval: str,
        start_datetime: Optional[datetime] = None,
        callback_url: Optional[str] = None
    ) -> pd.DataFrame:
        # does not support subscription

        if callback_url:
            logger.warning(
                "YFinanceSource does not support callback_url parameter")

        try:
            df = yf.download(
                ticker,
                interval=interval,
                start=start_datetime,
                end=datetime.now(timezone.utc),
                progress=False  # Suppress progress bar
            )

            if df is None or df.empty:
                logger.warning(
                    f"No data returned from Yahoo Finance for {ticker} {interval}")
                return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

            return df[['Open', 'High', 'Low', 'Close']].dropna()

        except Exception as e:
            logger.error(f"YFinance fetch failed for {ticker}: {e}")
            raise DataSourceError(
                f"Failed to fetch data from Yahoo Finance: {e}")

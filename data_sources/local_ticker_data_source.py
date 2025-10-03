import requests
from datetime import datetime, timezone
from data_source import DataSource, DataSourceError
import pandas as pd
from utils.config import config
from typing import Optional
import logging
from schema import strategy_schema
from pydantic import HttpUrl

logger = logging.getLogger(__name__)


class LocalTickerDataSource(DataSource):
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        self.max_retries = max_retries
        self.timeout = timeout

    def supports_callback(self) -> bool:
        return True

    def fetch_data(
        self,
        ticker: str,
        interval: str,
        start_datetime: Optional[datetime] = None,
        callback_url: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from the external data feed service.
        Columns names will be mapped to expected names for startegy builder
        incoming Df will be sorted by datetime ascending
        If callback_url is provided, it can also be registered for live updates.
        """

        self.validate_params(ticker, interval)

        # Prepare subscription request payload
        subscription_data = strategy_schema.DataSubscriptionRequest(
            start_date_time=start_datetime or datetime.now(timezone.utc),
            end_date_time=datetime.now(timezone.utc),
            interval=interval,
            ticker=ticker,
            callback_url=callback_url
        )

        url = f"{config.DATA_FEED_BASE_URL}/subscribe"

        logger.info(
            f"Sending subscription request:{url}\n{subscription_data} ")

        # Retry logic for network failures
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=subscription_data.model_dump(mode="json"),
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code != 200:
                    raise DataSourceError(
                        f"Failed to fetch data from {url}: {response.status_code} {response.text}"
                    )

                # Parse the new response schema
                response_data = strategy_schema.DataSubscriptionResponse(
                    **response.json())

                # Log subscription details
                if response_data.subscription_key:
                    logger.info(
                        f"Subscription created with key: {response_data.subscription_key}")

                # Extract historical candles from nested structure
                if not response_data.historical_data or not response_data.historical_data.historical_candles:
                    logger.warning(
                        f"No historical candles returned for {ticker} {interval}")
                    return pd.DataFrame(columns=["datetime, Open", "High", "Low", "Close"])

                candles_data = response_data.historical_data.historical_candles
                logger.info(
                    f"Received {response_data.historical_data.candle_count} historical candles")

                # logger.info(candles_data)
                # Convert to DataFrame
                df = pd.DataFrame(candles_data)

                # Handle both lowercase and uppercase column names
                column_mapping = {}
                for col in df.columns:
                    if col.lower() == 'datetime':
                        column_mapping[col] = 'datetime'
                    elif col.lower() == 'timestamp':
                        column_mapping[col] = 'datetime'
                    elif col.lower() == 'open':
                        column_mapping[col] = 'Open'
                    elif col.lower() == 'high':
                        column_mapping[col] = 'High'
                    elif col.lower() == 'low':
                        column_mapping[col] = 'Low'
                    elif col.lower() == 'close':
                        column_mapping[col] = 'Close'

                df = df.rename(columns=column_mapping)

                if 'datetime' not in df.columns:
                    raise DataSourceError(
                        "Response missing required 'datetime' column")

                df["datetime"] = pd.to_datetime(
                    df["datetime"], utc=True, format="mixed")
                df.set_index("datetime", inplace=True)

                # Sort by datetime ascending
                df.sort_index(inplace=True)

                return df

            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise DataSourceError(
                        f"Failed to fetch data after {self.max_retries} attempts: {e}")

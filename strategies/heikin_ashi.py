import pandas as pd
from datetime import datetime, timezone
from strategy_base import Strategy
from data_sources.yfinance_data_source import YFinanceSource
from data_sources.local_ticker_data_source import LocalTickerDataSource
from typing import Optional, Dict, List
from data_source import DataSource
from utils.intervals import get_interval_timedelta
from utils.callback_url import get_callback_url
import logging
logger = logging.getLogger(__name__)


class HeikinAshiStrategy(Strategy):
    strategy_name = "HeikinAshiStrategy"

    @classmethod
    def validate_creation_request(
        cls,
        tickers: Dict[str, List[str]],
        params: Optional[dict] = None
    ) -> None:
        # Call parent validation first
        super().validate_creation_request(tickers, params)

        # HeikinAshi-specific parameter validation
        if params:
            fastma = params.get("fastma", 5)
            slowma = params.get("slowma", 30)
            if fastma >= slowma:
                raise ValueError("fastma must be less than slowma")
            if fastma < 1 or slowma < 1:
                raise ValueError("Moving average periods must be positive")

    def __init__(
        self,
        params: Optional[dict] = None,
        storage_path: Optional[str] = None,
        data_source: Optional[DataSource] = None
    ):
        super().__init__(params, storage_path)
        self.data_source = data_source or LocalTickerDataSource()  # or YFinanceSource()

    def initialize_ticker(
        self,
        ticker: str, interval: str,
        params: dict,
        start_datetime: Optional[datetime] = None
    ) -> None:
        """
        Load initial candles and compute Heikin-Ashi values.
        This should populate _signals[(ticker, interval)].
        """
        # --- 1. Minimum required candles ---
        slowma = params.get("slowma", 30)
        fastma = params.get("fastma", 5)
        lookback = params.get("lookback", 100)

        # ✅ determine how many candles we need
        min_candles = lookback + max(slowma, fastma)

        # ✅ calculate start_datetime based on interval + min_candles
        delta = get_interval_timedelta(interval) * min_candles

        if not start_datetime:
            start_datetime = datetime.now(timezone.utc) - delta

        logger.info(
            f"Fetching data from {start_datetime}, {min_candles}, {delta}, {interval}")

        # ✅ fetch historical candles
        df = self.data_source.fetch_data(
            ticker=ticker,
            interval=interval, start_datetime=start_datetime,
            callback_url=get_callback_url(ticker=ticker, interval=interval)
        )

        logger.info(f"Fetched {df.shape} \n {df}")

        if df.empty:
            raise ValueError(f"No data returned for {ticker} {interval}")

        self._signals[(ticker, interval)
                      ] = self.calculate_indicators(df, params)
        self.running = True
        self.save_to_disk(ticker, interval)

        print(
            f"✅ Initialization complete from {start_datetime}, got {len(df)} candles.")

    def calculate_indicators(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        ha_df = self.get_heikin_ashi(df)
        fma = ha_df['HA_Close'].ewm(
            span=params.get("fastma", 5), adjust=False).mean()
        sma = ha_df['HA_Close'].ewm(span=params.get(
            "slowma", 30), adjust=False).mean()

        signals = pd.DataFrame(index=ha_df.index)
        signals['HA_Close'] = ha_df['HA_Close']
        signals['FMA'] = fma
        signals['SMA'] = sma
        signals['GoLong'] = (fma > sma) & (fma.shift(1) <= sma.shift(1))
        signals['GoShort'] = (fma < sma) & (fma.shift(1) >= sma.shift(1))
        return signals

    def _on_new_candle(self, ticker: str, interval: str, ohlc: dict) -> dict:
        """
        Process a new OHLC candle and compute HA + signals.
        """
        df = self._signals.get((ticker, interval))
        if df is None:
            return {"error": f"{ticker}-{interval} not initialized"}

        dt = ohlc.get("Datetime") or ohlc.get("datetime") or datetime.now()
        ts = pd.to_datetime(dt)
        open = ohlc.get("Open") or ohlc.get("open")
        high = ohlc.get("High") or ohlc.get("high")
        low = ohlc.get("Low") or ohlc.get("low")
        close = ohlc.get("Close") or ohlc.get("close")

        if any(v is None for v in [open, high, low, close]):
            return {"error": "Missing required OHLC data"}

        # Compute HA values
        ha_close = (open + high + low + close) / 4
        if len(df) == 0:
            ha_open = (open + close) / 2
        else:
            ha_open = (df.iloc[-1]["HA_Open"] + df.iloc[-1]["HA_Close"]) / 2

        ha_high = max(high, ha_open, ha_close)
        ha_low = min(low, ha_open, ha_close)

        go_long = ha_close > ha_open
        go_short = ha_close < ha_open

        df.loc[ts] = {
            "Open": open,
            "High": high,
            "Low": low,
            "Close": close,
            "HA_Open": ha_open,
            "HA_Close": ha_close,
            "HA_High": ha_high,
            "HA_Low": ha_low,
            "GoLong": go_long,
            "GoShort": go_short,
        }

        self._signals[(ticker, interval)] = df

        return {
            "datetime": str(ts),
            "GoLong": go_long,
            "GoShort": go_short,
            "HA_Open": ha_open,
            "HA_Close": ha_close,
            "HA_High": ha_high,
            "HA_Low": ha_low,
        }

    @staticmethod
    def get_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
        ha_df = df.copy()

        # Ensure numeric dtypes
        ha_df[['Open', 'High', 'Low', 'Close']] = ha_df[[
            'Open', 'High', 'Low', 'Close']].astype(float)

        # Heikin-Ashi Close
        ha_df['HA_Close'] = (ha_df['Open'] + ha_df['High'] +
                             ha_df['Low'] + ha_df['Close']) / 4

        # Heikin-Ashi Open (vectorized: recursive avg of prev HA values)
        ha_open = pd.Series(index=ha_df.index, dtype=float)
        ha_open.iloc[0] = (ha_df['Open'].iloc[0] + ha_df['Close'].iloc[0]) / 2
        for i in range(1, len(ha_df)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] +
                               ha_df['HA_Close'].iloc[i - 1]) / 2
        ha_df['HA_Open'] = ha_open

        # HA High / Low
        ha_df['HA_High'] = ha_df[['High', 'HA_Open',
                                  'HA_Close']].astype(float).max(axis=1)
        ha_df['HA_Low'] = ha_df[['Low', 'HA_Open',
                                 'HA_Close']].astype(float).min(axis=1)

        # Normalize index (remove timezone)
        ha_df.index = ha_df.index.tz_localize(None)

        return ha_df[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]

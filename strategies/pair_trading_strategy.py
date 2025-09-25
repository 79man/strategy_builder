import pandas as pd
from datetime import datetime, timezone
from strategy_base import Strategy
from typing import Optional, List, Dict
from data_sources.yfinance_data_source import YFinanceSource
from data_sources.local_ticker_data_source import LocalTickerDataSource
from data_source import DataSource
from utils.intervals import get_interval_timedelta
from utils.config import config
from utils.callback_url import get_callback_url
from utils import helpers
import logging

logger = logging.getLogger(__name__)


class PairTradingStrategy(Strategy):
    strategy_name = "PairTradingStrategy"

    def __init__(
        self, params: Optional[dict] = None,
        storage_path: Optional[str] = None,
        data_source: Optional[DataSource] = None
    ):
        super().__init__(params, storage_path)
        self.data_source = data_source or LocalTickerDataSource()  # or YFinanceSource()

    @classmethod
    def validate_creation_request(
        cls,
        tickers: Dict[str, List[str]],
        params: Optional[dict] = None
    ) -> None:
        # Call parent validation first
        super().validate_creation_request(tickers, params)

        # PairTrading-specific validation
        total_pairs = sum(len(intervals) for intervals in tickers.values())
        if total_pairs != 2:
            raise ValueError(
                "PairTrading strategy requires exactly 2 ticker-interval pairs")

    def initialize_ticker(
        self,
        ticker: str, interval: str,
        params: dict,
        start_datetime: Optional[datetime] = None
    ) -> None:
        """
        For pair trading, each ticker is tracked separately.
        Later, signals depend on relative movement.
        """
        # --- 1. Minimum required candles ---
        # For pair trading, we need enough history to establish baseline spread statistics
        lookback = params.get("lookback", 100)  # Default lookback period
        # Window for spread calculations
        spread_window = params.get("spread_window", 20)

        # ✅ determine how many candles we need
        # For pair trading, we need lookback + spread calculation window
        min_candles = lookback + spread_window

        delta = get_interval_timedelta(interval) * min_candles

        if not start_datetime:
            start_datetime = datetime.now(timezone.utc) - delta

        # ✅ fetch historical candles
        df = self.data_source.fetch_data(
            ticker=ticker,
            interval=interval, start_datetime=start_datetime,
            callback_url=get_callback_url(ticker=ticker, interval=interval)
        )

        if df.empty:
            raise ValueError(f"No data returned for {ticker} {interval}")

        # Spread & signal flags
        df["Spread"] = pd.Series(dtype="float")
        df["GoLong"] = False
        df["GoShort"] = False

        self._tickers[(ticker, interval)] = df
        self.save_to_disk(ticker, interval)

    def _on_new_candle(self, ticker: str, interval: str, ohlc: dict) -> dict:
        """
        Ingest a new candle for one ticker.
        Signals are computed only when BOTH tickers have aligned timestamps.
        """
        df = self._tickers.get((ticker, interval))
        if df is None:
            return {"error": f"{ticker}-{interval} not initialized"}

        # Standardize timestamp and OHLC data
        ts = pd.to_datetime(ohlc.get("datetime") or ohlc.get(
            "Datetime") or datetime.now())
        open = ohlc.get("open") or ohlc.get("Open")
        high = ohlc.get("high") or ohlc.get("High")
        low = ohlc.get("low") or ohlc.get("Low")
        close = ohlc.get("close") or ohlc.get("Close")

        open, high, low, close = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]

        if any(v is None for v in [open, high, low, close]):
            return {"error": "Missing required OHLC data"}

        df.loc[ts] = {
            "Open": open,
            "High": high,
            "Low": low,
            "Close": close,
            "Spread": 0,
            "GoLong": False,
            "GoShort": False
        }
        self._tickers[(ticker, interval)] = df

        # logger.info(f"df: {helpers.get_df_info(df)}")

        # Identify the other ticker (2-ticker pair)
        tickers = {t for (t, i) in self._tickers if i == interval}
        if len(tickers) != 2:
            return {"warning": "Pair not complete yet"}

        other_tickers = tickers - {ticker}
        if not other_tickers:
            return {"error": "Could not identify pair ticker"}

        other = next(iter(other_tickers))  # Safe extraction
        df_other = self._tickers.get((other, interval))

        if df_other is None or df_other.empty:
            return {"error": f"Other ticker {other} not found or empty"}
        
        # logger.info(f"df_other::: {helpers.get_df_info(df_other)}")

        # Handle timing differences with tolerance
        if ts not in df_other.index:
            # Ensure df_other.index is timezone-aware and matches ts timezone
            other_index = df_other.index
            # logger.info(other_index)
            # if other_index.tz is None:
            #     other_index = other_index.tz_localize('UTC')
            # elif other_index.tz != ts.tz:
            #     other_index = other_index.tz_convert(ts.tz)

            # Look for close timestamps within 1 minute tolerance
            tolerance = pd.Timedelta(minutes=1)

            time_diffs = abs(other_index - ts)
            close_mask = time_diffs <= tolerance
            close_times = other_index[close_mask]

            if close_times.empty:
                return {"info": f"Waiting for {other} candle near {ts}"}

            # Use closest timestamp
            closest_idx = time_diffs[close_mask].argmin()
            closest_ts = close_times[closest_idx]
            other_close = df_other.loc[closest_ts]["Close"]
        else:
            other_close = df_other.loc[ts]["Close"]

        # Compute spread and signals
        spread = close - other_close
        threshold = self.params.get("spread_threshold", 1.0)

        go_long = spread < -threshold
        go_short = spread > threshold

        # Update both frames safely
        self._tickers[(ticker, interval)].at[ts, "Spread"] = spread
        self._tickers[(ticker, interval)].at[ts, "GoLong"] = go_long
        self._tickers[(ticker, interval)].at[ts, "GoShort"] = go_short

        # Update other ticker's frame if timestamp exists
        if ts in self._tickers[(other, interval)].index:
            self._tickers[(other, interval)].at[ts, "Spread"] = spread
            self._tickers[(other, interval)].at[ts, "GoLong"] = go_long
            self._tickers[(other, interval)].at[ts, "GoShort"] = go_short

        return {
            "datetime": str(ts),
            "Spread": spread,
            "GoLong": go_long,
            "GoShort": go_short,
        }

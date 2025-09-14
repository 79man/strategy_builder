import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from strategy_base import Strategy, STORAGE_PATH
from data_sources.yfinance_data_source import YFinanceSource
from data_sources.local_ticker_data_source import LocalTickerDataSource

import logging
logger = logging.getLogger(__name__)


class HeikinAshiStrategy(Strategy):
    strategy_name = "HeikinAshiStrategy"

    def __init__(
        self, ticker: str, interval: str = "1h", params: dict = {},
        storage_path: str = STORAGE_PATH
    ):
        super().__init__(ticker, interval, params, storage_path)
        self.data_source = YFinanceSource()

    def initialize(self, lookback: int = 200):

        # ToDo:
        # Get initial data
        # Calc Minimum candles needed based on the params.
        # POST /subscribe
        # Subscribe for live data
        # {
        #     "start date_time"
        #     "interval"
        #     "ticker"
        #     "callback_url"
        # }

        # --- 1. Minimum required candles ---
        slowma = self.params.get("slowma", 30)
        fastma = self.params.get("fastma", 5)
        lookback = self.params.get("lookback", 100)

        # ✅ determine how many candles we need
        min_candles = lookback + max(slowma, fastma)

        # ✅ calculate start_datetime based on interval + min_candles
        interval_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
        }
        if self.interval not in interval_map:
            raise ValueError(f"Unsupported interval: {self.interval}")

        delta = interval_map[self.interval] * min_candles
        start_datetime = datetime.now(timezone.utc) - delta

        logger.info(f"Fetching data from {start_datetime}, {min_candles}, {delta}, {self.interval}")

        # ✅ fetch historical candles
        self.df = self.data_source.fetch_data(
            self.ticker, self.interval, start_datetime)
        
        logger.info(f"Fetched {self.df.shape} \n {self.df}")
        # if len(self.df) < min_candles:
        #     raise ValueError(
        #         f"Not enough data: got {len(self.df)}, need at least {min_candles}"
        #     )

        # self.df = self.df.tail(
        #     lookback + max(self.params.get("slowma", 30),
        #                    self.params.get("fastma", 5))
        # )
        self.calculate_indicators()
        self.save_to_disk()

        print(f"✅ Initialization complete from {start_datetime}, got {len(self.df)} candles.")


    def calculate_indicators(self):
        ha_df = self.get_heikin_ashi(self.df)
        fma = ha_df['HA_Close'].ewm(
            span=self.params.get("fastma", 5), adjust=False).mean()
        sma = ha_df['HA_Close'].ewm(span=self.params.get(
            "slowma", 30), adjust=False).mean()

        signals = pd.DataFrame(index=ha_df.index)
        signals['HA_Close'] = ha_df['HA_Close']
        signals['FMA'] = fma
        signals['SMA'] = sma
        signals['GoLong'] = (fma > sma) & (fma.shift(1) <= sma.shift(1))
        signals['GoShort'] = (fma < sma) & (fma.shift(1) >= sma.shift(1))
        self.signals = signals

    def _on_new_candle(self, ohlc: dict) -> dict:
        ts = pd.to_datetime(ohlc["Datetime"])
        self.df.loc[ts] = [ohlc["Open"],
                           ohlc["High"], ohlc["Low"], ohlc["Close"]]
        self.calculate_indicators()

        # Return last row indicators only; summary signal handled by base
        last_row = self.signals.iloc[-1].to_dict()
        return {"datetime": ts.isoformat(), "indicators": last_row}

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

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from strategy_base import Strategy
from data_sources.yfinance_data_source import YFinanceSource
from data_sources.local_ticker_data_source import LocalTickerDataSource
from typing import Optional, Dict, List, Any, Tuple
from data_source import DataSource
from utils.intervals import get_interval_timedelta, get_interval_freq
from utils.callback_url import get_callback_url
from utils import helpers
import logging

logger = logging.getLogger(__name__)


class HeikinAshiStrategy(Strategy):
    '''
    The strategy uses two distinct data streams for its calculations:
        1. Heikin Ashi candle data. ha_close is the closing price of the Heikin Ashi candle on the ha_timeframe (default '1D'),
           and applies an optional ha_shift.
        2. Moving Average candle data. mha_close is the closing price of the Heikin Ashi candle on the ema_timeframe (default '1W').
           This is used as the source for the faster EMA, effectively creating an EMA based on a different timeframe.

    The core logic is based on a moving average crossover system:
        - `Fast EMA (fma)`: This is an EMA of the mha_close data, which is from the `ema_timeframe`. 
                        The period is set by fast_ema_period (default 30).
        - `Slow EMA (sma)`: This is an EMA of the ha_close data, which is from the `ha_timeframe`. 
                        The period is set by slow_ema_period (default 50).
    Entry Signals:
        - A buy signal (golong) is generated when the Fast EMA crosses over the Slow EMA.
        - A sell signal (goshort) is generated when the Fast EMA crosses under the Slow EMA.

    Strategy Entries:
        - The strategy enters a long position (strategy.long) on a buy signal.
        - The strategy enters a short position (strategy.short) on a sell signal, 
          but only if the show_long_only input is set to false.
          If show_long_only is set to true, a sell signal will not open a short position; 
          instead, it will close any existing long position.
    '''
    strategy_name = "HeikinAshiStrategy"

    def __init__(
        self,
        params: Optional[dict] = None,
        storage_path: Optional[str] = None,
        data_source: Optional[DataSource] = None
    ):
        super().__init__(params=params, storage_path=storage_path)

        self.data_source = data_source or LocalTickerDataSource()  # or YFinanceSource()

        p = params or {}
        self.ha_interval = p.get("ha_interval", "15m")
        self.ema_interval = p.get("ema_interval", "60m")

        self.ha_shift = int(p.get("ha_shift", 1))
        self.ema_shift = int(p.get("ema_shift", 0))

        self.fast_ema_period = int(p.get("fast_ema_period", 1))
        self.slow_ema_period = int(p.get("slow_ema_period", 30))

        self.fast_ema_shift = int(p.get("fast_ema_shift", 1))
        self.slow_ema_shift = int(p.get("slow_ema_shift", 1))

        self.show_long_only = bool(p.get("show_long_only", False))
        self.lookback = int(p.get("lookback", 100))

        # Store canonical params
        self.params.update({
            "ha_interval": self.ha_interval,
            "ema_interval": self.ema_interval,
            "ha_shift": self.ha_shift,
            "ema_shift": self.ema_shift,
            "fast_ema_period": self.fast_ema_period,
            "slow_ema_period": self.slow_ema_period,
            "fast_ema_shift": self.fast_ema_shift,
            "slow_ema_shift": self.slow_ema_shift,
            "show_long_only": self.show_long_only,
            "lookback": self.lookback,
        })

    # ----------------------------
    # Validation
    # ----------------------------
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
            fast_ema_period = int(params.get("fast_ema_period", 1))
            slow_ema_period = int(params.get("slow_ema_period", 30))
            ha_interval = params.get("ha_interval", "15m")
            ema_interval = params.get("ema_interval", "60m")

            if ha_interval == ema_interval:
                raise ValueError(
                    "ha_interval and ema_interval cannot be same")

            if fast_ema_period >= slow_ema_period:
                raise ValueError(
                    "fast_ema_period must be less than slow_ema_period")

            for key in (
                "ha_shift", "ema_shift",
                "fast_ema_shift", "slow_ema_shift",
                "fast_ema_period", "slow_ema_period"
            ):
                if int(params.get(key, 0)) < 0:
                    raise ValueError(f"{key} must be non-negative")

    # -----------------------------------
    # Technical Indicator Calculations
    # -----------------------------------
    def _heikin_ashi(self, df):
        # Calculate HA_Close first, as it's straightforward
        df['HA_Close'] = (df['Open'] + df['High'] +
                          df['Low'] + df['Close']) / 4

        # Calculate HA_Open
        ha_open = pd.Series(0.0, index=df.index)

        # The first HA_Open is a special case
        ha_open.iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2

        # Recursively calculate subsequent HA_Open values
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] +
                               df['HA_Close'].iloc[i-1]) / 2

        # Add the HA_Open series as a new column
        df['HA_Open'] = ha_open

        # Now, with all four HA columns in the DataFrame, calculate HA_High and HA_Low
        df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

        return df

    # ----------------------------
    # Initialization
    # ----------------------------
    def initialize_ticker(
        self,
        ticker: str,
        interval: str,
        params: dict,
        start_datetime: Optional[datetime] = None
    ) -> None:
        """
        Fetch OHLC, convert to Heikin Ashi, and store.
        """

        # ✅ determine how many candles we need
        min_candles = self.lookback + \
            max(self.slow_ema_period, self.fast_ema_period) + \
            max(self.ha_shift, self.ema_shift)

        # ✅ calculate start_datetime based on interval + min_candles
        delta = get_interval_timedelta(interval) * min_candles

        if not start_datetime:
            start_datetime = datetime.now(timezone.utc) - delta

        logger.info(
            f"[{ticker}-{interval}] Fetching {min_candles}({delta}) candles from {start_datetime}"
        )

        # ✅ Step 1: Fetch historical OHLC candles
        ohlc_df = self.data_source.fetch_data(
            ticker=ticker,
            interval=interval,
            start_datetime=start_datetime,
            callback_url=get_callback_url(ticker=ticker, interval=interval)
        )

        if ohlc_df.empty:
            raise ValueError(f"No data returned for {ticker} {interval}")

        # ✅ Step 2: Deduplication Step
        # Drop duplicates based on the index (datetime) to ensure uniqueness
        ohlc_df = ohlc_df[~ohlc_df.index.duplicated(keep='last')]
        # Ensure chronological order after deduplication
        ohlc_df.sort_index(inplace=True)

        # ✅ Step 3: Compute Heikin-Ashi and update the fetched data
        ohlc_df = self._heikin_ashi(ohlc_df)

        # ✅ Step 4: Store the combined DataFrame
        self._tickers[(ticker, interval)] = ohlc_df

        if (ticker, self.ema_interval) in self._tickers and \
                (ticker, self.ha_interval) in self._tickers:
            # Update self._signals if enough data exists
            self._compute_historical_signals(ticker)

        self.save_to_disk()  # Save tickers (and signals) to disk
        logger.info(
            f"[{ticker}-{interval}] Initialization complete, candles={len(ohlc_df)}")

    def _get_aligned_emas(self, ticker: str):
        ha_key = (ticker, self.ha_interval)
        ema_key = (ticker, self.ema_interval)

        if ha_key not in self._tickers or ema_key not in self._tickers:
            return None, None, None

        ha_df = self._tickers[ha_key].copy()
        ema_df = self._tickers[ema_key].copy()

        ha_ema_series = ha_df['HA_Close'].ewm(
            span=self.slow_ema_period, adjust=False).mean()
        ema_ema_series = ema_df['HA_Close'].ewm(
            span=self.fast_ema_period, adjust=False).mean()

        # Determine which interval is shorter and align accordingly
        if get_interval_freq(self.ha_interval) < get_interval_freq(self.ema_interval):
            final_df = ha_df
            final_interval = self.ha_interval
            merged_emax = pd.DataFrame({
                'FastEMA': ema_ema_series.reindex(ha_ema_series.index, method='ffill'),
                'SlowEMA': ha_ema_series
            })
        else:
            final_df = ema_df
            final_interval = self.ema_interval
            merged_emax = pd.DataFrame({
                'FastEMA': ema_ema_series,
                'SlowEMA': ha_ema_series.reindex(ema_ema_series.index, method='ffill')
            })

        # Apply shifts to the aligned EMA data
        merged_emax['FastEMA_shifted'] = merged_emax['FastEMA'].shift(
            self.fast_ema_shift)
        merged_emax['SlowEMA_shifted'] = merged_emax['SlowEMA'].shift(
            self.slow_ema_shift)

        # Compute signals based on the crossover
        merged_emax['GoLong'] = (merged_emax['FastEMA_shifted'] > merged_emax['SlowEMA_shifted']) & \
                                (merged_emax['FastEMA_shifted'].shift(
                                    1) <= merged_emax['SlowEMA_shifted'].shift(1))
        merged_emax['GoShort'] = (merged_emax['FastEMA_shifted'] < merged_emax['SlowEMA_shifted']) & \
            (merged_emax['FastEMA_shifted'].shift(1) >=
             merged_emax['SlowEMA_shifted'].shift(1))

        # Return the EMA data with signals, the final OHLC DataFrame, and the interval
        return merged_emax, final_df, final_interval

    def _compute_historical_signals(self, ticker: str):
        """
        Compute signals for a ticker based on configured HA and EMA intervals.
        """

        # Use the configured intervals
        ha_key = (ticker, self.ha_interval)
        ema_key = (ticker, self.ema_interval)

        # We need both HA and EMA interval data to compute signals
        if ha_key not in self._tickers:
            logger.warning(
                f"Cannot compute signals: missing HA data for {ticker} @ {self.ha_interval}")
            return

        if ema_key not in self._tickers:
            logger.warning(
                f"Cannot compute signals: missing EMA data for {ticker} @ {self.ema_interval}")
            return

        # Use a copy of the HA dataframes for calculation
        ha_df = self._tickers[ha_key].copy()
        ema_df = self._tickers[ema_key].copy()

        # # Calculate the slow EMA on the HA interval data
        # ha_ema_series = ha_df['HA_Close'].ewm(
        #     span=self.slow_ema_period, adjust=False).mean()

        # # Calculate the fast EMA on the EMA interval data
        # ema_ema_series = ema_df['HA_Close'].ewm(
        #     span=self.fast_ema_period, adjust=False).mean()

        fma_series = ema_df['HA_Close'].ewm(
            span=self.fast_ema_period, adjust=False).mean()
        sma_series = ha_df['HA_Close'].ewm(
            span=self.slow_ema_period, adjust=False).mean()

        # Determine which interval is shorter and align accordingly
        if get_interval_freq(self.ha_interval) < get_interval_freq(self.ema_interval):
            # Case 1: ha_interval is shorter (e.g., '1D' < '1W')
            # The final signals DataFrame will be based on the HA interval
            final_df = ha_df.copy()
            final_interval = self.ha_interval
            final_df['FastEMA'] = fma_series.reindex(
                sma_series.index, method='ffill')
            final_df['SlowEMA'] = sma_series
        else:
            # Case 2: ema_interval is shorter (e.g., '1D' < '1W') or equal
            # The final signals DataFrame will be based on the EMA interval
            final_df = ema_df.copy()
            final_interval = self.ema_interval
            final_df['FastEMA'] = fma_series
            final_df['SlowEMA'] = sma_series.reindex(
                fma_series.index, method='ffill')

        # Apply shifts
        fast_ema_shifted = final_df['FastEMA'].shift(self.fast_ema_shift)
        slow_ema_shifted = final_df['SlowEMA'].shift(self.slow_ema_shift)

        # Compute signals based on the crossover
        final_df['GoLong'] = (fast_ema_shifted > slow_ema_shifted) & (
            fast_ema_shifted.shift(1) <= slow_ema_shifted.shift(1))
        final_df['GoShort'] = (fast_ema_shifted < slow_ema_shifted) & (
            fast_ema_shifted.shift(1) >= slow_ema_shifted.shift(1))

        # Compute the final signal column with the loop for state management
        signals_list = []
        in_long_position = False

        for _, row in final_df.iterrows():
            signal = 'HOLD'
            go_long = row['GoLong']
            go_short = row['GoShort']

            # Apply the signal logic based on the state
            if go_long:
                signal = 'BUY'
                in_long_position = True
            elif not self.show_long_only and go_short:
                signal = 'SELL'
                in_long_position = False
            elif self.show_long_only and in_long_position and go_short:
                signal = 'CLOSE'
                in_long_position = False

            signals_list.append(signal)

        final_df['Signal'] = signals_list

        # Create the final DataFrame with all the required columns for correlation
        signals_df = final_df.copy()
        signals_df.index.name = 'datetime'

        signals_df['ticker'] = ticker
        signals_df['interval'] = final_interval
        signals_df = signals_df[[
            'ticker', 'interval', 'Signal', 
            'Open', 'High', 'Low', 'Close', 
            'HA_Open', 'HA_High', 'HA_Low', 'HA_Close',
            'FastEMA', 'SlowEMA', 
            'GoLong', 'GoShort'
        ]]

        # A list of the columns you want to round
        cols_to_round = [
            'Open', 'High', 'Low', 'Close', 
            'HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 
            'FastEMA', 'SlowEMA'
        ]

        # Apply the rounding to the specified columns
        signals_df[cols_to_round] = signals_df[cols_to_round].round(2)

        # Update the master signals DataFrame for the first time or with a full history
        if self._signals is None or self._signals.empty:
            self._signals = signals_df
        else:
            # Use concat and drop duplicates for a clean merge of historical data
            self._signals = pd.concat([self._signals, signals_df])
            self._signals = self._signals[~self._signals.index.duplicated(
                keep='last')]

    # ----------------------------
    # Signal Computation
    # ----------------------------
    def _compute_next_signal(self, ticker: str) -> Dict[str, Any]:
        """
        Compute signals for a ticker based on configured HA and EMA intervals.
        """

        # Use the configured intervals
        ha_key = (ticker, self.ha_interval)
        ema_key = (ticker, self.ema_interval)

        # We need both HA and EMA interval data to compute signals
        if ha_key not in self._tickers or ema_key not in self._tickers:
            logger.warning(
                f"Cannot compute signals: missing data for {ticker} @ {self.ha_interval} or {self.ema_interval}"
            )
            return {}

        ha_df = self._tickers[ha_key]
        ema_df = self._tickers[ema_key]

        # Calculate EMAs on the full DataFrames
        # ha_ema_series = ha_df['HA_Close'].ewm(
        #     span=self.slow_ema_period, adjust=False).mean()
        # ema_ema_series = ema_df['HA_Close'].ewm(
        #     span=self.fast_ema_period, adjust=False).mean()

        fma_series = ema_df['HA_Close'].ewm(
            span=self.fast_ema_period, adjust=False).mean()
        sma_series = ha_df['HA_Close'].ewm(
            span=self.slow_ema_period, adjust=False).mean()

        # Determine which interval is shorter and align accordingly
        if get_interval_freq(self.ha_interval) < get_interval_freq(self.ema_interval):
            # ha_interval is shorter (e.g., 1D < 1W)

            merged_emax = pd.DataFrame({
                'FastEMA': fma_series.reindex(sma_series.index, method='ffill'),
                'SlowEMA': sma_series
            }).dropna()

            final_df = ha_df
            final_interval = self.ha_interval
            
        else:
            merged_emax = pd.DataFrame({
                'FastEMA': fma_series,
                'SlowEMA': sma_series.reindex(fma_series.index, method='ffill')
            }).dropna()

            final_df = ema_df
            final_interval = self.ema_interval

        # Apply shifts
        fast_ema_shifted = merged_emax['FastEMA'].shift(self.fast_ema_shift)
        slow_ema_shifted = merged_emax['SlowEMA'].shift(self.slow_ema_shift)

        # Drop any NaNs that result from the shift
        shifted_df = pd.DataFrame({
            'FastEMA_shifted': fast_ema_shifted,
            'SlowEMA_shifted': slow_ema_shifted
        }).dropna()

        # Get the latest values from the shifted series
        latest_shifted = shifted_df.iloc[-1]
        prev_shifted = shifted_df.iloc[-2] if len(
            shifted_df) > 1 else pd.Series()

        # logger.info(latest_shifted)
        # logger.info(prev_shifted)

        # Compute signals based on the crossover of the SHIFTED EMAs
        go_long = (latest_shifted['FastEMA_shifted'] > latest_shifted['SlowEMA_shifted']) & \
            (prev_shifted['FastEMA_shifted'] <= prev_shifted['SlowEMA_shifted']
             ) if not prev_shifted.empty else False

        go_short = (latest_shifted['FastEMA_shifted'] < latest_shifted['SlowEMA_shifted']) & \
            (prev_shifted['FastEMA_shifted'] >= prev_shifted['SlowEMA_shifted']
             ) if not prev_shifted.empty else False

        # Determine the signal
        signal = 'HOLD'
        if go_long:
            signal = 'BUY'
        elif go_short:
            if not self.show_long_only:
                signal = 'SELL'
            else:
                signal = 'CLOSE'

        # Get the latest OHLC data for the signal row
        latest_candle = final_df.iloc[-1]

        # The fix for the error: use .name to get the timestamp from the Series
        timestamp = latest_candle.name

        # Create the final DataFrame with all the required columns
        new_signal_data = {
            'ticker': ticker,
            'interval': final_interval,
            'Signal': signal,
            'Open': latest_candle['Open'],
            'High': latest_candle['High'],
            'Low': latest_candle['Low'],
            'Close': latest_candle['Close'],
            'HA_Open': latest_candle['HA_Open'],
            'HA_High': latest_candle['HA_High'],
            'HA_Low': latest_candle['HA_Low'],
            'HA_Close': latest_candle['HA_Close'],
            'FastEMA': merged_emax['FastEMA'].iloc[-1],
            'SlowEMA': merged_emax['SlowEMA'].iloc[-1],
            'GoLong': go_long,
            'GoShort': go_short
        }

        new_signal_row = pd.DataFrame([new_signal_data], index=[timestamp])

        # A list of the columns you want to round
        cols_to_round = [
            'Open', 'High', 'Low', 'Close', 
            'HA_Open', 'HA_High', 'HA_Low', 'HA_Close',
            'FastEMA', 'SlowEMA'
        ]

        # Apply the rounding to the specified columns
        new_signal_row[cols_to_round] = new_signal_row[cols_to_round].round(2)

        # Store the signal
        if self._signals is None or self._signals.empty:
            self._signals = new_signal_row
        else:
            # Simplified and more robust way to append/update
            self._signals.loc[new_signal_row.index[0]] = new_signal_row.iloc[0]

        logger.info(
            f"Signals computed for {ticker} @ {new_signal_data['interval']}, signal: {signal}")

        return new_signal_data

    # ----------------------------
    # Live Candle Handling
    # ----------------------------

    def _on_new_candle(self, ticker: str, interval: str, ohlc: dict) -> Dict[str, Any]:
        # Retrieve the existing DataFrame for the ticker and interval
        df = self._tickers.get((ticker, interval), pd.DataFrame())

        ts = pd.to_datetime(ohlc.get("datetime") or ohlc.get(
            "Datetime") or datetime.now(timezone.utc), utc=True)

        new_ohlc_row_data = {
            "Open": ohlc.get("open") or ohlc.get("Open"),
            "High": ohlc.get("high") or ohlc.get("High"),
            "Low": ohlc.get("low") or ohlc.get("Low"),
            "Close": ohlc.get("close") or ohlc.get("Close"),
        }

        # 🔹 The Simplification: Use .loc to assign directly
        df.loc[ts] = new_ohlc_row_data

        # Sort the index to ensure chronological order
        df.sort_index(inplace=True)

        # Check if there's enough history to calculate HA Open
        if len(df) > 1:
            prev_ha_open = df.iloc[-2]['HA_Open']
            prev_ha_close = df.iloc[-2]['HA_Close']
            ha_open = (prev_ha_open + prev_ha_close) / 2
        else:
            # First HA candle, use regular Open/Close
            ha_open = (df.iloc[-1]['Open'] + df.iloc[-1]['Close']) / 2

        ha_close = (df.iloc[-1]['Open'] + df.iloc[-1]['High'] +
                    df.iloc[-1]['Low'] + df.iloc[-1]['Close']) / 4
        ha_high = max(df.iloc[-1]['High'], ha_open, ha_close)
        ha_low = min(df.iloc[-1]['Low'], ha_open, ha_close)

        # 🔹 Update the latest row with the new HA values
        df.loc[ts, 'HA_Open'] = ha_open
        df.loc[ts, 'HA_Close'] = ha_close
        df.loc[ts, 'HA_High'] = ha_high
        df.loc[ts, 'HA_Low'] = ha_low

        # 🔹 Ensure chronological order and remove duplicates one last time, just in case
        # This is also good for a live stream where you may receive the same candle multiple times
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)

        # Store the updated DataFrame in the master tickers dictionary
        self._tickers[(ticker, interval)] = df
        new_signal_data = self._compute_next_signal(ticker)

        self.save_to_disk()  # Save DF and possibly signal
        return new_signal_data

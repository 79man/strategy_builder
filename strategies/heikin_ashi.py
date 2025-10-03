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

from distutils.util import strtobool
from utils import indicators
from schema.strategy_schema import CandleFeedRequest, SignalResponse

import logging
logger = logging.getLogger(__name__)


class HeikinAshiStrategy(Strategy):
    '''
    The strategy uses two distinct data streams for its calculations:
        1. Heikin Ashi candle data. HA_close is the closing price of the Heikin Ashi candle on the ha_timeframe (default '1D'),
           and applies an optional ha_shift.
        2. Moving Average candle data. mHA_close is the closing price of the Heikin Ashi candle on the ema_timeframe (default '1W').
           This is used as the source for the faster EMA, effectively creating an EMA based on a different timeframe.

    The core logic is based on a moving average crossover system:
        - `Fast EMA (fma)`: This is an EMA of the mHA_close data, which is from the `ema_timeframe`. 
                        The period is set by fast_ema_period (default 30).
        - `Slow EMA (sma)`: This is an EMA of the HA_close data, which is from the `ha_timeframe`. 
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

        self.ha_shift = int(p.get("ha_shift", 0))
        self.ema_shift = int(p.get("ema_shift", 0))

        self.fast_ema_period = int(p.get("fast_ema_period", 30))
        self.slow_ema_period = int(p.get("slow_ema_period", 50))

        self.fast_ema_shift = int(p.get("fast_ema_shift", 1))
        self.slow_ema_shift = int(p.get("slow_ema_shift", 1))

        self.show_long_only = bool(
            strtobool(str(p.get("show_long_only", "False")).lower()))

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

        self.heikin_ashi = indicators.HeikinAshiCalculator()

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
    '''
    def _heikin_ashi(self, df):
        # Calculate HA_close first, as it's straightforward
        df['HA_close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

        # Vectorized calculation for HA_open using pandas' accumulate
        # The first HA_open is (Open[0] + Close[0]) / 2, rest: (prev_HA_open + prev_HA_close) / 2
        HA_open = np.zeros(len(df))
        if len(df) > 0:
            HA_open[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
            if len(df) > 1:
                # Use numpy for fast cumulative calculation
                for i in range(1, len(df)):
                    HA_open[i] = (HA_open[i-1] + df['HA_close'].iloc[i-1]) / 2
        df['HA_open'] = HA_open

        # Now, with all four HA columns in the DataFrame, calculate HA_high and HA_low
        df['HA_high'] = df[['High', 'HA_open', 'HA_close']].max(axis=1)
        df['HA_low'] = df[['Low', 'HA_open', 'HA_close']].min(axis=1)

        return df
    '''

    # ----------------------------
    # Initialization
    # ----------------------------
    def initialize_ticker(
        self,
        ticker: str,
        interval: str,
        params: dict,
        start_datetime: Optional[datetime] = None,
        ohlc_df: Optional[pd.DataFrame] = None,
        use_callback: bool = True
    ) -> None:
        """
        Fetch OHLC, convert to Heikin Ashi, and store.
        """

        # if valid ohlc_df is passed, use it directly, else use data source
        if ohlc_df is not None and not ohlc_df.empty:
            pass
        else:
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
            callback_url = None
            if use_callback:
                callback_url = get_callback_url(
                    ticker=ticker, interval=interval)

            ohlc_df = self.data_source.fetch_data(
                ticker=ticker,
                interval=interval,
                start_datetime=start_datetime,
                callback_url=callback_url
            )

        if ohlc_df.empty:
            raise ValueError(f"No data returned for {ticker} {interval}")

        # ✅ Step 2: Deduplication Step
        # Drop duplicates based on the index (datetime) to ensure uniqueness
        ohlc_df = ohlc_df[~ohlc_df.index.duplicated(keep='last')]
        # Ensure chronological order after deduplication
        ohlc_df.sort_index(inplace=True)

        # ✅ Step 3: Compute Heikin-Ashi and update the fetched data
        ohlc_df = self.heikin_ashi.calculate_ha(ohlc_df)

        logger.info(
            f"[{ticker}-{interval}] HA DF: {ohlc_df.columns}"
        )

        # ✅ Step 4: Store the combined DataFrame
        self._tickers[(ticker, interval)] = ohlc_df

        if (ticker, self.ema_interval) in self._tickers and \
                (ticker, self.ha_interval) in self._tickers:
            # If both intervals are initialized, compute signals
            self._compute_signals(ticker)

        # self.save_to_disk()  # Save tickers (and signals) to disk
        logger.info(
            f"[{ticker}-{interval}] Initialization complete, candles={len(ohlc_df)}")

    def _get_data_for_signals(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Helper to retrieve HA and EMA data for a ticker and validate it.
        """
        ha_key = (ticker, self.ha_interval)
        ema_key = (ticker, self.ema_interval)

        ha_df = self._tickers.get(ha_key)
        ema_df = self._tickers.get(ema_key)

        if ha_df is None or ha_df.empty:
            logger.warning(
                f"Missing or empty HA data for {ticker} @ {self.ha_interval}")
            return None, None
        if ema_df is None or ema_df.empty:
            logger.warning(
                f"Missing or empty EMA data for {ticker} @ {self.ema_interval}")
            return None, None

        return ha_df, ema_df

    def _calculate_emas(self, ha_df: pd.DataFrame, ema_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculates EMA series based on the HA_close price.
        """
        fma_series = ema_df['HA_close'].ewm(
            span=self.fast_ema_period, adjust=False).mean()
        sma_series = ha_df['HA_close'].ewm(
            span=self.slow_ema_period, adjust=False).mean()
        return fma_series, sma_series

    def _align_emas_and_calculate_signals(
        self, ha_df: pd.DataFrame, ema_df: pd.DataFrame, fma_series: pd.Series, sma_series: pd.Series
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Aligns EMAs based on interval and computes the crossover signals.
        """
        if get_interval_freq(self.ha_interval) < get_interval_freq(self.ema_interval):
            final_df = ha_df.copy()
            final_interval = self.ha_interval
            final_df['FastEMA'] = fma_series.reindex(
                final_df.index, method='ffill')
            final_df['SlowEMA'] = sma_series
        else:
            final_df = ema_df.copy()
            final_interval = self.ema_interval
            final_df['FastEMA'] = fma_series
            final_df['SlowEMA'] = sma_series.reindex(
                final_df.index, method='ffill')

        final_df['FastEMA_shifted'] = final_df['FastEMA'].shift(
            self.fast_ema_shift)
        final_df['SlowEMA_shifted'] = final_df['SlowEMA'].shift(
            self.slow_ema_shift)

        final_df.dropna(subset=['FastEMA_shifted',
                        'SlowEMA_shifted'], inplace=True)

        if len(final_df) < 2:
            logger.debug(
                "Not enough data after alignment or shifting for crossover logic.")
            return None, None

        final_df['GoLong'] = (final_df['FastEMA_shifted'] > final_df['SlowEMA_shifted']) & \
                             (final_df['FastEMA_shifted'].shift(1) <=
                              final_df['SlowEMA_shifted'].shift(1))
        final_df['GoShort'] = (final_df['FastEMA_shifted'] < final_df['SlowEMA_shifted']) & \
                              (final_df['FastEMA_shifted'].shift(1) >=
                               final_df['SlowEMA_shifted'].shift(1))

        return final_df, final_interval

    def _compute_signals(self, ticker: str):
        """
        Computes signals by recalculating on full history and returning new signals.
        Includes 'Signal' column generation and specific column selection.
        """
        ha_df_full, ema_df_full = self._get_data_for_signals(ticker)
        if ha_df_full is None or ema_df_full is None:
            return None, None

        fma_series_full, sma_series_full = self._calculate_emas(
            ha_df_full, ema_df_full)

        # Calculate Signals on the FULL history to ensure consistent EMA and Signal generation
        final_df, final_interval = self._align_emas_and_calculate_signals(
            ha_df_full, ema_df_full,
            fma_series_full, sma_series_full
        )

        if final_df is None:
            return None, None

        # --- Compute 'Signal' Column ---
        # Initialize with 'HOLD'
        final_df['Signal'] = 'HOLD'

        # Apply 'BUY' for GoLong signals
        final_df.loc[final_df['GoLong'], 'Signal'] = 'BUY'

        if not self.show_long_only:
            # For non-long-only, 'SELL' when GoShort
            final_df.loc[final_df['GoShort'], 'Signal'] = 'SELL'
        else:
            # For long-only, 'CLOSE' when in position and GoShort
            # This uses a vectorized approximation of position tracking:
            # If (total buys so far) > (total sells so far), then currently in a long position
            # Note: This is an approximation and might not perfectly reflect complex real-world position management.
            in_long = final_df['GoLong'].cumsum(
            ) > final_df['GoShort'].cumsum()
            final_df.loc[in_long & final_df['GoShort'], 'Signal'] = 'CLOSE'

        # --- Prepare the final signals DataFrame for storage and return ---
        signals_df = final_df.copy()
        signals_df.index.name = 'datetime'

        signals_df['ticker'] = ticker
        signals_df['interval'] = final_interval

        # Ensure all required columns exist before selecting
        required_cols = [
            'ticker', 'interval', 'Signal',
            'Open', 'High', 'Low', 'Close',
            'HA_open', 'HA_high', 'HA_low', 'HA_close',
            'FastEMA', 'SlowEMA',
            'GoLong', 'GoShort'
        ]

        # Filter signals_df to keep only the columns that actually exist
        # This handles cases where original_df might not have 'Open', 'High', 'Low', 'Close'
        # if using HA-only DFs for EMA intervals, though in your setup they're expected.
        # If 'Open', etc., are not in final_df (which is derived from HA/EMA DFs), this will fail.
        # Ensure your ha_df_full and ema_df_full always contain 'Open', 'High', 'Low', 'Close'.
        available_cols = [
            col for col in required_cols if col in signals_df.columns]
        signals_df = signals_df[available_cols]

        # Determine which signals are new since the last computation
        if ticker in self._signals and not self._signals[ticker].empty:
            last_signal_time = self._signals[ticker].index[-1]
            new_signals = signals_df[signals_df.index >
                                     last_signal_time].copy()
        else:
            # First time calculating signals for this ticker
            new_signals = signals_df.copy()

        # Store the full signal history, overwriting old data with new full recalculation
        self._signals = signals_df

        if new_signals.empty:
            logger.info(f"No new signals generated for {ticker}.")
            return None, None

        logger.info(f"Generated {len(new_signals)} new signals for {ticker}.")
        return new_signals, final_interval

    #################################################
    '''
    def _get_aligned_emas(
        self, ticker: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:

        ha_key = (ticker, self.ha_interval)
        ema_key = (ticker, self.ema_interval)

        if ha_key not in self._tickers or ema_key not in self._tickers:
            return None, None, None

        ha_df = self._tickers[ha_key].copy()
        ema_df = self._tickers[ema_key].copy()

        ha_ema_series = ha_df['HA_close'].ewm(
            span=self.slow_ema_period, adjust=False).mean()
        ema_ema_series = ema_df['HA_close'].ewm(
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

        if ha_df.empty:
            logger.warning(
                f"Cannot compute signals: empty HA data for {ticker} @ {self.ha_interval}")
            return

        if ema_df.empty:
            logger.warning(
                f"Cannot compute signals: empty EMA data for {ticker} @ {self.ema_interval}")
            return

        # Calculate the fast EMA on the EMA interval data
        fma_series = ema_df['HA_close'].ewm(
            span=self.fast_ema_period, adjust=False).mean()

        # Calculate the slow EMA on the HA interval data
        sma_series = ha_df['HA_close'].ewm(
            span=self.slow_ema_period, adjust=False).mean()

        # Determine which interval is shorter and align accordingly
        # If interval is shorter that series will have more rows than the other
        if get_interval_freq(self.ha_interval) < get_interval_freq(self.ema_interval):
            # Case 1: ha_interval is shorter than ema_interval(e.g., '1D' < '1W')
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
        # signals_list = []
        # in_long_position = False

        # for _, row in final_df.iterrows():
        #     signal = 'HOLD'
        #     go_long = row['GoLong']
        #     go_short = row['GoShort']

        #     # Apply the signal logic based on the state
        #     if go_long:
        #         signal = 'BUY'
        #         in_long_position = True
        #     elif not self.show_long_only and go_short:
        #         signal = 'SELL'
        #         in_long_position = False
        #     elif self.show_long_only and in_long_position and go_short:
        #         signal = 'CLOSE'
        #         in_long_position = False

        #     signals_list.append(signal)

        # final_df['Signal'] = signals_list
        
        # Vectorized assignment for stateless signals
        final_df['Signal'] = 'HOLD'
        final_df.loc[final_df['GoLong'], 'Signal'] = 'BUY'
        if not self.show_long_only:
            final_df.loc[final_df['GoShort'], 'Signal'] = 'SELL'
        else:
            # For long-only, 'CLOSE' when in position and GoShort
            # This still requires a stateful approach, but can be handled with pandas' cumsum for most cases
            in_long = final_df['GoLong'].cumsum(
            ) > final_df['GoShort'].cumsum()
            final_df.loc[in_long & final_df['GoShort'], 'Signal'] = 'CLOSE'

        # Create the final DataFrame with all the required columns for correlation
        signals_df = final_df.copy()
        signals_df.index.name = 'datetime'

        signals_df['ticker'] = ticker
        signals_df['interval'] = final_interval
        signals_df = signals_df[[
            'ticker', 'interval', 'Signal',
            'Open', 'High', 'Low', 'Close',
            'HA_open', 'HA_high', 'HA_low', 'HA_close',
            'FastEMA', 'SlowEMA',
            'GoLong', 'GoShort'
        ]]

        # A list of the columns you want to round
        cols_to_round = [
            'Open', 'High', 'Low', 'Close',
            'HA_open', 'HA_high', 'HA_low', 'HA_close',
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
        # ha_ema_series = ha_df['HA_close'].ewm(
        #     span=self.slow_ema_period, adjust=False).mean()
        # ema_ema_series = ema_df['HA_close'].ewm(
        #     span=self.fast_ema_period, adjust=False).mean()

        fma_series = ema_df['HA_close'].ewm(
            span=self.fast_ema_period, adjust=False).mean()
        sma_series = ha_df['HA_close'].ewm(
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
            'HA_open': latest_candle['HA_open'],
            'HA_high': latest_candle['HA_high'],
            'HA_low': latest_candle['HA_low'],
            'HA_close': latest_candle['HA_close'],
            'FastEMA': merged_emax['FastEMA'].iloc[-1],
            'SlowEMA': merged_emax['SlowEMA'].iloc[-1],
            'GoLong': go_long,
            'GoShort': go_short
        }

        new_signal_row = pd.DataFrame([new_signal_data], index=[timestamp])

        # A list of the columns you want to round
        cols_to_round = [
            'Open', 'High', 'Low', 'Close',
            'HA_open', 'HA_high', 'HA_low', 'HA_close',
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
    '''
    # ----------------------------
    # Live Candle Handling
    # ----------------------------

    def _on_new_candles(self, ticker: str, interval: str, new_ohlc_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Handles the arrival of new candles (using a Pydantic List) for a specific
        ticker and interval. Triggers incremental HA calculation and recomputes signals.

        Args:
            ticker (str): The symbol of the asset.
            interval (str): The interval of the incoming candles (e.g., '1D', '1H').
            new_ohlc_df (pd.DataFrame): A Dataframe containing the new candle data.
        """
        if new_ohlc_df.empty:
            logger.error(
                f"Received empty DataFrame of candles for {ticker} @ {interval}.")
            return {'error': f"Received empty DataFrame of candles for {ticker} @ {interval}."}

        # Convert the list of Pydantic models to a DataFrame
        new_ohlc_df.sort_index(inplace=True)

        # Eliminate duplicates in the new data
        new_ohlc_df = new_ohlc_df[~new_ohlc_df.index.duplicated(keep='last')]

        # Update the existing DataFrame for the ticker and interval
        original_df = self._tickers.get((ticker, interval), pd.DataFrame())

        # ✅ Step 3: Compute Heikin-Ashi and update the fetched data
        new_ohlc_df = self.heikin_ashi.incremental_ha_calculation(
            original_df, new_ohlc_df)

        self._tickers[(ticker, interval)] = new_ohlc_df

        # Compute Signals
        new_signals, final_interval = self._compute_signals(ticker)

        if new_signals is not None and not new_signals.empty:
            logger.info(
                f"New signals generated for {ticker} @ {final_interval}: \n{len(new_signals)}")
            return {'new_signals': len(new_signals), 'details': new_signals.to_dict(orient='records')}
        else:
            logger.debug(
                f"No new signals or signals are empty for {ticker} @ {final_interval}.")
            return {'error': f"No new signals or signals are empty for {ticker} @ {final_interval}."}

    '''
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
            prev_HA_open = df.iloc[-2]['HA_open']
            prev_HA_close = df.iloc[-2]['HA_close']
            HA_open = (prev_HA_open + prev_HA_close) / 2
        else:
            # First HA candle, use regular Open/Close
            HA_open = (df.iloc[-1]['Open'] + df.iloc[-1]['Close']) / 2

        HA_close = (df.iloc[-1]['Open'] + df.iloc[-1]['High'] +
                    df.iloc[-1]['Low'] + df.iloc[-1]['Close']) / 4
        HA_high = max(df.iloc[-1]['High'], HA_open, HA_close)
        HA_low = min(df.iloc[-1]['Low'], HA_open, HA_close)

        # 🔹 Update the latest row with the new HA values
        df.loc[ts, 'HA_open'] = HA_open
        df.loc[ts, 'HA_close'] = HA_close
        df.loc[ts, 'HA_high'] = HA_high
        df.loc[ts, 'HA_low'] = HA_low

        # 🔹 Ensure chronological order and remove duplicates one last time, just in case
        # This is also good for a live stream where you may receive the same candle multiple times
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)

        # Store the updated DataFrame in the master tickers dictionary
        self._tickers[(ticker, interval)] = df
        new_signal_data = self._compute_next_signal(ticker)

        # self.save_to_disk()  # Save DF and possibly signal
        return new_signal_data
    '''

    def get_last_signal(self) -> SignalResponse:
        """
        Return the last signal generated by the strategy. 

        _signals.df can either be empty or it has rows of the following form:
        datetime is the index:
        Example:
        datetime,ticker,interval,Signal,Open,High,Low,Close,HA_open,HA_high,HA_low,HA_close,FastEMA,SlowEMA,GoLong,GoShort
            2025-09-19 19:10:00+00:00,BTCUSD,1m,HOLD,115270.0,115277.5,115270.0,115277.5,115273.75,115277.5,115270.0,115273.75,115887.94,115273.75,False,False
            2025-09-19 19:11:00+00:00,BTCUSD,1m,HOLD,115277.5,115292.5,115276.5,115278.5,115273.75,115292.5,115273.75,115281.25,115887.94,115274.04,False,False
            2025-09-19 19:12:00+00:00,BTCUSD,1m,HOLD,115277.5,115278.5,115266.5,115267.5,115277.5,115278.5,115266.5,115272.5,115887.94,115273.98,False,False

        """
        # take the last row from self._signals and return it as a SignalResponse
        if self._signals is None or self._signals.empty:
            return SignalResponse(
                strategy_id="",
                ticker="-",
                interval="-",
                signal="None",
                message="No signals generated yet."
            )

        last_row = self._signals.iloc[-1]
        # Get the index value (datetime) for the last row
        last_index = self._signals.index[-1]

        return SignalResponse(
            strategy_id=self.strategy_id,
            ticker=last_row['ticker'],
            interval=last_row['interval'],
            datetime=str(last_index),
            signal=last_row['Signal'],
            indicators=last_row.to_dict(),
            message=f"Last signal for {last_row['ticker']} @ {last_row['interval']}: {last_row['Signal']}"
        )

    def get_all_signals(
        self,
        offset: int = 0,
        limit: int | None = None,
        type: str | None = None
    ) -> List[SignalResponse]:
        '''
        Return all signals generated by the strategy
        '''
        if self._signals is None or self._signals.empty:
            return [SignalResponse(
                ticker="-",
                interval="-",
                signal="None",
                message="No signals generated yet."
            )]

        results = []
        df = self._signals.copy()

        if type and type in ['BUY', 'SELL', 'HOLD']:
            df = df[df['Signal'] == type]

        sliced = df.iloc[offset: (None if limit is None else offset + limit)]

        for ts, row in sliced.iterrows():
            results.append(SignalResponse(
                # strategy_id=self.strategy_id,
                datetime=str(ts),
                ticker=row['ticker'],
                interval=row['interval'],
                signal=row['Signal'],
                # indicators={
                #     'FastEMA': float(row['FastEMA']) if row.get('FastEMA') is not None else 0.0,
                #     'SlowEMA': float(row['SlowEMA']) if row.get('SlowEMA') is not None else 0.0,
                #     'GoLong': bool(row['GoLong']) if row.get('GoLong') is not None else False,
                #     'GoShort': bool(row['GoShort']) if row.get('GoShort') is not None else False
                # },
                indicators=row.to_dict(),
                # message=f"Signal for {row['ticker']} @ {row['interval']}: {row['Signal']}"
            ))
        return results

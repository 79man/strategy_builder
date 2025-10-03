from typing import Optional
from strategies import HeikinAshiStrategy
from data_sources.local_ticker_data_source import LocalTickerDataSource

from utils.indicators import HeikinAshiCalculatorTest, HeikinAshiCalculator

from datetime import datetime
import pandas as pd
import logging

import logging

# Basic global logging configuration (optional, but good practice for overall application)
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(name)s - %(levelname)s -  %(funcName)s():%(lineno)d - %(message)s')

logger = logging.getLogger(__name__)


from io import StringIO
from typing import TextIO

def print_df_characteristics(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str):
    """
    Prints a detailed comparison of the characteristics of two pandas DataFrames.

    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        name1 (str): The name for the first DataFrame.
        name2 (str): The name for the second DataFrame.
    """
    print("-" * 80)
    print(f"Comparison of DataFrame '{name1}' and DataFrame '{name2}'")
    print("-" * 80)

    # Use StringIO to capture info() output for display
    def get_info_str(df: pd.DataFrame) -> str:
        buffer = StringIO()
        df.info(buf=buffer, verbose=True)
        return buffer.getvalue()

    info1 = get_info_str(df1)
    info2 = get_info_str(df2)

    # --- Print Basic Info ---
    print(f"\n--- Basic Info ---")
    print(f"Shape of {name1}: {df1.shape}")
    print(f"Shape of {name2}: {df2.shape}")
    print(f"\nInfo for {name1}:\n{info1}")
    print(f"\nInfo for {name2}:\n{info2}")

    # --- Print Dtypes ---
    print("\n--- Dtype Comparison ---")
    dtypes_df = pd.DataFrame(
        {f"{name1} Dtypes": df1.dtypes, f"{name2} Dtypes": df2.dtypes})
    print(dtypes_df)

    # --- Print Index Info ---
    print("\n--- Index Comparison ---")
    print(f"Index for {name1}: {df1.index}")
    print(f"Index for {name2}: {df2.index}")

    # --- Print Column Info ---
    print("\n--- Column Comparison ---")
    print(f"Columns for {name1}: {list(df1.columns)}")
    print(f"Columns for {name2}: {list(df2.columns)}")

    # --- Check for exact equality ---
    are_equal = False
    try:
        are_equal = df1.equals(df2)
    except Exception as e:
        print(f"\nWarning: df.equals() raised an error: {e}")

    print(f"\nFinal Check: {name1}.equals({name2}) returned: {are_equal}")

    # --- Print Visual Inspection ---
    if not are_equal:
        print("\n--- Visual Inspection (First 5 Rows) ---")
        print(f"Head of '{name1}':\n{df1.head()}")
        print(f"\nHead of '{name2}':\n{df2.head()}")

        print("\n--- Visual Inspection (Last 5 Rows) ---")
        print(f"Tail of '{name1}':\n{df1.tail()}")
        print(f"\nTail of '{name2}':\n{df2.tail()}")

        # --- Value-level Comparison ---
        # Only compare if basic characteristics match
        if df1.shape == df2.shape and list(df1.columns) == list(df2.columns):
            print("\n--- Value-Level Comparison (using df.compare()) ---")
            try:
                diff = df1.compare(df2)
                if not diff.empty:
                    print(diff)
                else:
                    print(
                        "df.compare() found no differences, likely due to floating-point or dtype issues.")
            except ValueError as e:
                print(f"Cannot perform df.compare(): {e}")

def run_tests():
    logger.info("Starting Heikin Ashi Strategy Tests...")

    # Plan:
    # 1. Confirm Heikin Ashi Generation for 2 tickers.
    # 1.1. Confirm if Heikin Ashi calculations are correct. Debug if necessary.
    # 1.2. Confirm if incremental heikinAshi calculations are correct and efficient. Debug if necessary.
    # 1.2.1. Confirm this for 1 candle, and multiple candles.

    test_creation_request = {
        "params": {
            "ha_interval": "1m",
            "ema_interval": "15m",

            "ha_shift": 0,
            "ema_shift": 0,

            "fast_ema_period": 30,
            "slow_ema_period": 50,

            "fast_ema_shift": 0,
            "slow_ema_shift": 0,

            "lookback": 5000
        },
        "tickers": {
            "BTCUSD": [
                "15m",
                "1m"
            ]
        }
    }

    storage_path = "test_strategy_data"

    # Generate intial OHLC DataFrame for testing
    full_1m_ohlc_data: pd.DataFrame = HeikinAshiCalculatorTest.generate_ohlc_data(
        5000, '1min')
    full_15m_ohlc_data: pd.DataFrame = HeikinAshiCalculatorTest.generate_ohlc_data(
        5000, '15min')

    # Clean data (remove NaNs if any)
    full_1m_ohlc_data.dropna(inplace=True)
    full_15m_ohlc_data.dropna(inplace=True)

    # Split into initial and new data for incremental testing. Do this for both data sets

    initial_ohlc = {}
    new_ohlc = {}

    initial_ohlc['1m'] = full_1m_ohlc_data.iloc[:1000].copy()
    new_ohlc['1m'] = full_1m_ohlc_data.iloc[1000:].copy()

    initial_ohlc['15m'] = full_15m_ohlc_data.iloc[:1000].copy()
    new_ohlc['15m'] = full_15m_ohlc_data.iloc[1000:].copy()

    # Create a HeikinAshiStrategy instance with LocalTickerDataSource
    # data_source = LocalTickerDataSource()
    strategy = HeikinAshiStrategy(
        params=test_creation_request["params"],
        storage_path=storage_path
    )

    # Check if the creation request is valid
    strategy.validate_creation_request(
        tickers=test_creation_request["tickers"],
        params=test_creation_request["params"]
    )

    # Initialize each ticker and interval (without start_datetime for simplicity)
    for ticker, intervals in test_creation_request["tickers"].items():
        for interval in intervals:
            strategy.initialize_ticker(
                ticker=ticker,
                interval=interval,
                params=test_creation_request["params"],
                use_callback=False,
                ohlc_df=initial_ohlc[interval] if interval in initial_ohlc else None
            )

    # Get ticker DataFrames to verify initialization
    df_1m_ticker: Optional[pd.DataFrame] = strategy.get_ticker_df(
        "BTCUSD", "1m")
    df_15m_ticker: Optional[pd.DataFrame] = strategy.get_ticker_df(
        "BTCUSD", "15m")

    if df_1m_ticker is None or df_15m_ticker is None:
        logger.error("Failed to retrieve ticker DataFrames.")
        raise ValueError("Ticker DataFrames are not available.")
    else:
        ha_calc = HeikinAshiCalculator()

        logger.info(
            f"Ticker DataFrames retrieved successfully with lengths: 1m={len(df_1m_ticker)}, 15m={len(df_15m_ticker)}")

        # Compare with same data being passed to HeikinAshiCalculator

        df_1m_ha = ha_calc.calculate_ha(initial_ohlc["1m"])
        df_15m_ha = ha_calc.calculate_ha(initial_ohlc["15m"])

        are_1m_dataframes_equal = df_1m_ha.equals(df_1m_ticker)
        are_15m_dataframes_equal = df_15m_ha.equals(df_15m_ticker)
        print(
            f"\n1m Ticker HA Result is identical to full HA calc? : {are_1m_dataframes_equal}")
        print(
            f"\n15m Ticker HA Result is identical to full HA calc? : {are_15m_dataframes_equal}")

        if not are_1m_dataframes_equal:
            print_df_characteristics(
                df_1m_ha, df_1m_ticker, "df_1m_ha", "df_1m_ticker")

        if not are_15m_dataframes_equal:
            print_df_characteristics(
                df_15m_ha, df_15m_ticker, "df_15m_ha", "df_15m_ticker")

        if are_1m_dataframes_equal and are_15m_dataframes_equal:
            print("Initial HA calculation test passed for both intervals.")
            # Now test incremental HA calculation
            for ticker, intervals in test_creation_request["tickers"].items():
                for interval in intervals:
                    strategy.update_new_candles(
                        ticker, interval, new_ohlc[interval])

            # Get updated ticker DataFrames after incremental update
            updated_df_1m_ticker: Optional[pd.DataFrame] = strategy.get_ticker_df(
                "BTCUSD", "1m")
            updated_df_15m_ticker: Optional[pd.DataFrame] = strategy.get_ticker_df(
                "BTCUSD", "15m")

            if updated_df_1m_ticker is None or updated_df_15m_ticker is None:
                logger.error(
                    "Failed to retrieve updated ticker DataFrames.")
                raise ValueError(
                    "Updated Ticker DataFrames are not available.")
            else:
                # Compare with full recalculation on combined data
                full_1m_ha = ha_calc.calculate_ha(full_1m_ohlc_data.copy())
                full_15m_ha = ha_calc.calculate_ha(
                    full_15m_ohlc_data.copy())

                are_updated_1m_dataframes_equal = updated_df_1m_ticker.equals(
                    full_1m_ha)
                are_updated_15m_dataframes_equal = updated_df_15m_ticker.equals(
                    full_15m_ha)

                print(
                    f"\nUpdated 1m Ticker HA Result is identical to full HA calc? : {are_updated_1m_dataframes_equal}")
                print(
                    f"\nUpdated 15m Ticker HA Result is identical to full HA calc? : {are_updated_15m_dataframes_equal}")

                if not are_updated_1m_dataframes_equal:
                    print_df_characteristics(
                        updated_df_1m_ticker, full_1m_ha, "updated_df_1m_ticker", "full_1m_ha")

                if not are_updated_15m_dataframes_equal:
                    print_df_characteristics(
                        updated_df_15m_ticker, full_15m_ha, "updated_df_15m_ticker", "full_15m_ha")
        else:
            print(
                "Initial HA calculation test failed for one or both intervals. Skipping incremental test.")
            return
        
    logger.info("Heikin Ashi Strategy Calculation Tests completed.")

    # Check Signals
    signal = strategy.get_last_signal()

    if signal is not None:
        print(f"\nGenerated Signals:\n{signal}")
    else:
        print("No signals generated.")

    # Get all generated signals and plot as a graph
    all_signals = strategy.get_signals_df()
    
    if all_signals is not None:
        print(f"\nAll Signals DataFrame:\n{len(all_signals)} rows")

        # Signals DataFrame should have columns ticker,interval,Signal,Open,High,Low,Close,HA_open,HA_high,HA_low,HA_close,FastEMA,SlowEMA,GoLong,GoShort
        # It should have datetime as index
        if not all_signals.empty:
            expected_columns = [
                'ticker', 'interval', 'Signal', 'Open', 'High', 'Low', 'Close',
                'HA_open', 'HA_high', 'HA_low', 'HA_close',
                'FastEMA', 'SlowEMA', 'GoLong', 'GoShort'
            ]
            missing_columns = [
                col for col in expected_columns if col not in all_signals.columns]
            if missing_columns:
                print(
                    f"Warning: Signals DataFrame is missing expected columns: {missing_columns}")
            else:
                print("Signals DataFrame contains all expected columns.")
            
            if not pd.api.types.is_datetime64_any_dtype(all_signals.index):
                print("Warning: Signals DataFrame index is not datetime type.")
            else:
                print("Signals DataFrame index is datetime type.")

    else: 
        print("No signals DataFrame available.")
    

    # import plotly.express as px
    # fig = px.scatter(all_signals, x='datetime', y='HA_close', color='Signal',
    #                  title='Heikin Ashi Strategy Signals', labels={'HA_close': 'Heikin Ashi Close Price'})
    # fig.update_traces(marker=dict(size=6))
    # fig.update_layout(legend_title_text='Signal Type')
    # fig.show()

def run_timimg_tests():
    # Get 10K DFs for 1m and 15m intervals, using HeikinAshiCalculatorTest.generate_ohlc_data
    full_1m_ohlc_data: pd.DataFrame = HeikinAshiCalculatorTest.generate_ohlc_data(
        10000, '1min')
    full_15m_ohlc_data: pd.DataFrame = HeikinAshiCalculatorTest.generate_ohlc_data(
        10000, '15min')
    
    # Run timing tests on HeikinAshiStrategy for the following cases:
    # 1. Calculate HA for full 10K rows
    # 2. Calculate HA for initial 1K rows, then one incremental call for remaining 9K rows
    # 3. Calculate HA for initial 9K rows, then one incremental calls for additional 1K rows
    # 4. Calculate HA for initial 1K rows, then one incremental call for remaining 9K rows in 1K chunks (9 calls)
    # 5. Calculate HA for initial 9500 rows, then incremental calls for 10 rows each (50 calls)

    storage_path = "timing_test_strategy_data"
    test_creation_request = {
        "params": {
            "ha_interval": "1m",
            "ema_interval": "15m",

            "ha_shift": 0,
            "ema_shift": 0,

            "fast_ema_period": 30,
            "slow_ema_period": 50,

            "fast_ema_shift": 0,
            "slow_ema_shift": 0,

            "lookback": 5000,

            'show_long_only': "False"
        },
        "tickers": {
            "BTCUSD": [
                "1m", "15m"
            ]
        }
    }

    print("\n--- Timing Test: Full 10K rows for 1m, 15m intervals ---")
    strategy = HeikinAshiStrategy(
        params=test_creation_request["params"],
        storage_path=storage_path
    )

    # Check if the creation request is valid
    strategy.validate_creation_request(
        tickers=test_creation_request["tickers"],
        params=test_creation_request["params"]
    )

    # Split into initial and new data for incremental testing. Do this for both data sets
    initial_ohlc = {}
    initial_ohlc['1m'] = full_1m_ohlc_data.copy()
    initial_ohlc['15m'] = full_15m_ohlc_data.copy()

    #Print summary of dateranges for 1m and 15m dfs
    print(f"1m OHLC Data: {initial_ohlc['1m'].index.min()} to {initial_ohlc['1m'].index.max()}, {len(initial_ohlc['1m'])} rows")
    print(f"15m OHLC Data: {initial_ohlc['15m'].index.min()} to {initial_ohlc['15m'].index.max()}, {len(initial_ohlc['15m'])} rows")

    # Start timing
    import timeit
    import time

    start_time = time.perf_counter()
    # Initialize each ticker and interval (without start_datetime for simplicity)
    for ticker, intervals in test_creation_request["tickers"].items():
        for interval in intervals:
            strategy.initialize_ticker(
                ticker=ticker,
                interval=interval,
                params=test_creation_request["params"],
                use_callback=False,
                ohlc_df=initial_ohlc[interval] if interval in initial_ohlc else None
            )
    end_time = time.perf_counter()
    print(f"Time taken for initialization: {end_time - start_time:.4f} seconds")

    # Check number of signals generated matches length of 1m dataframe
    signals_df = strategy.get_signals_df()
    if signals_df is None:
        raise ValueError("No signals generated.")
    else:
        assert signals_df.shape[0] == initial_ohlc['1m'].shape[0], f"Number of signals generated ({signals_df.shape[0]}) does not match 1m dataframe length"

    # Print the count of various types of signals generated in the Signal Column. Use groupBy.
    signal_counts = signals_df['Signal'].value_counts()
    print("Count of various types of signals generated:")
    for signal_type, count in signal_counts.items():
        print(f"  {signal_type}: {count}")

    ################################################################################

    print("\n--- Timing Test: HA for initial 1K rows, then one incremental call for remaining 9K rows ---")
    strategy = HeikinAshiStrategy(
        params=test_creation_request["params"],
        storage_path=storage_path
    )

    # Check if the creation request is valid
    strategy.validate_creation_request(
        tickers=test_creation_request["tickers"],
        params=test_creation_request["params"]
    )

    # Split into initial and new data for incremental testing. Do this for both data sets
    initial_ohlc = {}
    new_data = {}
    initial_ohlc['1m'] = full_1m_ohlc_data.iloc[:1000].copy()
    new_data['1m'] = full_1m_ohlc_data.iloc[1000:].copy()

    initial_ohlc['15m'] = full_15m_ohlc_data.iloc[:1000].copy()
    new_data['15m'] = full_15m_ohlc_data.iloc[1000:].copy()

    #Print summary of dateranges for 1m and 15m dfs
    print(f"1m OHLC Data: {initial_ohlc['1m'].index.min()} to {initial_ohlc['1m'].index.max()}, {len(initial_ohlc['1m'])} rows")
    print(f"15m OHLC Data: {initial_ohlc['15m'].index.min()} to {initial_ohlc['15m'].index.max()}, {len(initial_ohlc['15m'])} rows")

    # Start timing
    import timeit
    import time

    start_time = time.perf_counter()
    # Initialize each ticker and interval (without start_datetime for simplicity)
    for ticker, intervals in test_creation_request["tickers"].items():
        for interval in intervals:
            strategy.initialize_ticker(
                ticker=ticker,
                interval=interval,
                params=test_creation_request["params"],
                use_callback=False,
                ohlc_df=initial_ohlc[interval] if interval in initial_ohlc else None
            )
    end_time = time.perf_counter()
    print(f"Time taken for initialization: {end_time - start_time:.4f} seconds")

    # Check number of signals generated matches length of 1m dataframe
    signals_df = strategy.get_signals_df()
    if signals_df is None:
        raise ValueError("No signals generated.")
    else:
        assert signals_df.shape[0] == initial_ohlc['1m'].shape[0], f"Number of signals generated ({signals_df.shape[0]}) does not match 1m dataframe length"

    # Print the count of various types of signals generated in the Signal Column. Use groupBy.
    signal_counts = signals_df['Signal'].value_counts()
    print("Count of various types of signals generated:")
    for signal_type, count in signal_counts.items():
        print(f"  {signal_type}: {count}")

    start_time = time.perf_counter()
    # Call strategy.update_new_candles() to incorporate new data for each ticker interval
    for ticker, intervals in test_creation_request["tickers"].items():
        for interval in intervals:
            strategy.update_new_candles(
                ticker=ticker,
                interval=interval,
                ohlc_df=new_data[interval]
            )
    end_time = time.perf_counter()
    print(f"Time taken for updating new candles: {end_time - start_time:.4f} seconds")

    # Check number of signals generated matches length of 1m dataframe
    signals_df = strategy.get_signals_df()
    if signals_df is None:
        raise ValueError("No signals generated.")
    else:
        assert signals_df.shape[0] == full_1m_ohlc_data.shape[0], f"Number of signals generated ({signals_df.shape[0]}) does not match 1m dataframe length"

    # Print the count of various types of signals generated in the Signal Column. Use groupBy.
    signal_counts = signals_df['Signal'].value_counts()
    print("Count of various types of signals generated:")
    for signal_type, count in signal_counts.items():
        print(f"  {signal_type}: {count}")

    ###################################################################
    

if __name__ == "__main__":
    # run_tests()
    run_timimg_tests()

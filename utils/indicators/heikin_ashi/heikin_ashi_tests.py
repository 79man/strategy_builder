import timeit
import time
import numpy as np
import pandas as pd

from .heikin_ashi_calculator import HeikinAshiCalculator


class HeikinAshiCalculatorTest:
    # --- Generate test data ---
    @staticmethod
    def generate_ohlc_data(size: int, freq='1min') -> pd.DataFrame:
        """Generates synthetic OHLC data for testing."""
        start_date = pd.Timestamp('2020-01-01')
        dates = pd.date_range(
            start=start_date, periods=size, freq=freq, tz='UTC')

        np.random.seed(42)  # For reproducible results
        price = 100 + np.random.randn(size).cumsum()
        volumes = np.random.randint(100, 95678432, size=size)
        memo_fields = [f'Test{i}' for i in range(size)]

        ohlc = pd.DataFrame(
            index=dates,
            columns=['Open', 'High', 'Low', 'Close', 'volume', 'memo']
        )

        # ohlc['datetime'] = dates
        ohlc['volume'] = volumes
        ohlc['memo'] = memo_fields

        ohlc['Open'] = price
        ohlc['Close'] = price + np.random.randn(size)
        ohlc['High'] = ohlc[['Open', 'Close']].max(
            axis=1) + np.abs(np.random.randn(size) * 5)
        ohlc['Low'] = ohlc[['Open', 'Close']].min(
            axis=1) - np.abs(np.random.randn(size) * 5)
        ohlc['High'] = ohlc.apply(lambda row: max(
            row['Open'], row['Close'], row['High']), axis=1)
        ohlc['Low'] = ohlc.apply(lambda row: min(
            row['Open'], row['Close'], row['Low']), axis=1)

        ohlc.index.name = 'datetime'

        return ohlc

    @staticmethod
    def run_6000_row_test_with_timing():

        full_ohlc_data = HeikinAshiCalculatorTest.generate_ohlc_data(6000)
        ha_calculator = HeikinAshiCalculator()

        # Time the full calculation (for verification and comparison)
        print(f"--- Generated {len(full_ohlc_data)} rows data ---")

        full_scratch_timer = timeit.Timer(
            'ha_calculator.calculate_ha(full_ohlc_data)',
            globals={'ha_calculator': ha_calculator,
                     'full_ohlc_data': full_ohlc_data}
        )
        full_scratch_time = full_scratch_timer.timeit(number=1)
        print(
            f"Full scratch calculation time for {len(full_ohlc_data)}: {full_scratch_time:.6f} seconds")

        # Time the incremental calculation approach
        print("\nTiming incremental calculation approach...")

        # Setup for the incremental timing: do the initial 1000 rows once
        initial_ohlc = full_ohlc_data.iloc[:1000]
        new_ohlc_data = full_ohlc_data.iloc[1000:]

        start_time = time.perf_counter()
        initial_ha_df = ha_calculator.calculate_ha(initial_ohlc)
        end_time = time.perf_counter()
        initial_ha_time = end_time - start_time

        print(
            f"Initial HA calculation time for {len(initial_ohlc)}: {initial_ha_time:.6f} seconds")
        # print("Last 5 rows of initial HA:\n", initial_ha_df.tail())

        incremental_timer = timeit.Timer(
            'ha_calculator.incremental_ha_calculation(initial_ha_df, new_ohlc_data)',
            globals={'ha_calculator': ha_calculator,
                     'initial_ha_df': initial_ha_df, 'new_ohlc_data': new_ohlc_data}
        )
        # Run the incremental part multiple times for a more stable average
        incremental_runs = 25
        incremental_time = incremental_timer.timeit(
            number=incremental_runs) / incremental_runs
        print(
            f"Average incremental calculation time for {len(new_ohlc_data)} ({incremental_runs} runs): {incremental_time:.6f} seconds")
        # --- Performance Summary ---
        print("\n--- Performance Summary ---")
        print(
            f"Full calculation ({len(full_ohlc_data)} rows): {full_scratch_time:.6f} seconds")
        print(
            f"Incremental calculation ({len(new_ohlc_data)} rows): {incremental_time:.6f} seconds (per run)")

        # For verification, run the incremental one more time and check against the full result
        full_ha_history_incremental = ha_calculator.incremental_ha_calculation(
            initial_ha_df, new_ohlc_data)
        # full_ha_history_incremental = pd.concat(
        #     [initial_ha_df, incremental_ha_df])
        full_ha_history_scratch = ha_calculator.calculate_ha(
            full_ohlc_data)
        are_dataframes_equal = full_ha_history_incremental.reset_index(drop=True).equals(
            full_ha_history_scratch.reset_index(drop=True)
        )
        print(
            f"\nIncremental result is identical to full scratch result: {are_dataframes_equal}")
        if not are_dataframes_equal:
            comparison = full_ha_history_incremental.compare(
                full_ha_history_scratch)
            print("\nDifferences found:\n", comparison)

            print("Dtypes of incremental result:\n",
                  full_ha_history_incremental.dtypes)
            print("Dtypes of full scratch result:\n",
                  full_ha_history_scratch.dtypes)

        ################################################
        # Checking for incremental test wrapper approach
        ################################################
        print("\nTiming incremental test wrapper calculation approach...")
        incremental_timer = timeit.Timer(
            'ha_calculator.incremental_ha_calculation_test(initial_ha_df, new_ohlc_data)',
            globals={'ha_calculator': ha_calculator,
                     'initial_ha_df': initial_ha_df, 'new_ohlc_data': new_ohlc_data}
        )
        # Run the incremental part multiple times for a more stable average
        incremental_runs = 25
        incremental_time = incremental_timer.timeit(
            number=incremental_runs) / incremental_runs
        print(
            f"Average incremental calculation time (over {incremental_runs} runs): {incremental_time:.6f} seconds")
        # --- Performance Summary ---
        print("\n--- Performance Summary ---")
        print(
            f"Full calculation ({len(full_ohlc_data)} rows): {full_scratch_time:.6f} seconds")
        print(
            f"Incremental calculation ({len(new_ohlc_data)} rows): {incremental_time:.6f} seconds (per run)")

        # For verification, run the incremental one more time and check against the full result
        full_ha_history_incremental = ha_calculator.incremental_ha_calculation_test(
            initial_ha_df, new_ohlc_data)
        # full_ha_history_incremental = pd.concat(
        #     [initial_ha_df, incremental_ha_df])
        full_ha_history_scratch = ha_calculator.calculate_ha(
            full_ohlc_data)
        are_dataframes_equal = full_ha_history_incremental.reset_index(drop=True).equals(
            full_ha_history_scratch.reset_index(drop=True)
        )
        print(
            f"\nIncremental result is identical to full scratch result: {are_dataframes_equal}")
        if not are_dataframes_equal:
            comparison = full_ha_history_incremental.compare(
                full_ha_history_scratch)
            print("\nDifferences found:\n", comparison)

            print("Dtypes of incremental result:\n",
                  full_ha_history_incremental.dtypes)
            print("Dtypes of full scratch result:\n",
                  full_ha_history_scratch.dtypes)

    @staticmethod
    def test_other_columns_are_preserved():
        """
        Test to ensure that non-OHLC columns are preserved in the HA calculation.
        """
        ohlc_data = HeikinAshiCalculatorTest.generate_ohlc_data(10)
        ha_calculator = HeikinAshiCalculator()
        ha_df = ha_calculator.calculate_ha(ohlc_data)

        assert 'volume' in ha_df.columns, "Volume column missing after HA calculation"
        assert 'memo' in ha_df.columns, "Memo column missing after HA calculation"
        assert ha_df['volume'].equals(
            ohlc_data['volume']), "Volume data altered after HA calculation"
        assert ha_df['memo'].equals(
            ohlc_data['memo']), "Memo data altered after HA calculation"

        print(
            "Test passed: Non-OHLC columns are preserved correctly for from scratch calc.")

        new_ohlc_data = HeikinAshiCalculatorTest.generate_ohlc_data(500)
        orig_ha_df = ha_calculator.calculate_ha(
            new_ohlc_data[:100])  # Initial calculation
        new_ha_df = ha_calculator.incremental_ha_calculation_test(
            ha_df_previous=orig_ha_df, df_new=new_ohlc_data[100:])

        assert 'volume' in new_ha_df.columns, "Volume column missing after HA calculation"
        assert 'memo' in new_ha_df.columns, "Memo column missing after HA calculation"
        assert new_ha_df['volume'].equals(
            new_ohlc_data['volume']), "Volume data altered after HA calculation"
        assert new_ha_df['memo'].equals(
            new_ohlc_data['memo']), "Memo data altered after HA calculation"

        print(
            "Test passed: Non-OHLC columns are preserved correctly for incremental calc.")


def run_tests():
    ha_test = HeikinAshiCalculatorTest()
    ha_test.test_other_columns_are_preserved()
    ha_test.run_6000_row_test_with_timing()


if __name__ == '__main__':

    # Run the test
    run_tests()

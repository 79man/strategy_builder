import pandas as pd
import pandas_ta as ta

import logging

logger = logging.getLogger(__name__)


class HeikinAshiCalculator:

    def calculate_ha(self, ohlc_df: pd.DataFrame):
        """
        Calculates Heikin-Ashi candles for an entire OHLC DataFrame from scratch,
        retaining all original columns.

        Args:
            ohlc_df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns.

        Returns:
            pd.DataFrame: A new DataFrame with the original data and the HA columns.
        """
        ha_only_df = ohlc_df.ta.ha().astype('float64')
        return pd.concat([ohlc_df, ha_only_df], axis=1)

    def _incremental_ha_calculation_internal(self, ha_df_previous: pd.DataFrame, df_new: pd.DataFrame):
        """
        Calculates Heikin-Ashi for a new set of candles, using the last row
        of a previously calculated HA DataFrame as a seed, and combines the
        result with df_new.

        Args:
            ha_df_previous (pd.DataFrame): DataFrame with previously calculated HA candles and other original columns.
            df_new (pd.DataFrame): DataFrame with one or more new OHLC candles and other original columns.

        Returns:
            pd.DataFrame: A new DataFrame containing ha_df_previous concatenated with the newly calculated HA candles and other original columns from df_new.
        """
        if ha_df_previous.empty:
            # This retains all original columns
            return self.calculate_ha(df_new)
        if df_new.empty:
            return ha_df_previous

        last_ha_row = ha_df_previous.iloc[-1]
        previous_ha_open = last_ha_row.get('HA_open')
        previous_ha_close = last_ha_row.get('HA_close')

        last_known_timestamp = last_ha_row.name

        # Keep only rows in df_new where the index is strictly greater than the last known timestamp.
        df_new = df_new[df_new.index > last_known_timestamp].copy()

        if df_new.empty:
            return ha_df_previous

        ha_only_df_new = pd.DataFrame(
            columns=['HA_open', 'HA_high', 'HA_low', 'HA_close'], index=df_new.index)

        current_ha_open = previous_ha_open
        current_ha_close = previous_ha_close

        for i, row in df_new.iterrows():
            ha_close = (row['Open'] + row['High'] +
                        row['Low'] + row['Close']) / 4
            ha_open = (current_ha_open + current_ha_close) / 2
            ha_high = max(row['High'], ha_open, ha_close)
            ha_low = min(row['Low'], ha_open, ha_close)

            ha_only_df_new.loc[i, 'HA_open'] = ha_open
            ha_only_df_new.loc[i, 'HA_close'] = ha_close
            ha_only_df_new.loc[i, 'HA_high'] = ha_high
            ha_only_df_new.loc[i, 'HA_low'] = ha_low

            current_ha_open = ha_open
            current_ha_close = ha_close

        ha_only_df_new = ha_only_df_new.astype('float64')
        new_combined_df = pd.concat([df_new, ha_only_df_new], axis=1)
        return pd.concat([ha_df_previous, new_combined_df])

    def incremental_ha_calculation(
        self,
        ha_df_previous: pd.DataFrame,
        df_new: pd.DataFrame,
        threshold: int = 1000
    ):
        """
        Calculates Heikin-Ashi for new candles, optimizing by using a full recalculation
        with pandas-ta if the new data chunk is large, or incremental calculation otherwise,
        retaining all columns.

        Args:
            ha_df_previous (pd.DataFrame): DataFrame with previously calculated HA candles and other original columns.
            df_new (pd.DataFrame): DataFrame with one or more new OHLC candles and other original columns.
            threshold (int): The number of new rows above which a full recalculation is preferred.

        Returns:
            pd.DataFrame: A new DataFrame containing the ha_df_previous concatenated with the
                          newly calculated HA candles and original columns from df_new.
        """

        if ha_df_previous.empty:
            # This retains all original columns
            return self.calculate_ha(df_new)
        if df_new.empty:
            return ha_df_previous

        # Decide on strategy based on the number of new candles
        if len(df_new) > threshold:
            # Recalulate from scratch for large new data chunks
            logger.info(
                "Full recalculation triggered due to large new data chunk.")

            dropped_ha_df_previous = ha_df_previous.drop(
                columns=['HA_open', 'HA_high', 'HA_low', 'HA_close'], errors='ignore')
            logger.info(
                f"dropped_ha_df_previous: {dropped_ha_df_previous.columns},\n{dropped_ha_df_previous.index}")
            logger.info(f"df_new: {df_new.columns},\n {df_new.index}")

            combined_ohlc = pd.concat([dropped_ha_df_previous, df_new])
            combined_ohlc = combined_ohlc[~combined_ohlc.index.duplicated(
                keep='last')]
            return self.calculate_ha(combined_ohlc)
        else:
            # Calculate incrementally for smaller chunks
            logger.info(
                f"Incremental recalculation triggered due to small new data chunk. {len(df_new)}")
            # logger.info(f"{ha_df_previous.tail(10)}\n {df_new.head(10)}")

            return self._incremental_ha_calculation_internal(ha_df_previous, df_new)

    def incremental_ha_calculation_test(self, ha_df_previous: pd.DataFrame, df_new: pd.DataFrame):
        ''' 
        A test wrapper to call the internal incremental calculation directly.
        This is to facilitate performance testing without the threshold logic.
        Args:
            ha_df_previous (pd.DataFrame): DataFrame with previously calculated HA candles and other original columns.
            df_new (pd.DataFrame): DataFrame with one or more new OHLC candles and other original columns.
        Returns:
            pd.DataFrame: A new DataFrame containing the ha_df_previous concatenated with the
                          newly calculated HA candles and original columns from df_new.
        '''
        return self._incremental_ha_calculation_internal(ha_df_previous, df_new)

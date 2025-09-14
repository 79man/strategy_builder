import pandas as pd
from datetime import datetime
from strategy_base import Strategy
from typing import Optional

class PairTradingStrategy(Strategy):
    strategy_name = "PairTrading"

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
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        df.index.name = "datetime"

        # Spread & signal flags
        df["Spread"] = pd.Series(dtype="float")
        df["GoLong"] = False
        df["GoShort"] = False

        self._signals[(ticker, interval)] = df
        self.save_to_disk(ticker, interval)

    def _on_new_candle(self, ticker: str, interval: str, ohlc: dict) -> dict:
        """
        Ingest a new candle for one ticker.
        Signals are computed only when BOTH tickers have aligned timestamps.
        """
        df = self._signals.get((ticker, interval))
        if df is None:
            return {"error": f"{ticker}-{interval} not initialized"}

        ts = pd.to_datetime(ohlc["datetime"])
        open_, high, low, close = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]

        df.loc[ts] = {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Spread": None,
            "GoLong": False,
            "GoShort": False,
        }
        self._signals[(ticker, interval)] = df

        # Identify the other ticker (assuming 2-ticker pair)
        tickers = {t for (t, i) in self._signals if i == interval}
        if len(tickers) != 2:
            return {"warning": "Pair not complete yet"}

        other = (tickers - {ticker}).pop()
        df_other = self._signals.get((other, interval))
        if df_other is None or ts not in df_other.index:
            return {"info": f"Waiting for {other} candle at {ts}"}

        # Compute spread
        spread = close - df_other.loc[ts]["Close"]
        threshold = self.params.get("spread_threshold", 1.0)

        go_long = spread < -threshold
        go_short = spread > threshold

        # Update both frames
        self._signals[(ticker, interval)].at[ts, "Spread"] = spread
        self._signals[(ticker, interval)].at[ts, "GoLong"] = go_long
        self._signals[(ticker, interval)].at[ts, "GoShort"] = go_short

        self._signals[(other, interval)].at[ts, "Spread"] = spread
        self._signals[(other, interval)].at[ts, "GoLong"] = go_long
        self._signals[(other, interval)].at[ts, "GoShort"] = go_short

        return {
            "datetime": str(ts),
            "Spread": spread,
            "GoLong": go_long,
            "GoShort": go_short,
        }

from dotenv import load_dotenv
from abc import ABC, abstractmethod
import pandas as pd
import os
import json
from datetime import datetime, timezone
from typing import Dict, Tuple, Final, Optional, List
from schema.strategy_schema import SignalResponse

import importlib
import pkgutil
import strategies

import logging
logger = logging.getLogger(__name__)

# _DATA_DIR: Final = "./strategy_data"
# _DEFAULT_DATA_FEED_BASE_URL: Final = "http://localhost:9000"
# _DEFAULT_CALLBACK_BASE_URL: Final = "http://localhost:9000"

from config import config

# DATA_FEED_BASE_URL = config.DATA_FEED_BASE_URL
# CALLBACK_BASE_URL = config.CALLBACK_BASE_URL
STORAGE_PATH = config.STORAGE_PATH

def autodiscover_strategies():
    for _, module_name, _ in pkgutil.iter_modules(strategies.__path__):
        importlib.import_module(f"strategies.{module_name}")


def reload_all_instances(storage_path: str = STORAGE_PATH):
    for file in os.listdir(storage_path):
        if not file.endswith("_meta.json"):
            continue

        meta_path = os.path.join(storage_path, file)
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            strategy_name = meta.get("strategy","")
            ticker = meta.get("ticker","")
            interval = meta.get("interval", "")
            params = meta.get("params", {})

            cls = _STRATEGY_REGISTRY.get(strategy_name)
            if not cls:
                logger.warning(
                    f"⚠️ Strategy class {strategy_name} not registered, skipping {file}")
                continue

            instance = cls(ticker, interval, params)
            instance.load_from_disk()
            add_strategy_instance((strategy_name, ticker, interval), instance)
            logger.info(f"🔄 Reloaded {strategy_name} for {ticker} {interval}")

        except Exception as e:
            logger.error(f"⚠️ Failed to reload {file}: {e}")


# -------------------------------
# Strategy registry + manager
# -------------------------------
_STRATEGY_REGISTRY: Dict[str, type] = {}
_STRATEGY_INSTANCES: Dict[Tuple[str, str, str], "Strategy"] = {}

# -------------------------------
# Public interface for registry
# -------------------------------


def register_strategy_class(name: str, cls: type):
    _STRATEGY_REGISTRY[name] = cls


def get_strategy_class(name: str):
    return _STRATEGY_REGISTRY.get(name)


def list_strategy_classes():
    return list(_STRATEGY_REGISTRY.keys())


def add_strategy_instance(key: Tuple[str, str, str], instance: "Strategy"):
    _STRATEGY_INSTANCES[key] = instance


def get_strategy_instance(key: Tuple[str, str, str]):
    return _STRATEGY_INSTANCES.get(key)


def list_strategy_instances():
    return list(_STRATEGY_INSTANCES.keys())




class Strategy(ABC):
    # Optional: default fallback, can be overridden in subclasses
    strategy_name: str = None

    # Auto-registration hook
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "strategy_name", cls.__name__)
        register_strategy_class(name, cls)

    def __init__(
            self,
            ticker: str, interval: str = "1h", params: dict = {},
            storage_path: str = STORAGE_PATH
    ):
        self.ticker = ticker
        self.interval = interval
        self.params = params or {}
        self.storage_path = storage_path
        self.df = pd.DataFrame()
        self.signals = pd.DataFrame()

        os.makedirs(self.storage_path, exist_ok=True)

    # -----------------------------------
    # Abstract methods
    # -----------------------------------
    @abstractmethod
    def initialize(self):
        """Load initial data and prepare strategy"""
        pass

    @abstractmethod
    def _on_new_candle(self, ohlc: dict) -> dict:
        """Strategy-specific candle handling (without autosave)"""
        pass

    # -----------------------------------
    # Candle ingestion wrapper
    # -----------------------------------
    def on_new_candle(self, ohlc: dict) -> dict:
        """Wrapper around strategy-specific handler that also auto-saves"""
        result = self._on_new_candle(ohlc)
        self.save_to_disk()
        return result

    # -----------------------------------
    # Persistence
    # -----------------------------------
    @property
    def base_filename(self):
        return f"{self.__class__.__name__}_{self.ticker}_{self.interval}"

    def save_to_disk(self):
        csv_path = os.path.join(
            self.storage_path, f"{self.base_filename}_signals.csv")
        self.signals.to_csv(csv_path)

        meta = {
            "strategy": self.__class__.__name__,
            "ticker": self.ticker,
            "interval": self.interval,
            "params": self.params,
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = os.path.join(
            self.storage_path, f"{self.base_filename}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    def load_from_disk(self):
        csv_path = os.path.join(
            self.storage_path, f"{self.base_filename}_signals.csv")
        meta_path = os.path.join(
            self.storage_path, f"{self.base_filename}_meta.json")

        if os.path.exists(csv_path):
            self.signals = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.ticker = meta.get("ticker", self.ticker)
            self.interval = meta.get("interval", self.interval)
            self.params = meta.get("params", self.params)

    def get_last_signal(self) -> SignalResponse:
        """Return latest signal summary + indicators"""
        if self.signals.empty:
            return SignalResponse(
                ticker=self.ticker,
                interval=self.interval,
                strategy=self.strategy_name,
                message="No signals yet"
            )

        last_row = self.signals.iloc[-1]
        prev_row = self.signals.iloc[-2] if len(self.signals) > 1 else None

        signal = "HOLD"
        if prev_row is not None:
            if last_row.get("GoLong") and not prev_row.get("GoLong"):
                signal = "BUY"
            elif last_row.get("GoShort") and not prev_row.get("GoShort"):
                signal = "SELL"

        return SignalResponse(
            ticker=self.ticker,
            interval=self.interval,
            strategy=self.strategy_name,
            datetime=str(self.signals.index[-1]),
            signal=signal,
            indicators={
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in last_row.to_dict().items()
            }
        )

    def get_all_signals(
        self,
        # strategy_name: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[SignalResponse]:
        """Return all signals with optional pagination (limit + offset)."""
        if self.signals.empty:
            return [
                SignalResponse(
                    ticker=self.ticker,
                    interval=self.interval,
                    strategy=self.strategy_name,
                    message="No signals yet"
                )
            ]

        # Slice signals with offset/limit
        df = self.signals.iloc[offset:]
        if limit is not None:
            df = df.iloc[:limit]

        responses: List[SignalResponse] = []
        prev_row = None
        for ts, row in df.iterrows():
            signal = "HOLD"
            if prev_row is not None:
                if row.get("GoLong") and not prev_row.get("GoLong"):
                    signal = "BUY"
                elif row.get("GoShort") and not prev_row.get("GoShort"):
                    signal = "SELL"

            responses.append(
                SignalResponse(
                    ticker=self.ticker,
                    interval=self.interval,
                    strategy=self.strategy_name,
                    datetime=str(ts),
                    signal=signal,
                    indicators={
                        k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in row.to_dict().items()
                    }
                )
            )
            prev_row = row

        return responses
    
    def delete

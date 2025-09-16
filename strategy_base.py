from utils.config import config
from abc import ABC, abstractmethod
import pandas as pd
import os
import json
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional, List, Any, Union

import importlib
from pathlib import Path
import pkgutil
import hashlib
from utils.intervals import validate_interval, get_interval_timedelta
from utils.callback_url import get_callback_url
from data_source import DataSourceError

import logging
logger = logging.getLogger(__name__)

STORAGE_PATH = config.STORAGE_PATH
Path(STORAGE_PATH).mkdir(parents=True, exist_ok=True)

# -------------------------------
# Strategy registry + manager
# -------------------------------
# -------------------------------
# Private registries
# -------------------------------
# available strategy classes: name -> class
_STRATEGY_REGISTRY: Dict[str, type] = {}

# instantiated strategy instances:
# key -> (strategy_name, params_hash) -> Strategy instance
_STRATEGY_INSTANCES: Dict[Tuple[str, str], "Strategy"] = {}

# reverse map for fast lookup
_TICKER_INTERVAL_MAP: Dict[Tuple[str, str], list["Strategy"]] = {}

# -------------------------------
# Helpers
# -------------------------------


def make_params_hash(params: Optional[dict]) -> str:
    if not params:
        return "default"
    raw = json.dumps(params, sort_keys=True, separators=(",", ":"))
    h = hashlib.md5(raw.encode()).hexdigest()
    # short unique id but deterministic
    return h[:10]


def strategy_id_from_parts(name: str, params_hash: str) -> str:
    return f"{name}:{params_hash}"


def parse_strategy_id(strategy_id: str) -> Tuple[str, str]:
    if ":" not in strategy_id:
        raise ValueError(
            "Invalid strategy_id. Expect format '<Name>:<params_hash>'")
    name, ph = strategy_id.split(":", 1)
    return name, ph

# -------------------------------
# Auto-discover (imports strategy modules)
# -------------------------------


def autodiscover_strategies(package_name: str = "strategies"):
    """
    Import all modules in the strategies package so subclasses register themselves.
    Call this at startup (lifespan) or import-time if you prefer.
    """
    try:
        pkg = importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        logger.error(
            f"Failed to locate module {package_name}: {e}")
        return  # no strategies package present
    for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{package_name}.{module_name}")

# -------------------------------
# Public interface for registry
# -------------------------------


def register_strategy_class(name: str, cls: type):
    _STRATEGY_REGISTRY[name] = cls


def get_strategy_class(name: str) -> Optional[type]:
    return _STRATEGY_REGISTRY.get(name)


def list_strategy_classes() -> List[str]:
    return list(_STRATEGY_REGISTRY.keys())


def add_strategy_instance(
    # strategy_name: str,
    # params: Optional[Dict[str, Any]],
    instance: "Strategy"
):
    # params_hash = instance.params_hash  # make_params_hash(params=params)
    key = (instance.name, instance.params_hash)
    _STRATEGY_INSTANCES[key] = instance

    # update reverse lookup
    for ticker, interval in instance._signals.keys():
        _TICKER_INTERVAL_MAP.setdefault(
            (ticker, interval), []).append(instance)


def remove_strategy_instance(instance: "Strategy"):
    """
    Remove a strategy instance from the registry.
    """
    keys_to_delete = [
        key for key, inst in _STRATEGY_INSTANCES.items()
        if inst is instance
    ]
    for key in keys_to_delete:
        del _STRATEGY_INSTANCES[key]

    # remove from ticker/interval map
    for key in instance._signals.keys():
        if key in _TICKER_INTERVAL_MAP:
            _TICKER_INTERVAL_MAP[key].remove(instance)
            if not _TICKER_INTERVAL_MAP[key]:
                del _TICKER_INTERVAL_MAP[key]


def get_strategy_instance_by_key(
    strategy_name: str,
    params_hash: str
) -> Optional["Strategy"]:
    return _STRATEGY_INSTANCES.get((strategy_name, params_hash))


def get_strategy_instance_by_id(strategy_id: str) -> Optional["Strategy"]:
    name, ph = parse_strategy_id(strategy_id)
    return get_strategy_instance_by_key(name, ph)


def get_strategy_instances_by_ticker_interval(ticker: str, interval: str):
    """
    Fast lookup using reverse map.
    """
    return _TICKER_INTERVAL_MAP.get((ticker, interval), [])


def list_strategy_instances() -> List[Dict[str, Any]]:
    """Return list of dictionaries with strategy details"""
    return [
        {
            instance.strategy_id: instance.get_status()
        }
        for key, instance in _STRATEGY_INSTANCES.items()
    ]


def get_strategy_instances_by_name(strategy_name: str):
    """
    Return a list of all Strategy instances with the given strategy_name.
    Supports multiple instances with different params.
    """
    return [
        inst for inst in _STRATEGY_INSTANCES.values()
        if inst.name == strategy_name
    ]


# -------------------------------
# Base Strategy
# -------------------------------


class Strategy(ABC):
    """
    Base Strategy:
     - Each instance corresponds to a unique (strategy class + params) combination
     - Each instance can manage multiple (ticker, interval) series
     - Persistence is per (strategy, params_hash, ticker, interval)
    """

    # child classes may set a class variable `strategy_name` (optional)
    strategy_name: Optional[str] = None

    # Auto-registration hook
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "strategy_name", cls.__name__)
        register_strategy_class(name, cls)

    def __init__(
        self,
        params: Optional[dict] = None,
        storage_path: Optional[str] = None
    ):
        self.params: dict = params or {}
        self.storage_path: Path = Path(
            storage_path) if storage_path else Path(STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # internal maps keyed by (ticker, interval)
        self._signals: Dict[Tuple[str, str], pd.DataFrame] = {}

        # precompute params hash
        self.params_hash: str = make_params_hash(self.params)
        self.data_source = None  # To be updated by the concrete Strategy

        self.running: bool = False

    def get_status(self) -> str:
        return "RUNNING" if self.running else "STOPPED"

    @classmethod
    def validate_creation_request(cls, tickers: Dict[str, List[str]], params: Optional[dict] = None) -> None:
        """
        Validate strategy creation parameters before instantiation.
        Override in subclasses for strategy-specific validation.

        Args:
            tickers: Dict mapping ticker symbols to list of intervals
            params: Strategy parameters

        Raises:
            ValueError: If validation fails
        """
        # Default validation - can be overridden
        if not tickers:
            raise ValueError("At least one ticker must be specified")

        for ticker, intervals in tickers.items():
            if not ticker or not isinstance(ticker, str):
                raise ValueError(f"Invalid ticker: {ticker}")

            for interval in intervals:
                validate_interval(interval=interval)

    @classmethod
    def reload_from_metadata(cls, meta_path: Path, storage_path: Path) -> Optional["Strategy"]:
        """
        Class method to reload a strategy instance from its metadata file.
        Returns the loaded instance or None if loading fails.
        """
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Validate metadata structure
            required_fields = ["strategy_class", "params", "tickers"]
            missing_fields = [
                field for field in required_fields if field not in meta]
            if missing_fields:
                logger.warning(
                    f"Invalid metadata in {meta_path.name}: missing {missing_fields}")
                return None

            strategy_class_name = meta.get("strategy_class")
            params = meta.get("params", {})
            tickers_map = meta.get("tickers", {})
            last_update_str = meta.get("last_update")

            # Parse last update timestamp
            last_update = None
            if last_update_str:
                last_update = datetime.fromisoformat(
                    last_update_str.replace('Z', '+00:00'))

            # Get the actual strategy class
            strategy_cls = get_strategy_class(strategy_class_name)
            if not strategy_cls:
                logger.warning(
                    f"⚠️ Unknown strategy class {strategy_class_name} in {meta_path.name}")
                return None

            # Create instance
            instance = strategy_cls(
                params=params, storage_path=str(storage_path))

            # Load data for all tickers+intervals
            loaded_any = False
            for ticker, intervals in tickers_map.items():
                for interval in intervals:
                    try:
                        # Load CSV data directly into _signals
                        base = instance.base_filename(ticker, interval)
                        csv_path = storage_path / f"{base}_signals.csv"

                        if csv_path.exists():
                            df = pd.read_csv(
                                csv_path, index_col=0, parse_dates=True)
                            
                            df.index = pd.to_datetime(df.index, utc=True, format="mixed")
                            
                            # if not hasattr(df.index, 'tz') and df.index.tz is None:
                            #     df.index = df.index.tz_localize('UTC')

                            instance._signals[(ticker, interval)] = df
                            loaded_any = True
                            logger.info(
                                f"Loaded signals for {ticker}/{interval}: {df.shape}, {df.index.dtype}, {df.index.shape}")
                        else:
                            logger.warning(f"Missing signals file: {csv_path}")

                    except Exception as e:
                        logger.error(
                            f"Failed to load {ticker}/{interval} for {strategy_class_name}: {e}")

            if loaded_any:
                # Call rehydrate if it exists
                if hasattr(instance, 'rehydrate'):
                    instance.rehydrate()
                return instance
            else:
                logger.warning(
                    f"No valid data found for {strategy_class_name}, skipping")
                return None

        except Exception as e:
            logger.error(f"Failed to reload from {meta_path.name}: {e}")
            return None

    @property
    def name(self) -> str:
        return getattr(self.__class__, "strategy_name", self.__class__.__name__)

    @property
    def strategy_id(self) -> str:
        return strategy_id_from_parts(self.name, self.params_hash)

    # -----------------------------------
    # Abstract methods
    # -----------------------------------
    @abstractmethod
    def initialize_ticker(
        self,
        ticker: str, interval: str,
        params: dict,
        start_datetime: Optional[datetime] = None
    ) -> None:
        """Initialize signals and history for one ticker/interval.
        Strategy subclasses should override this."""

        raise NotImplementedError(
            "initialize_ticker must be implemented in subclass")

    @abstractmethod
    def _on_new_candle(self, ticker: str, interval: str, ohlc: dict) -> dict:
        """
        Child-specific handling for a single new candle.
        Should update self._dfs[(ticker,interval)] and self._signals[(ticker,interval)]
        and return a dict (e.g. indicators or raw row).
        """
        pass

    # --- public API for multi-ticker behavior ---
    def ensure_ticker(self, ticker: str, interval: str):
        """Ensure ticker/interval is registered and validated"""
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        if not interval or not isinstance(interval, str):
            raise ValueError("Interval must be a non-empty string")

        # Validate interval format
        validate_interval(interval=interval)

        key = (ticker, interval)
        if key not in self._signals:
            self._signals[key] = pd.DataFrame(
                columns=["Open", "High", "Low", "Close"])
            self.save_to_disk(ticker, interval)

            # update reverse lookup map
            _TICKER_INTERVAL_MAP.setdefault(key, []).append(self)

    # -----------------------------------
    # Candle ingestion wrapper
    # -----------------------------------

    def on_new_candle(self, ticker: str, interval: str, ohlc: dict) -> dict:
        result = self._on_new_candle(ticker, interval, ohlc)
        self.save_to_disk(ticker, interval)
        return result

        # --- persistence helpers (per ticker/interval) ---
    def base_filename(self, ticker: str, interval: str) -> str:
        # include params_hash in filename so different instances don't clash
        return f"{self.name}_{self.params_hash}_{ticker}_{interval}"

    def meta_filename(self) -> str:
        return f"{self.name}_{self.params_hash}_meta.json"

    def load_metadata(self) -> dict:
        meta_path = os.path.join(self.storage_path, self.meta_filename())
        with open(meta_path, "r") as f:
            meta = json.load(f)

            # Validate metadata structure
            required_fields = ["strategy_class", "params", "tickers"]
            missing_fields = [
                field for field in required_fields if field not in meta]
            if missing_fields:
                logger.warning(
                    f"Invalid metadata in {meta_path}: missing {missing_fields}")
                return {}

            last_update_str = meta.get("last_update")

            # Parse last update timestamp
            last_update = None
            if last_update_str:
                last_update = datetime.fromisoformat(
                    last_update_str.replace('Z', '+00:00'))

            meta['last_updated'] = last_update
            return meta

        return {}

    def save_to_disk(
        self,
        ticker: Optional[str] = None,
        interval: Optional[str] = None
    ):
        """Save strategy state to disk with improved error handling"""
        try:
            if ticker and interval:
                # Save only this one
                df = self._signals.get((ticker, interval))
                logger.info(f"Saving: {ticker}/{interval}: {df.shape}")
                if df is not None:
                    base = self.base_filename(ticker, interval)
                    csv_path = os.path.join(
                        self.storage_path, f"{base}_signals.csv")

                    # Ensure directory exists
                    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

                    # Save with proper index handling
                    if not df.empty:
                        # if hasattr(df.index, 'tz') and df.index.tz is not None:
                        #     # Save with timezone info preserved
                        #     df.to_csv(
                        #         csv_path, date_format='%Y-%m-%d %H:%M:%S%z', index=True)
                        # else:
                            df.to_csv(csv_path, index=True)
                    else:
                        # Create empty file with proper headers
                        pd.DataFrame(columns=["Open", "High", "Low", "Close"]).to_csv(
                            csv_path, index=True)
            else:
                # Save everything
                for (tk, iv), df in self._signals.items():
                    base = self.base_filename(tk, iv)
                    csv_path = os.path.join(
                        self.storage_path, f"{base}_signals.csv")
                    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

                    if not df.empty:
                        df.to_csv(csv_path, index=True)
                    else:
                        pd.DataFrame(columns=["Open", "High", "Low", "Close"]).to_csv(
                            csv_path, index=True)

            # Meta is always saved (covers all tickers)
            tickers_map = {}
            for tk, iv in self._signals.keys():
                tickers_map.setdefault(tk, []).append(iv)

            meta = {
                "strategy_name": self.name,
                "strategy_class": self.__class__.__name__,
                "params": self.params,
                "params_hash": self.params_hash,
                "tickers": tickers_map,
                "last_update": datetime.now(timezone.utc).isoformat(),
                "version": "1.0"  # For future compatibility
            }

            meta_path = os.path.join(
                self.storage_path, self.meta_filename())
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)

        except Exception as e:
            logger.error(
                f"Failed to save strategy {self.strategy_id} to disk: {e}")
            raise

    def rehydrate(self, last_updated: Union[datetime, None] = None) -> None:
        """Fetch missing candles since last update and subscribe to live feeds"""
        if not last_updated:
            # check if last_updated can be found from meta_data.
            meta_data = self.load_metadata()

            if 'last_updated' not in meta_data:
                logger.warning(
                    f"last_updated missing in metadata. Rehydration skipped")
                return
            last_updated = meta_data.get('last_updated')
            logger.info(
                f"Found last Updated in metadata: {last_updated}({last_updated.astimezone()})")

        current_time = datetime.now(timezone.utc)
        # Only sync if there's a significant gap

        for (ticker, interval) in self._signals.keys():
            # 3600:  # 1 hour threshold
            if (current_time - last_updated).total_seconds() > get_interval_timedelta(interval).seconds:
                if self.data_source:
                    try:
                        missing_df = self.data_source.fetch_data(
                            ticker=ticker,
                            interval=interval,
                            start_datetime=last_updated,
                            callback_url=get_callback_url(ticker, interval)
                        )
                        logger.info(
                            f"{ticker}/{interval}: Fetched {len(missing_df)} missing candles")

                        # Process missing candles through strategy logic
                        for timestamp, row in missing_df.iterrows():
                            ohlc = {
                                "Datetime": timestamp.isoformat(),
                                "open": row["Open"],
                                "high": row["High"],
                                "low": row["Low"],
                                "close": row["Close"]
                            }
                            self.on_new_candle(ticker, interval, ohlc)
                    except DataSourceError as e:
                        logger.error(
                            f"Error in fetching missing data and subscribing. {e}")
                        pass
                else:
                    logger.warning(
                        "datasource instance missing. Rehydration skipped.")

    def delete_from_disk(self, ticker: Optional[str] = None, interval: Optional[str] = None):
        """
        Delete persisted files for either:
        - specific ticker+interval if provided
        - ALL tickers for this strategy instance if ticker/interval is None
        """

        if ticker and interval:
            # delete just this one
            base = self.base_filename(ticker, interval)
            patterns = [
                os.path.join(self.storage_path, f"{base}_signals.csv"),
            ]
        else:
            # delete everything for this instance
            prefix = f"{self.name}_{self.params_hash}_"
            patterns = [os.path.join(self.storage_path, f"{prefix}*")]

        import glob
        for pattern in patterns:
            for path in glob.glob(pattern):
                try:
                    os.remove(path)
                    logger.info(f"🗑️ Deleted {path}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete {path}: {e}")
                    pass

        # always refresh meta file if there are still tickers left
        if ticker and interval:
            if (ticker, interval) in self._signals:
                del self._signals[(ticker, interval)]
            if self._signals:
                self.save_to_disk()  # rewrite meta with remaining tickers
            else:
                # if no signals left, nuke meta file
                meta_path = os.path.join(
                    self.storage_path, self.meta_filename()
                )
                try:
                    os.remove(meta_path)
                except FileNotFoundError as e:
                    logger.warning(f"Failed to delete file {meta_path}: {e}")
                    pass

    def remove_ticker(self, ticker: str, interval: str):
        """
        Remove a specific ticker/interval from this strategy instance
        and delete its data from disk.
        """
        key = (ticker, interval)
        if key in self._signals:
            del self._signals[key]
            self.delete_from_disk(ticker, interval)

            # remove from reverse map
            if key in _TICKER_INTERVAL_MAP:
                _TICKER_INTERVAL_MAP[key].remove(self)
                if not _TICKER_INTERVAL_MAP[key]:
                    del _TICKER_INTERVAL_MAP[key]

    # --- accessors for signals/dfs ---

    def get_signals_df(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        return self._signals.get((ticker, interval))

    def _get_signal(self, curr_row: pd.Series, prev_row: Optional[pd.Series]) -> str:
        signal = "HOLD"
        if prev_row is not None:
            if curr_row.get("GoLong") and not prev_row.get("GoLong"):
                signal = "BUY"
            elif curr_row.get("GoShort") and not prev_row.get("GoShort"):
                signal = "SELL"
        return signal

    # --- convenience: summary signal logic (same as earlier) ---
    def _summarize_signal_from_df(self, signals_df: Optional[pd.DataFrame]) -> dict:
        if signals_df is None or signals_df.empty:
            return {"message": "No signals yet"}
        last_row = signals_df.iloc[-1]
        prev_row = signals_df.iloc[-2] if len(signals_df) > 1 else None
        # signal = "HOLD"
        # if prev_row is not None:
        #     if last_row.get("GoLong") and not prev_row.get("GoLong"):
        #         signal = "BUY"
        #     elif last_row.get("GoShort") and not prev_row.get("GoShort"):
        #         signal = "SELL"
        signal = self._get_signal(last_row, prev_row)
        return {
            "datetime": str(signals_df.index[-1]),
            "signal": signal,
            "indicators": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in last_row.to_dict().items()
            }
        }

    def get_last_signal(self, ticker: str, interval: str) -> dict:
        df = self.get_signals_df(ticker, interval)
        res = self._summarize_signal_from_df(df)
        return {
            "strategy_id": self.strategy_id,
            "ticker": ticker,
            "interval": interval,
            **res
        }

    def get_all_signals(self, ticker: str, interval: str, offset: int = 0, limit: Optional[int] = None) -> List[dict]:
        df = self.get_signals_df(ticker, interval)
        if df is None or df.empty:
            return []
        sliced = df.iloc[offset: (None if limit is None else offset + limit)]
        results = []
        prev = None
        for ts, row in sliced.iterrows():
            # signal = "HOLD"
            # if prev is not None:
            #     if row.get("GoLong") and not prev.get("GoLong"):
            #         signal = "BUY"
            #     elif row.get("GoShort") and not prev.get("GoShort"):
            #         signal = "SELL"
            signal = self._get_signal(row, prev)
            results.append({
                "strategy_id": self.strategy_id,
                "ticker": ticker,
                "interval": interval,
                "datetime": str(ts),
                "signal": signal,
                "indicators": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.to_dict().items()}
            })
            prev = row
        return results


# -------------------------------
# Reloading all instances from disk
# -------------------------------
def reload_all_instances(storage_path: Optional[str] = None):
    """Reload all strategy instances from disk with improved error handling"""

    storage = Path(storage_path) if storage_path else Path(STORAGE_PATH)
    if not storage.exists():
        logger.info(f"Storage path {storage} does not exist, skipping reload")
        return

    _STRATEGY_INSTANCES.clear()
    _TICKER_INTERVAL_MAP.clear()

    reloaded_count = 0
    failed_count = 0

    # Scan for *_meta.json files
    for entry in storage.glob("*_meta.json"):
        try:
            # Delegate loading to the strategy class
            instance = Strategy.reload_from_metadata(entry, storage)

            if instance:
                # Register the instance
                add_strategy_instance(instance)

                reloaded_count += 1
                logger.info(f"Reloaded strategy {instance.strategy_id}")
            else:
                logger.warning("Failed")
                failed_count += 1

        except Exception as e:
            logger.error(f"Failed to process metadata file {entry.name}: {e}")
            failed_count += 1

    logger.info(
        f"Reload complete: {reloaded_count} strategies loaded, {failed_count} failed")


def old_reload_all_instances(storage_path: Optional[str] = None):
    """Reload all strategy instances from disk with improved error handling"""

    storage = Path(storage_path) if storage_path else Path(STORAGE_PATH)
    if not storage.exists():
        logger.info(f"Storage path {storage} does not exist, skipping reload")
        return

    _STRATEGY_INSTANCES.clear()
    _TICKER_INTERVAL_MAP.clear()

    reloaded_count = 0
    failed_count = 0

    # scan for *_meta.json files
    for entry in storage.glob("*_meta.json"):
        try:
            with open(entry, "r") as f:
                meta = json.load(f)

            # Validate metadata structure
            required_fields = ["strategy_class", "params", "tickers"]
            missing_fields = [
                field for field in required_fields if field not in meta]
            if missing_fields:
                logger.warning(
                    f"Invalid metadata in {entry.name}: missing {missing_fields}")
                failed_count += 1
                continue

            strategy_class_name = meta.get("strategy_class")
            params = meta.get("params", {})
            tickers_map = meta.get("tickers", {})
            last_update_str = meta.get("last_update")

            # Parse last update timestamp
            last_update = None
            if last_update_str:
                last_update = datetime.fromisoformat(
                    last_update_str.replace('Z', '+00:00'))

            cls = _STRATEGY_REGISTRY.get(strategy_class_name)
            if not cls:
                logger.warning(
                    f"⚠️ Unknown strategy class {strategy_class_name} in {entry.name}")
                failed_count += 1
                continue

            # Create instance
            try:
                instance = cls(params=params, storage_path=str(storage))

                # Load data for all tickers+intervals
                loaded_any = False
                for ticker, intervals in tickers_map.items():
                    for interval in intervals:
                        try:
                            # Load CSV data
                            base = instance.base_filename(ticker, interval)
                            csv_path = storage / f"{base}_signals.csv"

                            if csv_path.exists():
                                df = pd.read_csv(
                                    csv_path,
                                    index_col=0,
                                    parse_dates=True
                                )
                                instance._signals[(ticker, interval)] = df
                                loaded_any = True

                                logger.info(
                                    f"Loaded signals for {ticker}/{interval}: {df.shape}")

                            else:
                                logger.warning(
                                    f"Missing signals file: {csv_path}")

                        except Exception as e:
                            logger.error(
                                f"Failed to load {ticker}/{interval} for {strategy_class_name}: {e}")
                            failed_count += 1

                if loaded_any:
                    # Register the instance
                    # ph = make_params_hash(params)
                    key = (strategy_class_name, instance.params_hash)
                    _STRATEGY_INSTANCES[key] = instance

                    instance.rehydrate(last_update)

                    # Update reverse lookup map
                    for ticker_interval in instance._signals.keys():
                        _TICKER_INTERVAL_MAP.setdefault(
                            ticker_interval, []).append(instance)

                    reloaded_count += 1
                    logger.info(f"Reloaded strategy {instance.strategy_id}")
                else:
                    logger.warning(
                        f"No valid data found for {strategy_class_name}, skipping")
                    failed_count += 1

            except Exception as e:
                logger.error(
                    f"Failed to create instance for {strategy_class_name}: {e}")
                failed_count += 1

        except Exception as e:
            logger.error(f"Failed to process metadata file {entry}: {e}")
            failed_count += 1

    logger.info(
        f"Reload complete: {reloaded_count} strategies loaded, {failed_count} failed")

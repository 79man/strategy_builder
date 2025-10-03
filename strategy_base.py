from utils.config import config
from abc import ABC, abstractmethod
import pandas as pd
import os
import json
import re
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional, List, Any, Union

import importlib
from pathlib import Path
import pkgutil
import hashlib
from utils.intervals import validate_interval, get_interval_timedelta
from utils.callback_url import get_callback_url
from data_source import DataSourceError
from utils import helpers
from schema.strategy_schema import CandleFeedRequest, SignalResponse, StrategyDetails, StrategyMetadata

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
    for ticker, interval in instance._tickers.keys():
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
    for key in instance._tickers.keys():
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
    try:
        name, ph = parse_strategy_id(strategy_id)
    except Exception as e:
        logger.warning(f"get_strategy_instance_by_id() failed: {e}")
        return None
    return get_strategy_instance_by_key(name, ph)


def get_strategy_instances_by_ticker_interval(ticker: str, interval: str):
    """
    Fast lookup using reverse map.
    """
    return _TICKER_INTERVAL_MAP.get((ticker, interval), [])


# def list_strategy_instances() -> List[Dict[str, Any]]:
#     """Return list of dictionaries with strategy details"""
#     return [
#         {
#             instance.strategy_id: instance.get_status()
#         }
#         for key, instance in _STRATEGY_INSTANCES.items()
#     ]

def list_strategy_instances() -> List[StrategyDetails]:
    """Return a list of StrategyDetails for all strategy instances."""
    result = []
    for instance in _STRATEGY_INSTANCES.values():
        meta_dict = instance.load_metadata()
        # Only pass fields that exist in StrategyMetadata
        metadata = None
        if meta_dict:
            try:
                metadata = StrategyMetadata(
                    tickers=meta_dict['tickers'],
                    params=meta_dict['params'],
                    last_updated=meta_dict['last_updated']
                )
            except Exception as e:
                logger.warning(f"Failed to parse metadata for {instance.strategy_id}: {e}")
                metadata = None
        
        result.append(
            StrategyDetails(
                strategy_id=instance.strategy_id,
                status=instance.get_status(),
                metadata=metadata
            )
        )
    return result

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
        self.request_params: dict = self.params.copy()

        self.storage_path: Path = Path(
            storage_path
        ) if storage_path else Path(STORAGE_PATH)

        self.storage_path.mkdir(parents=True, exist_ok=True)

        # internal maps keyed by (ticker, interval)
        self._tickers: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._signals: pd.DataFrame = pd.DataFrame()

        # precompute params hash
        # self.params_hash: str = make_params_hash(self.params)
        self.data_source = None  # To be updated by the concrete Strategy

        self.running: bool = False

        # Add pause state
        self._is_paused: bool = False

    def get_status(self) -> str:

        if self._is_paused:
            return "PAUSED"
        elif self.running:
            return "RUNNING"
        else:
            return "STOPPED"

    def set_paused(self, paused: bool):
        """Set the paused state of this strategy instance"""
        self._is_paused = paused

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

            # Get the actual strategy class
            strategy_cls = get_strategy_class(strategy_class_name)
            if not strategy_cls:
                logger.warning(
                    f"⚠️ Unknown strategy class {strategy_class_name} in {meta_path.name}")
                return None

            is_paused = meta.get("is_paused", False)

            # Create instance
            instance = strategy_cls(
                params=params,
                storage_path=str(storage_path)
            )

            # Set pause status
            if hasattr(instance, 'set_paused'):
                instance.set_paused(is_paused)

            # Skip loading data if paused
            if is_paused:
                logger.info(
                    f"Strategy {instance.strategy_id} is paused, skipping data load from disk")
                return instance  # Mark as loaded so it gets registered

            # Delegate loading to instance method
            if instance.load_from_disk():
                return instance
            else:
                logger.warning(
                    f"No valid data found for {strategy_class_name}, skipping")
                return None

        except Exception as e:
            logger.error(f"Failed to reload from {meta_path.name}: {e}")
            return None

    @property
    def params_hash(self) -> str:
        """Dynamically computes the hash from the current params."""
        return make_params_hash(self.params)

    @property
    def name(self) -> str:
        return getattr(self.__class__, "strategy_name", self.__class__.__name__)

    @property
    def strategy_id(self) -> str:
        return strategy_id_from_parts(self.name, self.params_hash)

    @property
    def disk_safe_strategy_id(self) -> str:
        # Replace anything not safe for filenames (not alphanumeric or dash/underscore) with underscore
        return re.sub(r'[^A-Za-z0-9_\-]', '_', self.strategy_id)

    # -----------------------------------
    # Abstract methods
    # -----------------------------------
    @abstractmethod
    def initialize_ticker(
        self,
        ticker: str, interval: str,
        params: dict,
        start_datetime: Optional[datetime] = None,
        ohlc_df: Optional[pd.DataFrame] = None,
        use_callback: bool = True
    ) -> None:
        """Initialize signals and history for one ticker/interval.
        Strategy subclasses should override this."""

        raise NotImplementedError(
            "initialize_ticker must be implemented in subclass")

    @abstractmethod
    def _on_new_candles(self, ticker: str, interval: str, new_ohlc_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Child-specific handling for a single new candle.
        Should update self._dfs[(ticker,interval)] and self._tickers[(ticker,interval)]
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
        if key not in self._tickers:
            self._tickers[key] = pd.DataFrame(
                columns=["Open", "High", "Low", "Close"])
            self.save_to_disk(ticker, interval)

            # update reverse lookup map
            _TICKER_INTERVAL_MAP.setdefault(key, []).append(self)

    # -----------------------------------
    # Candle ingestion wrapper
    # -----------------------------------

    def on_new_candles(
        self, ticker: str, interval: str, ohlc_list: List[CandleFeedRequest]
    ) -> dict:

        data_dicts = [c.model_dump() for c in ohlc_list]

        # Create the DataFrame from the list of dictionaries
        df = pd.DataFrame(data_dicts)

        # Set the 'datetime' column as the index and convert to datetime objects
        df = df.set_index('datetime')
        df.index = pd.to_datetime(df.index)

        df = df.rename(
            columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})

        return self.update_new_candles(ticker, interval, df)

    def update_new_candles(
        self, ticker: str, interval: str, ohlc_df: pd.DataFrame
    ) -> dict:
        if self._is_paused:
            logger.error(f"Strategy {self.strategy_id} is Paused.")
            return {'error': f"Strategy {self.strategy_id} is Paused."}

        if ohlc_df.empty:
            logger.error(
                f"Received empty DataFrame of candles for {ticker} @ {interval}.")
            return {'error': f"Received empty DataFrame of candles for {ticker} @ {interval}."}

        ohlc_df = ohlc_df.rename(
            columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        result = self._on_new_candles(
            ticker=ticker, interval=interval, new_ohlc_df=ohlc_df)
        self.save_to_disk(ticker, interval)
        return result

    # --- persistence helpers (per ticker/interval) ---

    def load_metadata(self) -> dict:
        strat_dir = Path(self.storage_path) / self.disk_safe_strategy_id
        meta_path = strat_dir / "meta.json"

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

    def pause(self) -> Tuple[bool, str]:

        if not self.running:
            return False, f"Strategy {self.strategy_id} is not running"
        if self._is_paused:
            return False, f"Strategy {self.strategy_id} is already paused"

        self._is_paused = True
        self.save_to_disk()

        return True, f"Pause Operation Successful"

    def resume(self) -> Tuple[bool, str]:
        if not self._is_paused:
            msg = f"Not allowed to resume strategy {self.strategy_id}, as it is not Paused"
            logger.error(msg)
            return False, msg

        # Check if data needs to be reloaded (empty signals indicate app restart scenario)
        if not self._tickers or all(df.empty for df in self._tickers.values()):
            # Reload data from disk first
            if not self.load_from_disk():
                return False, f"Failed to load Strategy Data from Disk for {self.strategy_id}"
        else:
            # disk loading was done previosuly
            if not self.rehydrate():
                return False, f"Failed to rehydrate Strategy Data from Data Source for {self.strategy_id}"

        self._is_paused = False
        self.save_to_disk()
        return True, f"Resume Operation Successful"

    def restart(self) -> Tuple[bool, str]:
        """  
        Restart strategy by clearing all signals data and deleting from disk.  
        Unlike resume, this removes all existing data both in memory and on disk.  
        """
        try:
            keys = list(self._tickers.keys())

            # Delete all ticker data from disk
            for ticker, interval in keys:
                self.remove_ticker(ticker, interval)

            for ticker, interval in keys:
                self.ensure_ticker(ticker, interval)
                self.initialize_ticker(ticker, interval, self.params)

            # Ensure strategy is not paused after restart
            self._is_paused = False
            self.save_to_disk()

            return True, f"Restart Operation Successful"

        except Exception as e:
            logger.error(f"Failed to restart strategy {self.strategy_id}: {e}")
            return False, f"Failed to restart strategy: {str(e)}"

    # Save given ticker+interval (or all, if not specified)
    def _save_ticker_df(self, strat_dir: Path, ticker: str, interval: str):
        df = self._tickers.get((ticker, interval))
        csv_path = strat_dir / f"{ticker}_{interval}.csv"

        if df is not None:
            logger.info(f"Saving: {ticker}/{interval}: {df.shape}")
            if not df.empty:
                df.to_csv(csv_path, index=True)
            else:
                pd.DataFrame(columns=["Open", "High", "Low", "Close"]).to_csv(
                    csv_path, index=True)
        else:
            logger.info(f"No data to save for {ticker}/{interval}")

    def delete_from_disk(
        self,
        ticker: Optional[str] = None,
        interval: Optional[str] = None
    ):
        """
        Delete persisted files for either:
        - specific ticker+interval if provided
        - ALL tickers for this strategy instance if ticker/interval is None
        """
        strat_dir = Path(self.storage_path) / self.disk_safe_strategy_id
        if ticker and interval:
            # delete just this one

            csv_path = strat_dir / f"{ticker}_{interval}.csv"
            try:
                os.remove(csv_path)
            except FileNotFoundError as e:
                logger.warning(f"Failed to delete file {csv_path}: {e}")
        else:
            for (tk, iv), _ in self._tickers.items():
                csv_path = strat_dir / f"{tk}_{iv}.csv"
                try:
                    os.remove(csv_path)
                except FileNotFoundError as e:
                    logger.warning(f"Failed to delete file {csv_path}: {e}")

        # always refresh meta file if there are still tickers left
        if ticker and interval:
            if (ticker, interval) in self._tickers:
                del self._tickers[(ticker, interval)]
            if self._tickers:
                self.save_to_disk()  # rewrite meta with remaining tickers
            else:
                # if no signals left, nuke meta file
                meta_path = strat_dir / "meta.json"
                try:
                    os.remove(meta_path)
                except FileNotFoundError as e:
                    logger.warning(f"Failed to delete file {meta_path}: {e}")
                    pass

    def save_to_disk(
        self,
        ticker: Optional[str] = None,
        interval: Optional[str] = None
    ):
        """Save strategy state to disk with improved error handling"""
        try:
            # Create per-strategy subdir
            strat_dir = Path(self.storage_path) / self.disk_safe_strategy_id
            strat_dir.mkdir(parents=True, exist_ok=True)

            # Signals is alawys saved
            if self._signals is not None:
                df = self._signals
                logger.info(f"Saving Signals: {strat_dir.name}: {df.shape}")
                csv_path = strat_dir / "signals.csv"

                # Save with proper index handling
                if not df.empty:
                    df.to_csv(csv_path, index=True)
                else:
                    # Create empty file with proper headers
                    # pd.DataFrame(columns=["Open", "High", "Low", "Close"]).to_csv(
                    #     csv_path, index=True)
                    logger.info(
                        f"Signals df for {self.strategy_id} is empty. Skipping save")

            if ticker and interval:
                self._save_ticker_df(
                    strat_dir=strat_dir,
                    ticker=ticker,
                    interval=interval
                )
            else:
                for (tk, iv), _ in self._tickers.items():
                    self._save_ticker_df(
                        strat_dir=strat_dir,
                        ticker=tk,
                        interval=iv
                    )

            # Meta is always saved (covers all tickers)
            tickers_map = {}
            for tk, iv in self._tickers.keys():
                tickers_map.setdefault(tk, []).append(iv)

            meta = {
                "strategy_name": self.name,
                "strategy_class": self.__class__.__name__,
                "request_params": self.request_params,
                "params": self.params or {},
                "params_hash": self.params_hash,
                "tickers": tickers_map,
                "last_update": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",  # For future compatibility
                "is_paused": self._is_paused,  # Add pause status
            }

            meta_path = strat_dir / "meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)

        except Exception as e:
            logger.error(
                f"Failed to save strategy {self.strategy_id} to disk: {e}")
            raise

    def load_from_disk(self) -> bool:
        """  
        Instance method to reload all tickers/intervals from disk.  
        Returns True if any data was loaded.  
        """
        loaded_any = False
        strat_dir = Path(self.storage_path) / self.disk_safe_strategy_id

        # Read metadata to get tickers map
        meta = self.load_metadata()

        if not meta:  # Only continue if meta is valid
            logger.error(f"Failed to load Metadata for {self.strategy_id}")
            return False

        try:
            logger.info(f"Reloading from disk: meta: {meta}")
            tickers_map = meta.get("tickers", {})

            logger.info(f"Reloading from disk: {tickers_map}")

            # Reload signals if existing
            # Reload signals.csv if existing
            signals_path = strat_dir / "signals.csv"

            if signals_path.exists():
                df = pd.read_csv(
                    signals_path, index_col=0, parse_dates=True
                )
                df.index = pd.to_datetime(
                    df.index, utc=True, format="mixed")
                self._signals = df
                logger.info(f"Loaded signals df. {len(self._signals)}")
            else:
                logger.info(f"Signals file does not exist. Skipping")

            # Load data for all tickers+intervals
            for ticker, intervals in tickers_map.items():
                for interval in intervals:
                    try:
                        csv_path = strat_dir / f"{ticker}_{interval}.csv"

                        if csv_path.exists():
                            df = pd.read_csv(
                                csv_path, index_col=0, parse_dates=True)
                            df.index = pd.to_datetime(
                                df.index, utc=True, format="mixed")
                            self._tickers[(ticker, interval)] = df
                            loaded_any = True
                            logger.info(
                                f"Loaded signals for {ticker}/{interval}: {df.shape}")
                        else:
                            logger.warning(
                                f"Missing tickers data file: {csv_path}")

                    except Exception as e:
                        logger.error(
                            f"Failed to load {ticker}/{interval}: {e}")

            # Call rehydrate if data was loaded and instance is not paused
            if loaded_any and hasattr(self, 'rehydrate'):
                self.rehydrate()

            return loaded_any

        except Exception as e:
            logger.error(f"Failed to reload tickers from disk: {e}")
            return False

    def rehydrate(self, last_updated: Union[datetime, None] = None) -> bool:
        """Fetch missing candles since last update and subscribe to live feeds"""
        if not last_updated:
            # check if last_updated can be found from meta_data.
            meta_data = self.load_metadata()

            if 'last_updated' not in meta_data:
                logger.warning(
                    f"last_updated missing in metadata. Rehydration skipped")
                return False

            last_updated = meta_data.get('last_updated')
            logger.debug(
                f"Found last Updated in metadata: {last_updated}({last_updated.astimezone()})")

        current_time = datetime.now(timezone.utc)
        # Only sync if there's a significant gap
        logger.info(f"Rehydrating tickers: {self._tickers.keys()}")

        for (ticker, interval) in self._tickers.keys():
            # 3600:  # 1 hour threshold
            # if (current_time - last_updated).total_seconds() > get_interval_timedelta(interval).seconds:

            if self.data_source:
                try:
                    df = self._tickers[(ticker, interval)]
                    # Check if the DataFrame exists and is not empty
                    if df is not None and not df.empty:
                        # Get the last timestamp from the DataFrame's index.
                        # This will be used as the start_datetime for fetching new data.
                        last_updated = df.index[-1]

                    missing_df = self.data_source.fetch_data(
                        ticker=ticker,
                        interval=interval,
                        start_datetime=last_updated,
                        callback_url=get_callback_url(ticker, interval)
                    )
                    logger.info(
                        f"{ticker}/{interval}: Fetched {len(missing_df)} missing candles")

                    self.running = True
                    # Process missing candles through strategy logic
                    # for timestamp, row in missing_df.iterrows():
                    #     ohlc = {
                    #         "Datetime": timestamp.isoformat(),
                    #         "open": row["Open"],
                    #         "high": row["High"],
                    #         "low": row["Low"],
                    #         "close": row["Close"]
                    #     }
                    # self.on_new_candles(ticker, interval, [ohlc])
                    self.update_new_candles(ticker, interval, missing_df)

                except DataSourceError as e:
                    logger.error(
                        f"Error in fetching missing data and subscribing. {e}")
                    pass
            else:
                logger.warning(
                    "datasource instance missing. Rehydration skipped.")
                return False
            # else:
            #     logger.info(
            #         f"Time Elapsed: {(current_time - last_updated).total_seconds()} seconds too low for {ticker}/{interval}. Skipped Rehydration")

        return True

    def remove_ticker(self, ticker: str, interval: str):
        """
        Remove a specific ticker/interval from this strategy instance
        and delete its data from disk.
        """
        key = (ticker, interval)
        if key in self._tickers:
            del self._tickers[key]
            self.delete_from_disk(ticker, interval)

            # remove from reverse map
            if key in _TICKER_INTERVAL_MAP:
                _TICKER_INTERVAL_MAP[key].remove(self)
                if not _TICKER_INTERVAL_MAP[key]:
                    del _TICKER_INTERVAL_MAP[key]

    # --- accessors for signals/dfs ---
    def get_signals_df(self) -> Optional[pd.DataFrame]:
        return self._signals if not self._signals.empty else None

    def get_ticker_df(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        key = (ticker, interval)
        return self._tickers.get(key) if key in self._tickers else None

    @abstractmethod
    def get_last_signal(self) -> SignalResponse:
        '''
        Return the last signal generated by the strategy.
        The conctrete strategy class must implement this method.
        '''
        pass

    @abstractmethod
    def get_all_signals(self, offset: int = 0, limit: Optional[int] = None, type: Optional[str] = None) -> List[SignalResponse]:
        '''
        Return all signals for the given ticker and interval.
        The conctrete strategy class must implement this method.
        '''
        pass

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
    for meta_file in storage.glob("*/meta.json"):
        try:
            # strategy_dir = meta_file.parent  # pass the folder, not meta.json itself
            instance = Strategy.reload_from_metadata(meta_file, storage)

            # Delegate loading to the strategy class
            # instance = Strategy.reload_from_metadata(entry, storage)

            if instance:
                # Register the instance
                add_strategy_instance(instance)

                reloaded_count += 1
                logger.info(f"Reloaded strategy {instance.strategy_id}")
            else:
                logger.warning("Failed")
                failed_count += 1

        except Exception as e:
            logger.error(
                f"Failed to process metadata file {meta_file.name}: {e}")
            failed_count += 1

    logger.info(
        f"Reload complete: {reloaded_count} strategies loaded, {failed_count} failed")

from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from typing import List

from strategy_base import (
    get_strategy_class,
    list_strategy_classes,
    add_strategy_instance,
    get_strategy_instance,
    list_strategy_instances,
    autodiscover_strategies,
    reload_all_instances
)
from schema.strategy_schema import (
    StrategyCreateRequest,
    CandleRequest,
    SignalResponse
)

import logging

# Basic global logging configuration (optional, but good practice for overall application)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    autodiscover_strategies()
    reload_all_instances()
    yield
    # Shutdown (optional cleanup)
    # e.g., close DB connections, stop background workers

app = FastAPI(title="Strategy API", lifespan=lifespan)


@app.post("/strategies")
def create_strategy(req: StrategyCreateRequest):
    cls = get_strategy_class(req.strategy)
    if not cls:
        raise HTTPException(
            status_code=400, detail=f"Unknown strategy Type {req.strategy}")

    key = (req.strategy, req.ticker, req.interval)

    if get_strategy_instance(key) and not req.restart:
        raise HTTPException(
            status_code=400, detail="Strategy instance already exists")

    instance = cls(req.ticker, req.interval, req.params)
    if not req.restart or not instance.load_from_disk():
        instance.initialize()
        instance.save_to_disk()

    add_strategy_instance(key, instance)
    return {"message": f"Strategy {req.strategy} created for {req.ticker} {req.interval}"}


@app.post(
    "/strategies/{strategy_name}/{ticker}/{interval}/candle",
    response_model=SignalResponse
)
def feed_candle(strategy_name: str, ticker: str, interval: str, candle: CandleRequest):
    key = (strategy_name, ticker, interval)
    instance = get_strategy_instance(key)
    if not instance:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Feed new candle and auto-save
    instance.on_new_candle(candle.model_dump())

    # Return the latest summarized signal
    return instance.get_last_signal()


@app.get("/strategies")
def list_strategies():
    return {"active_strategies": list_strategy_instances()}


@app.get("/strategies/available")
def list_available_strategies():
    return {"available_strategies": list_strategy_classes()}


@app.get(
    "/strategies/{strategy_name}/{ticker}/{interval}/last-signal",
    response_model=SignalResponse
)
def get_last_signal(strategy_name: str, ticker: str, interval: str):
    key = (strategy_name, ticker, interval)
    instance = get_strategy_instance(key)
    if not instance:
        raise HTTPException(status_code=404, detail="Strategy not found")

    return instance.get_last_signal()


@app.get("/strategies/{strategy_name}/{ticker}/{interval}/signals", response_model=List[SignalResponse])
def get_all_signals(
    strategy_name: str,
    ticker: str,
    interval: str,
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000,
                       description="Number of records to return"),
):
    key = (strategy_name, ticker, interval)
    instance = get_strategy_instance(key)
    if not instance:
        raise HTTPException(status_code=404, detail="Strategy not found")
    logger.info(f"Found Instance of {instance.strategy_name}")
    if instance.signals is None or instance.signals.empty:
        return []

    return instance.get_all_signals(limit=limit, offset=offset)


@app.delete("/strategies/{strategy}/{ticker}/{interval}")
def delete_strategy(strategy: str, ticker: str, interval: str):
    key = (strategy, ticker, interval)
    instance = get_strategy_instance(key)

    if not instance:
        raise HTTPException(status_code=404, detail="Strategy instance not found")

    # Delete from disk
    instance.delete_from_disk()

    # Remove from memory
    if key in _STRATEGY_INSTANCES:
        del _STRATEGY_INSTANCES[key]

    return {"message": f"✅ Strategy {strategy} {ticker} {interval} deleted successfully"}
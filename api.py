from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from typing import List, Tuple

import strategy_base
import schema.strategy_schema as strategy_schema

import logging

# Basic global logging configuration (optional, but good practice for overall application)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    strategy_base.autodiscover_strategies()
    strategy_base.reload_all_instances()
    yield
    # Shutdown (optional cleanup)
    # e.g., close DB connections, stop background workers

app = FastAPI(title="Strategy API", lifespan=lifespan)


@app.post("/strategies")
def create_strategy(req: strategy_schema.CreateStrategyRequest):
    cls = strategy_base.get_strategy_class(req.strategy_name)
    if not cls:
        raise HTTPException(
            status_code=400, detail=f"Unknown strategy Type {req.strategy_name}"
        )

    # Instantiate with params
    instance = cls(params=req.params)

    # Register all (ticker, interval) pairs
    for ticker, intervals in req.tickers.items():
        for interval in intervals:
            instance.ensure_ticker(ticker, interval)
            # initialize ticker-specific data instead of whole strategy
            instance.initialize_ticker(ticker, interval, req.params)

    # Save instance to registry
    strategy_base.add_strategy_instance(
        strategy_name=req.strategy_name,
        params=req.params,
        instance=instance
    )

    return {
        "message": "Strategy created",
        "strategy": req.strategy_name,
        "params": req.params,
        "tickers": req.tickers,
    }


@app.get("/strategies/available", response_model=List[str])
def list_available_strategies():
    return {"available_strategies": strategy_base.list_strategy_classes()}


@app.get("/strategies/instances", response_model=List[Tuple[str, str]])
def list_strategy_instances():
    return {"active_strategies": strategy_base.list_strategy_instances()}


@app.post(
    "/strategies/{ticker}/{interval}/candle",
    response_model=List[strategy_schema.SignalResponse]
)
def feed_candle(
    ticker: str,
    interval: str,
    candle: strategy_schema.CandleRequest
):
    """
    Feed a new candle to all strategies that have this ticker+interval.
    """
    instances = strategy_base.get_strategy_instances_by_ticker_interval(
        ticker, interval)
    if not instances:
        raise HTTPException(
            status_code=404, detail="No strategy instance found for this ticker/interval")

    responses = []
    for instance in instances:
        instance.on_new_candle(
            ticker=ticker, interval=interval, ohlc=candle.model_dump())
        responses.append(
            instance.get_last_signal(ticker=ticker, interval=interval)
        )

    # Optionally return the last signal of the first instance
    return responses


@app.get(
    "/strategies/{strategy_id}/{ticker}/{interval}/last-signal",
    response_model=strategy_schema.SignalResponse
)
def get_last_signal(strategy_id: str, ticker: str, interval: str):
    inst = strategy_base.get_strategy_instance_by_id(strategy_id)
    if not inst:
        raise HTTPException(
            status_code=404, detail="Strategy instance not found")
    return inst.get_last_signal(ticker, interval)


@app.get(
    "/strategies/{strategy_name}/{ticker}/{interval}/signals",
    response_model=List[strategy_schema.SignalResponse]
)
def get_all_signals(
    strategy_id: str,
    ticker: str,
    interval: str,
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000,
                       description="Number of records to return"),
):
    inst = strategy_base.get_strategy_instance_by_id(strategy_id)
    if not inst:
        raise HTTPException(
            status_code=404, detail="Strategy instance not found")
    return inst.get_all_signals(ticker, interval, offset=offset, limit=limit)


@app.delete("/strategies/{strategy_name}")
def delete_strategy_instance(strategy_name: str):
    """
    Delete the full strategy instance and all its tickers/intervals.
    """
    instances = strategy_base.get_strategy_instances_by_name(strategy_name)
    if not instances:
        raise HTTPException(status_code=404, detail="Strategy not found")

    for instance in instances:
        instance.delete_from_disk()  # deletes all tickers + meta
        strategy_base.remove_strategy_instance(
            instance)  # remove from registry

    return {"message": f"All instances of {strategy_name} deleted"}


@app.delete("/strategies/{strategy_name}/{ticker}/{interval}")
def delete_strategy_ticker(strategy_name: str, ticker: str, interval: str):
    """
    Delete only the specified ticker/interval for a strategy instance.
    """
    instances = strategy_base.get_strategy_instances_by_name(strategy_name)
    if not instances:
        raise HTTPException(status_code=404, detail="Strategy not found")

    deleted = False
    for instance in instances:
        if (ticker, interval) in instance._signals:
            instance.remove_ticker(ticker, interval)
            deleted = True

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No active ticker {ticker} with interval {interval} found"
        )

    return {
        "message": f"Ticker {ticker} with interval {interval} removed from {strategy_name}"
    }

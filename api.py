from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from typing import List, Any

import strategy_base
import schema.strategy_schema as strategy_schema
import threading

import logging

# Basic global logging configuration (optional, but good practice for overall application)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s -  %(funcName)s():%(lineno)d - %(message)s')

logger = logging.getLogger(__name__)

reload_complete = False


def reload_with_status():
    global reload_complete
    reload_complete = False
    strategy_base.reload_all_instances()
    reload_complete = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    strategy_base.autodiscover_strategies()
    threading.Thread(target=reload_with_status, daemon=True).start()
    yield
    # Shutdown (optional cleanup)
    # e.g., close DB connections, stop background workers

app = FastAPI(title="Strategy API", lifespan=lifespan)


@app.post("/strategies")
def create_strategy_instance(req: strategy_schema.CreateStrategyRequest):
    if not reload_complete:
        raise HTTPException(
            status_code=503,
            detail=f"Reloading in progress. Please try after sometime"
        )
    cls = strategy_base.get_strategy_class(req.strategy_name)
    if not cls:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy Type {req.strategy_name}"
        )

    try:
        # Let the strategy class validate its own requirements
        cls.validate_creation_request(req.tickers, req.params)

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
            instance=instance
        )

        return {
            "message": "Strategy created",
            "strategy_name": req.strategy_name,
            "strategy_id": instance.strategy_id,
            "params": req.params,
            "tickers": req.tickers,
        }
    except ValueError as e:
        logger.error(f"Failed to create strategy {req.strategy_name}: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as ex:
        logger.error(f"Failed to create strategy {req.strategy_name}: {ex}")
        raise HTTPException(
            status_code=500,
            detail=f"Strategy creation failed: {str(ex)}"
        )


@app.get("/strategies/available", response_model=dict)
def list_available_strategies():
    return {"available_strategies": strategy_base.list_strategy_classes()}


@app.get("/strategies/instances", response_model=dict)
def list_strategy_instances():
    if not reload_complete:
        raise HTTPException(
            status_code=503,
            detail=f"Reloading in progress. Please try after sometime"
        )

    return {"active_strategies": strategy_base.list_strategy_instances()}


@app.post(
    "/strategies/{ticker}/{interval}/candle",
    response_model=List[Any]
)
def feed_candle(
    ticker: str,
    interval: str,
    candle: strategy_schema.CandleRequest
):
    """
    Feed a new candle to all strategies that have this ticker+interval.
    """
    if not reload_complete:
        raise HTTPException(
            status_code=503,
            detail=f"Reloading in progress. Please try after sometime"
        )

    if not ticker or not interval:
        raise HTTPException(
            status_code=400,
            detail="Ticker and interval must be non-empty"
        )

    instances = strategy_base.get_strategy_instances_by_ticker_interval(
        ticker, interval)
    if not instances:
        raise HTTPException(
            status_code=404, detail="No strategy instance found for this ticker/interval")

    responses = []
    errors = []
    for instance in instances:
        try:
            result = instance.on_new_candle(
                ticker=ticker, interval=interval, ohlc=candle.model_dump())
            # Check if strategy returned an error

            if isinstance(result, dict) and "error" in result:
                errors.append(f"{instance.strategy_id}: {result['error']}")
                continue

            signal = instance.get_last_signal(ticker=ticker, interval=interval)
            responses.append(signal)
        except Exception as e:
            logger.error(
                f"Strategy {instance.strategy_id} {candle} failed to process candle: {e}")
            errors.append(f"{instance.strategy_id}: {str(e)}")

    if errors and not responses:
        raise HTTPException(
            status_code=500,
            detail=f"All strategies failed: {'; '.join(errors)}"
        )

    return responses


@app.get(
    "/strategies/{strategy_id}/{ticker}/{interval}/last-signal",
    response_model=strategy_schema.SignalResponse
)
def get_last_signal(strategy_id: str, ticker: str, interval: str):
    if not reload_complete:
        raise HTTPException(
            status_code=503,
            detail=f"Reloading in progress. Please try after sometime"
        )
    inst = strategy_base.get_strategy_instance_by_id(strategy_id)
    if not inst:
        raise HTTPException(
            status_code=404, detail="Strategy instance not found")
    return inst.get_last_signal(ticker, interval)


@app.get(
    "/strategies/{strategy_id}/{ticker}/{interval}/signals",
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
    if not reload_complete:
        raise HTTPException(
            status_code=503,
            detail=f"Reloading in progress. Please try after sometime"
        )

    inst = strategy_base.get_strategy_instance_by_id(strategy_id)
    if not inst:
        raise HTTPException(
            status_code=404, detail="Strategy instance not found")
    return inst.get_all_signals(ticker, interval, offset=offset, limit=limit)


@app.delete("/strategies/{strategy_id}")
def delete_strategy_instance(strategy_id: str):
    """
    Delete the full strategy instance and all its tickers/intervals.
    """
    if not reload_complete:
        raise HTTPException(
            status_code=503,
            detail=f"Reloading in progress. Please try after sometime"
        )

    instance = strategy_base.get_strategy_instance_by_id(strategy_id)
    if not instance:
        raise HTTPException(
            status_code=404, detail="Strategy Instance not found")

    instance.delete_from_disk()  # deletes all tickers + meta
    strategy_base.remove_strategy_instance(
        instance)  # remove from registry

    return {"message": f"Strategy {strategy_id} deleted"}


@app.delete("/strategies/{strategy_id}/{ticker}/{interval}")
def delete_strategy_ticker(strategy_id: str, ticker: str, interval: str):
    """
    Delete only the specified ticker/interval for a strategy instance.
    """
    if not reload_complete:
        raise HTTPException(
            status_code=503,
            detail=f"Reloading in progress. Please try after sometime"
        )
    instance = strategy_base.get_strategy_instance_by_id(strategy_id)
    if not instance:
        raise HTTPException(
            status_code=404, detail="Strategy Instance not found")

    deleted = False
    if (ticker, interval) in instance._signals:
        instance.remove_ticker(ticker, interval)
        deleted = True

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No active ticker {ticker} with interval {interval} found"
        )

    return {
        "message": f"Ticker {ticker} with interval {interval} removed from {strategy_id}"
    }


@app.put("/strategies/{strategy_id}/restart")
def restart_strategy_instance(strategy_id: str):
    logger.info(f"Received strategy_id: '{strategy_id}'")
    """
    Delete the full strategy instance and all its tickers/intervals.
    """
    if not reload_complete:
        raise HTTPException(
            status_code=503,
            detail=f"Reloading in progress. Please try after sometime"
        )

    instance = strategy_base.get_strategy_instance_by_id(strategy_id)
    if not instance:
        raise HTTPException(
            status_code=404, detail="Strategy Instance not found")

    try:
        success, message = instance.restart()
        if success:
            return {
                "message": message,
                "strategy_id": strategy_id,
                "status": instance.get_status()
            }
        else:
            raise HTTPException(
                status_code=403, detail=message
            )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in rehydrating: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error in rehydrating: {e}")


@app.put(
    "/strategies/pause",
    response_model=strategy_schema.StrategiesPauseResponse
)
def pause_strategy_instances(req: strategy_schema.StrategiesPauseRequest):
    """
    Resume the strategy instance
    """
    if not reload_complete:
        raise HTTPException(
            status_code=503,
            detail=f"Reloading in progress. Please try after sometime"
        )

    response = []
    strategy_ids_to_pause = req.strategy_ids or []
    if not strategy_ids_to_pause:
        return {
            'message': "No Strategy Ids specified"
        }

    for strategy_id in strategy_ids_to_pause:
        instance = strategy_base.get_strategy_instance_by_id(strategy_id)
        if not instance:
            # raise HTTPException(
            #     status_code=404, detail="Strategy Instance not found")
            response.append(
                strategy_schema.StrategyWithStatus(
                    strategy_id=strategy_id,
                    status="Strategy Instance not found"
                )
            )
            continue

        try:
            success, message = instance.pause()
            response.append(
                strategy_schema.StrategyWithStatus(
                    strategy_id=strategy_id,
                    status=instance.get_status() if success else message
                )
            )
        except Exception as e:
            response.append(
                strategy_schema.StrategyWithStatus(
                    strategy_id=strategy_id,
                    status=f"{e}"
                )
            )
    return {
        'message': "Success",
        'detail': response
    }


@app.put(
    "/strategies/resume",
    response_model=strategy_schema.StrategiesResumeResponse
)
def resume_strategy_instances(req: strategy_schema.StrategiesResumeRequest):
    """
    Resume the strategy instance
    """
    if not reload_complete:
        raise HTTPException(
            status_code=503,
            detail=f"Reloading in progress. Please try after sometime"
        )

    response = []
    strategy_ids_to_resume = req.strategy_ids or []
    if not strategy_ids_to_resume:
        return {
            'message': "No Strategy Ids specified"
        }

    for strategy_id in strategy_ids_to_resume:
        instance = strategy_base.get_strategy_instance_by_id(strategy_id)
        if not instance:
            # raise HTTPException(
            #     status_code=404, detail="Strategy Instance not found")
            response.append(
                strategy_schema.StrategyWithStatus(
                    strategy_id=strategy_id,
                    status="Strategy Instance not found"
                )
            )
            continue

        try:
            success, message = instance.resume()
            response.append(
                strategy_schema.StrategyWithStatus(
                    strategy_id=strategy_id,
                    status=instance.get_status() if success else message
                )
            )
        except Exception as e:
            response.append(
                strategy_schema.StrategyWithStatus(
                    strategy_id=strategy_id,
                    status=f"{e}"
                )
            )
    return {
        'message': "Success",
        'detail': response
    }

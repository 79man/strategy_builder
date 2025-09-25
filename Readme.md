# Strategy Builder

**A Python-based framework for building, backtesting, and deploying automated trading strategies with robust data handling, modular strategy definition, and RESTful APIs.**

## Features

- **Modular Strategy Architecture**: Easily define and register new trading strategies as Python classes.
- **Plug-and-Play Data Sources**: Integrate multiple data sources (e.g., Yahoo Finance, local feeds, custom services).
- **REST API**: Manage strategies, feed candles, fetch signals, and operate via FastAPI endpoints.
- **Persistence**: Strategies and their signals are saved per instance and ticker/interval for resilience and rehydration.
- **Scalable Management**: Multiple strategies, tickers, and intervals can be managed and backtested in parallel.
- **Robust Validation**: Thorough schema and parameter checks for safe and reproducible execution.

***

## Directory Structure

```
strategy_builder/
│
├── api.py                # FastAPI endpoints for the platform
├── run.py                # App entrypoint (uses Uvicorn)
├── strategy_base.py      # Core Strategy class, registry, instance management
├── data_source.py        # DataSource base class and error classes
├── requirements.txt      # Python dependencies
├── .env.example          # Example configuration file
│
├── data_sources/         # Concrete data source implementations (e.g., yfinance, local)
├── schema/               # Pydantic models and request/response schemas
├── strategies/           # User-defined strategies (template: inherit from Strategy)
├── utils/                # Configuration, intervals, helpers, callback URL logic
└── ...
```

***

## Getting Started

### Prerequisites

- Python 3.9+
- (Optional) Docker (for containerized runs)

### Installation

```bash
# Clone the repository
git clone https://github.com/79man/strategy_builder.git
cd strategy_builder

# (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate (Windows)

# Install Python dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the example environment file and update variables as needed.
   ```bash
   cp .env.example .env
   ```
2. Adjust values in `.env`:
    - `DATA_FEED_BASE_URL`: URL for your candle data provider or broker API
    - `CALLBACK_BASE_URL`: URL for candle feed callbacks to reach the strategy builder server
    - `STORAGE_PATH`: Path where strategy and signal data will be persisted

***

## Running the API Server

```bash
python run.py
```

*By default, the API is served on `http://0.0.0.0:8000` as defined in the config.*

***

## API Endpoints

### Create Strategy Instance

```
POST /strategies
```

Create a new strategy instance.

**Example Request Body:**
```json
{
  "strategy_name": "MACDStrategy",
  "params": { "macd_fast": 12, "macd_slow": 26 },
  "tickers": { "NIFTY": ["15m", "1h"] }
}
```

**Example Response:**

```json
{
  "message": "Strategy created",
  "strategy_name": "MACDStrategy",
  "strategy_id": "MACDStrategy:643212",
  "params": { "macd_fast": 12, "macd_slow": 26 },
  "tickers": { "NIFTY": ["15m", "1h"] }
}
```


***

### List Available Strategy Classes

```
GET /strategies/available
```

Returns a list of available strategy names.

**Example Response:**

```json
{
  "available_strategies": ["MACDStrategy", "ATRTrendFollower"]
}
```


***

### List Active Strategy Instances
```
GET /strategies/instances
```

List all currently active strategy instances.

**Example Response:**

```json
{
  "active_strategies": [
    {
      "strategy_id": "MACDStrategy:643212",
      "strategy_name": "MACDStrategy",
      "params": { "macd_fast": 12, "macd_slow": 26 },
      "tickers": { "NIFTY": ["15m", "1h"] }
    }
  ]
}
```


***

### Feed Candle Data to Strategies

```
POST /strategies/{ticker}/{interval}/candle
```
**Example Request Body:**

```json
{
  "datetime": "2025-09-18T10:15:00Z",
  "open": 22000.50,
  "high": 22040.75,
  "low": 21988.25,
  "close": 22031.40
}
```

**Example Response:**

```json
[
  {
    "ticker": "NIFTY",
    "interval": "15m",
    "strategy_id": "MACDStrategy:643212",
    "datetime": "2025-09-18T10:15:00Z",
    "signal": "BUY",
    "indicators": { "macd": 2.45, "signal_above_macd": true },
    "message": "MACD crossed above signal line"
  }
]
```


***

### Get Last Signal for a Strategy

```
GET /strategies/{strategy_id}/{ticker}/{interval}/last-signal
```

**Example Response:**

```json
{
  "ticker": "NIFTY",
  "interval": "15m",
  "strategy_id": "MACDStrategy:643212",
  "datetime": "2025-09-18T10:15:00Z",
  "signal": "BUY",
  "indicators": { "macd": 2.45, "signal_above_macd": true },
  "message": "MACD crossed above signal line"
}
```


***

### Get All Signals (Paginated)

```
GET /strategies/{strategy_id}/{ticker}/{interval}/signals?offset=0&limit=2
```

**Example Response:**

```json
[
  {
    "ticker": "NIFTY",
    "interval": "15m",
    "strategy_id": "MACDStrategy:643212",
    "datetime": "2025-09-18T10:15:00Z",
    "signal": "BUY",
    "indicators": { "macd": 2.45, "signal_above_macd": true },
    "message": "MACD crossed above signal line"
  },
  {
    "ticker": "NIFTY",
    "interval": "15m",
    "strategy_id": "MACDStrategy:643212",
    "datetime": "2025-09-18T10:00:00Z",
    "signal": "SELL",
    "indicators": { "macd": -1.13, "signal_above_macd": false }
  }
]
```


***

### Delete Full Strategy Instance

```
DELETE /strategies/{strategy_id}
```

**Example Response:**

```json
{ "message": "Strategy MACDStrategy:643212 deleted" }
```


***

### Remove Ticker/Interval from Strategy

```
DELETE /strategies/{strategy_id}/{ticker}/{interval}
```

**Example Response:**

```json
{
  "message": "Ticker NIFTY with interval 15m removed from MACDStrategy:643212"
}
```


***

### Restart, Pause, Resume Strategy Instance

```
PUT /strategies/{strategy_id}/restart
PUT /strategies/{strategy_id}/pause
PUT /strategies/{strategy_id}/resume
```

**Example Response:**

```json
{
  "message": "Strategy restarted",
  "strategy_id": "MACDStrategy:643212",
  "status": "running"
}
```

**All requests use JSON payloads, and all responses are JSON.**

***

## Example: Strategy Definition

To implement a new trading strategy, subclass `Strategy` and place your class in the `strategies/` folder:

```python
from strategy_base import Strategy

class MyAwesomeStrategy(Strategy):
    strategy_name = "MyAwesomeStrategy"

    @classmethod
    def validate_creation_request(cls, tickers, params=None):
        # Validate specific strategy arguments
        super().validate_creation_request(tickers, params)
        # Add custom argument checks here

    def initialize_ticker(self, ticker, interval, params, start_datetime=None):
        # Fetch and initialize candle data, compute indicators

    def _on_new_candle(self, ticker, interval, ohlc):
        # Custom per-candle logic returning indicator/data dict
```

See `strategies/heikin_ashi.py` and `strategies/pair_trading_strategy.py` for real examples.

***

## Data Source Integrations

- **LocalTickerDataSource** (default, connects to external/local data feed API)
- **YFinanceSource** (fetches from Yahoo Finance, for backtests/demo)

Plug them into strategies via constructor or override in your subclass.

***

## Environment Variables Reference

See `.env.example` for the full set:
- `DATA_FEED_BASE_URL` — The HTTP endpoint for OHLCV candle data subscription.
- `CALLBACK_BASE_URL` — Used so the data source can send new candle updates as they arrive.
- `STORAGE_PATH` — Where all signals, metadata, and state are stored.

***

## Dependency List

- `fastapi` — REST API interface
- `uvicorn` — ASGI server
- `pandas` — Data manipulation/analysis
- `python-dotenv`, `pydantic-settings` — Configuration and schema management

***

## Developing & Contributing

- Fork, branch, and submit pull requests!
- Use descriptive commit messages.
- Add/extend strategies in `strategies/`, data sources in `data_sources/`.
- Ensure your code passes basic linting/tests.

***

## License

MIT License (unless otherwise specified).

***

## Authors & Maintainers

- Created by [79man](https://github.com/79man)
- Contributions welcome via pull request!

***

**Note:** This project is in active development. Features, APIs, and modules may evolve rapidly. See commit history for recent changes, and raise issues/PRs for bugs or improvements.

***

For technical help or questions, please open an issue on GitHub or contact the repository owner.

[1](https://github.com/79man/strategy_builder)
[2](https://github.com/79man/strategy_builder/blob/master/api.py)
[3](https://github.com/79man/strategy_builder/blob/master/run.py)
[4](https://github.com/79man/strategy_builder/blob/master/strategy_base.py)
[5](https://github.com/79man/strategy_builder/blob/master/data_source.py)
[6](https://github.com/79man/strategy_builder/blob/master/requirements.txt)
[7](https://github.com/79man/strategy_builder/blob/master/.env.example)
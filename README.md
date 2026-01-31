# ETH Price Forecast (Deep Learning)

Proof-of-concept for forecasting one step ahead of the ETH price using a causal Conv1D + LSTM model with z-score normalization.

## Features

- Loads price data from `eth.json`.
- Trains a causal Conv1D + LSTM model.
- Produces plots for the series, training loss, and validation forecast.
- Saves metrics (`mse`, `mae`, `mean`, `std`) to `metrics.pkl`.

## Data Format

`eth.json` must contain a `prices` array with pairs of `[timestamp_ms, price]`:

```json
{
  "prices": [
    [1700000000000, 2100.50],
    [1700000060000, 2102.10]
  ]
}
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Outputs

- Training and forecast plots displayed in windows.
- `metrics.pkl` saved in the project root.

## Configuration

Key parameters are defined in `Config` inside `app.py`:

- `window_size`
- `batch_size`
- `epochs`
- `train_ratio`

## Notes

- This is a minimal POC and not production-ready.
- If you have very few data points, reduce `window_size` or provide more data.

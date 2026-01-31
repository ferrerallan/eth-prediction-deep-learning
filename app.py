
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Config:
    in_json: Path = Path("eth.json")
    window_size: int = 16
    batch_size: int = 32
    shuffle_buffer_size: int = 50
    epochs: int = 100
    train_ratio: float = 0.8
    metrics_pkl: Path = Path("metrics.pkl")


CFG = Config()


def plot_series(time: np.ndarray, series: np.ndarray, fmt: str = "-", start: int = 0, end: int | None = None) -> None:
    """Plot a time series slice."""
    plt.plot(time[start:end], series[start:end], fmt)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def load_eth_prices(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads eth.json and returns:
      - TIME: 0..N-1
      - SERIES: preços (float32)
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {json_path.resolve()}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    prices = payload.get("prices")
    if not prices:
        raise ValueError("JSON invalid: 'prices' key not found or empty.")

    if not isinstance(prices, list) or len(prices[0]) != 2:
        raise ValueError("'prices' must be a list of pairs [timestamp_ms, price].")

    series = np.asarray([p[1] for p in prices], dtype=np.float32)
    time = np.arange(series.shape[0], dtype=np.float32)
    return time, series


def train_val_split(
    time: np.ndarray, series: np.ndarray, split_time: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train/valid by index."""
    return time[:split_time], series[:split_time], time[split_time:], series[split_time:]


def windowed_dataset(series: np.ndarray, window_size: int, batch_size: int, shuffle_buffer_size: int) -> tf.data.Dataset:
    """
    Windowed dataset:
      X: (window_size, 1)
      y: (1,)
    """
    series = tf.convert_to_tensor(series, dtype=tf.float32)
    series = tf.expand_dims(series, axis=-1)  # (N, 1)

    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def create_uncompiled_model(window_size: int) -> tf.keras.Model:
    """Conv1D causal + LSTM stack for 1-step ahead forecast."""
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(window_size, 1)),
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=5,
                strides=1,
                padding="causal",
                activation="relu",
            ),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1),
        ]
    )


def create_model(window_size: int) -> tf.keras.Model:
    """Create and compile the model."""
    model = create_uncompiled_model(window_size)
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["mae"],
    )
    return model


def model_forecast(model: tf.keras.Model, series: np.ndarray, window_size: int) -> np.ndarray:
    """Vectorized forecast via tf.data."""
    series = tf.convert_to_tensor(series, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.map(lambda w: tf.expand_dims(w, axis=-1))  # (window_size, 1)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    forecast = model.predict(ds, verbose=0)
    return forecast.squeeze()


def compute_metrics(true_series: np.ndarray, forecast: np.ndarray) -> tuple[float, float]:
    """Returns (mse, mae) as floats."""
    true_series = tf.convert_to_tensor(true_series, dtype=tf.float32)
    forecast = tf.convert_to_tensor(forecast, dtype=tf.float32)
    mse = tf.reduce_mean(tf.keras.losses.MSE(true_series, forecast)).numpy().item()
    mae = tf.reduce_mean(tf.keras.losses.MAE(true_series, forecast)).numpy().item()
    return float(mse), float(mae)


def main() -> None:
    print(f"[1/6] Reading JSON: {CFG.in_json}")
    time, series = load_eth_prices(CFG.in_json)
    print(f"[2/6] Points: {len(series)}")

    if len(series) < CFG.window_size + 10:
        raise ValueError(f"Dataset too short ({len(series)} points). Reduce window_size or use more data    .")

    split_time = int(len(series) * CFG.train_ratio)
    split_time = max(split_time, CFG.window_size + 1)
    print(f"[3/6] split_time={split_time} (train_ratio={CFG.train_ratio})")

    time_train, series_train, time_valid, series_valid = train_val_split(time, series, split_time)

    mean = float(series_train.mean())
    std = float(series_train.std() + 1e-8)

    series_train_norm = (series_train - mean) / std
    series_norm = (series - mean) / std

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.title("ETH - Series (price)")
    plt.show()

    print("[4/6] Creating dataset and training...")
    train_ds = windowed_dataset(series_train_norm, CFG.window_size, CFG.batch_size, CFG.shuffle_buffer_size)

    model = create_model(CFG.window_size)
    history = model.fit(train_ds, epochs=CFG.epochs, verbose=1)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.title("Training loss")
    plt.legend(loc=0)
    plt.show()

    print("[5/6] Generating forecast...")
    forecast_input = series_norm[split_time - CFG.window_size : -1]
    forecast_norm = model_forecast(model, forecast_input, CFG.window_size)
    forecast_norm = forecast_norm[: len(series_valid)]
    forecast = forecast_norm * std + mean

    plt.figure(figsize=(10, 6))
    plot_series(time_valid, series_valid)
    plot_series(time_valid, forecast)
    plt.title("Validation vs Forecast (1 step ahead)")
    plt.show()

    mse, mae = compute_metrics(series_valid, forecast)
    print(f"[6/6] mse={mse:.2f} | mae={mae:.2f}")

    with open(CFG.metrics_pkl, "wb") as f:
        pickle.dump({"mse": mse, "mae": mae, "mean": mean, "std": std}, f)

    print(f"OK: metrics saved to {CFG.metrics_pkl.resolve()}")


if __name__ == "__main__":
    main()

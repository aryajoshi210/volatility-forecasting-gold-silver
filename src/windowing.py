"""Turns a return series into fixed-length windowed examples for the transformer."""

import numpy as np
import pandas as pd


def create_windows(returns, window_size):
    """For each position, takes `window_size` consecutive returns as input (X)
    and the very next day's absolute return as the target (y)."""
    values = returns.values
    X, y, dates = [], [], []
    for start in range(len(values) - window_size):
        end = start + window_size
        X.append(values[start:end])
        y.append(abs(values[end]))
        dates.append(returns.index[end])
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        pd.DatetimeIndex(dates),
    )

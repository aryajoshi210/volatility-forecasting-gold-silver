"""Tests for src/windowing.py - the sliding-window logic used to build training examples."""

import numpy as np
import pandas as pd
import pytest

from windowing import create_windows


@pytest.fixture
def returns():
    values = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0]
    dates = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates)


def test_shapes(returns):
    X, y, dates = create_windows(returns, window_size=5)
    # 10 values, window_size=5 -> 5 examples (10 - 5)
    assert X.shape == (5, 5)
    assert y.shape == (5,)
    assert len(dates) == 5


def test_first_window_matches_expected_slice(returns):
    X, y, dates = create_windows(returns, window_size=5)
    expected_first_window = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float32)
    np.testing.assert_allclose(X[0], expected_first_window)


def test_target_is_the_day_immediately_after_the_window(returns):
    X, y, dates = create_windows(returns, window_size=5)
    # window 0 covers values[0:5]; the target should be abs(values[5]) = abs(-0.6) = 0.6
    assert y[0] == pytest.approx(0.6)
    # target dates should be the day right after each window ends
    assert dates[0] == returns.index[5]


def test_targets_are_never_negative(returns):
    _, y, _ = create_windows(returns, window_size=5)
    assert (y >= 0).all()


def test_window_never_contains_its_own_target_date(returns):
    """Guards against a lookahead bug: a window's date range must end strictly before its target date."""
    X, y, dates = create_windows(returns, window_size=5)
    for i in range(len(X)):
        window_dates = returns.index[i : i + 5]
        target_date = dates[i]
        assert target_date not in window_dates
        assert window_dates[-1] < target_date

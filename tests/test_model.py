"""Tests for src/model.py - the VolatilityTransformer architecture."""

import torch

from model import VolatilityTransformer


def test_output_shape_matches_batch_size():
    model = VolatilityTransformer()
    batch = torch.randn(8, 30, 1)  # 8 windows, 30 days each, 1 feature per day
    output = model(batch)
    assert output.shape == (8,)


def test_output_is_always_positive():
    """Volatility can't be negative - the softplus output head must guarantee this,
    even for input the model has never been trained on."""
    model = VolatilityTransformer()
    batch = torch.randn(20, 30, 1) * 100  # deliberately extreme, untrained-for input
    output = model(batch)
    assert (output > 0).all()


def test_handles_a_single_example():
    model = VolatilityTransformer()
    batch = torch.randn(1, 30, 1)
    output = model(batch)
    assert output.shape == (1,)


def test_default_parameter_count():
    """Locks in the model size - catches accidental architecture changes
    (e.g. forgetting the dim_feedforward=64 fix that avoided a ~280k-parameter model)."""
    model = VolatilityTransformer()
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params == 17185

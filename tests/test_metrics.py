"""Tests for `runn.metrics` module."""
import numpy as np
import pytest

from runn import metrics


@pytest.fixture
def example_proba() -> np.ndarray:
    """Example of predicted probabilities and true choices."""
    proba = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
    return proba


def test_accuracy(example_proba):
    """Test the accuracy metric."""
    y = [2, 1, 2, 0]
    assert metrics.accuracy(example_proba, y) == 0.75


def test_AMPCA(example_proba):
    """Test the AMPCA metric."""
    y = [2, 1, 2, 0]
    assert np.round(metrics.AMPCA(example_proba, y), 6) == 0.733333


def test_CEL(example_proba):
    """Test the CEL metric."""
    y = [2, 1, 2, 0]
    assert np.round(metrics.CEL(example_proba, y), 6) == 0.900027


def test_GMPCA(example_proba):
    """Test the GMPCA metric."""
    y = [2, 1, 2, 0]
    assert np.round(metrics.GMPCA(example_proba, y), 6) == 0.406559

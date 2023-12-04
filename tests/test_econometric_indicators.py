"""Tests for `runn.metrics` module."""
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf

from runn import econometric_indicators


# Define a dummy model
class DummyModel:
    def __init__(self, params: dict = None, filename: str = None) -> None:
        """Dummy model for testing purposes."""
        self.params = params
        self.filename = filename

    def predict(self, x: Union[tf.Tensor, np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """Dummy predict method. Returns the choice probabilities."""
        # Apply a very simple softmax function
        x = np.array(x)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def test_market_shares():
    """Test the market_shares function."""
    # Define a dummy model
    model = DummyModel()

    # Define the input data
    x = np.array([[1, 2, 3], [4, 1, 6], [9, -2, -3], [1, 0, 3], [7, 8, 9]])

    # Compute the market shares
    market_shares = econometric_indicators.market_shares(model, x)

    # Check the result
    assert (np.round(market_shares, 4) == np.array([28.2547, 10.7477, 60.9977])).all()

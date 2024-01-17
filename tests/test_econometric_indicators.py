"""Tests for `runn.metrics` module."""
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.models import Model

from runn import econometric_indicators


# Define a dummy model
class DummyModel:
    def __init__(self, attributes: Optional[list] = None, n_alt: Optional[int] = None, filename: str = None) -> None:
        """Dummy model for testing purposes."""
        self.attributes = attributes
        self.n_alt = n_alt
        self.filename = filename
        self._build()
        self._compile()

    def _build(self):
        """Build the dummy model."""
        inputs = Input(shape=(3,))
        x = Dense(5, activation="relu")(inputs)
        u = Dense(3, activation="linear", name="U")(x)
        outputs = Activation("softmax", name="P")(u)
        self.keras_model = Model(inputs=inputs, outputs=outputs)

        # Set dummy weights
        np.random.seed(42)
        weigths = [np.random.rand(*w.shape) for w in self.keras_model.get_weights()]
        self.keras_model.set_weights(weigths)

    def _compile(self):
        """Compile the dummy model."""
        self.keras_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def predict(self, x: Union[tf.Tensor, np.ndarray, pd.DataFrame]) -> tf.Tensor:
        """Predict the choice probabilities of the alternatives for the given input data."""
        return self.keras_model.predict(x)

    def get_utility(self, x: Union[tf.Tensor, np.ndarray, pd.DataFrame]) -> tf.Tensor:
        """Get the utility of the alternatives for the given input data."""
        utility_model = Model(inputs=self.keras_model.input, outputs=self.keras_model.get_layer("U").output)
        return utility_model(x)


def test_willingness_to_pay():
    """Test the willingness_to_pay function."""
    # Define a dummy model
    params = {"attributes": ["a", "b", "c"], "n_alt": 3}
    model = DummyModel(**params)

    # Define the input data
    x = np.array([[1.0, 2.5, 3], [4, 1, 6], [9, -2.1, -3], [1, 0, 3.4], [7, 8, 9]])
    dummy_data = pd.DataFrame(x, columns=["a", "b", "c"])

    # Compute the WTP using a StandardScaler
    dummy_scaler = StandardScaler()
    dummy_scaler.fit(dummy_data)
    wtp = econometric_indicators.willingness_to_pay(model, dummy_data, "a", "b", 0, dummy_scaler)
    assert np.allclose(wtp, np.array([-2.22872288, -2.22872288, -2.51794411, -2.22872288, -2.22872288]))

    # Compute the WTP using a MinMaxScaler
    # TODO: Implement


def test_value_of_time():
    """Test the value_of_time function."""
    # Define a dummy model
    params = {"attributes": ["a", "b", "c"], "n_alt": 3}
    model = DummyModel(**params)

    # Define the input data
    x = np.array([[1.0, 2.5, 3], [4, 1, 6], [9, -2.1, -3], [1, 0, 3.4], [7, 8, 9]])
    dummy_data = pd.DataFrame(x, columns=["a", "b", "c"])

    # Compute the VOT using a StandardScaler
    dummy_scaler = StandardScaler()
    dummy_scaler.fit(dummy_data)
    vot = econometric_indicators.value_of_time(model, dummy_data, "a", "b", 0, dummy_scaler)
    assert np.allclose(vot, np.array([2.22872288, 2.22872288, 2.51794411, 2.22872288, 2.22872288]))

    # Compute the VOT using a MinMaxScaler
    # TODO: Implement


def test_market_shares():
    """Test the market_shares function."""
    # Define a dummy model
    model = DummyModel()

    # Define the input data
    x = np.array([[1.0, 2.5, 3], [4, 1, 6], [9, -2.1, -3], [1, 0, 3.4], [7, 8, 9]])
    dummy_data = pd.DataFrame(x, columns=["a", "b", "c"])

    # Compute the market shares
    market_shares = econometric_indicators.market_shares(model, dummy_data)

    # Check the result
    assert np.allclose(market_shares, np.array([1.6098, 22.9496, 75.4406]))

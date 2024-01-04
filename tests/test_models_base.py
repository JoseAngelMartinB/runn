"""Tests for `runn.metrics` module."""


import numpy as np
import pytest
import tensorflow as tf

from runn.models.base import BaseModel


# Tests for BaseModel
class TestBaseModel:
    def test_initialization_no_params(self):
        # Test initialization of the BaseModel
        with pytest.raises(ValueError):
            BaseModel()  # Expecting a ValueError for missing parameters

    def test_initialization_wrong_params(self):
        # Test initialization of the BaseModel
        dummy_params = {"attributes": ["a", "b", "c"], "n_alt": 3, "layers_dim": [10, 5]}
        with pytest.raises(ValueError):
            dummy_params_no_attributes = dummy_params.copy()
            dummy_params_no_attributes.pop("attributes")
            BaseModel(**dummy_params_no_attributes)
        with pytest.raises(ValueError):
            dummy_params_no_n_alt = dummy_params.copy()
            dummy_params_no_n_alt.pop("n_alt")
            BaseModel(**dummy_params_no_n_alt)
        with pytest.raises(ValueError):
            # Attributes should be a list
            dummy_params_wrong_attributes = dummy_params.copy()
            dummy_params_wrong_attributes["attributes"] = "a"
            BaseModel(**dummy_params_wrong_attributes)
        with pytest.raises(ValueError):
            # Attributes should be a list of strings
            dummy_params_wrong_attributes = dummy_params.copy()
            dummy_params_wrong_attributes["attributes"] = [1, 2, 3]
            BaseModel(**dummy_params_wrong_attributes)
        with pytest.raises(ValueError):
            # n_alt should be an integer
            dummy_params_wrong_n_alt = dummy_params.copy()
            dummy_params_wrong_n_alt["n_alt"] = 1.0
            BaseModel(**dummy_params_wrong_n_alt)
        with pytest.raises(ValueError):
            # layers_dim should be a list
            dummy_params_wrong_layers_dim = dummy_params.copy()
            dummy_params_wrong_layers_dim["layers_dim"] = "a"
            BaseModel(**dummy_params_wrong_layers_dim)

    def test_predict_before_fit(self):
        # Test calling predict before the model is fitted
        dummy_params = {"attributes": ["a", "b", "c"], "n_alt": 3, "layers_dim": [10, 5]}
        model = BaseModel(**dummy_params, warnings=False)
        with pytest.raises(Exception):
            model.predict(np.array([1, 2, 3]))

    def test_evaluate_before_fit(self):
        # Test calling evaluate before the model is fitted
        dummy_params = {"attributes": ["a", "b", "c"], "n_alt": 3, "layers_dim": [10, 5]}
        model = BaseModel(**dummy_params, warnings=False)
        with pytest.raises(Exception):
            model.evaluate(np.array([1, 2, 3]), np.array([1, 2, 3]))

    def test_regularizer(self):
        # Test the regularizer function
        dummy_params = {"attributes": ["a", "b", "c"], "n_alt": 3, "layers_dim": [10, 5]}
        # Test no regularizer
        model = BaseModel(**dummy_params, warnings=False)
        assert model._regularizer() is None
        # Test regularizer l1 with base parametrs
        dummy_params["regularizer"] = "l1"
        model = BaseModel(**dummy_params, warnings=False)
        assert isinstance(model._regularizer(), tf.keras.regularizers.L1)
        # Test wrong regularization rate
        dummy_params["regularization_rate"] = -0.001
        with pytest.raises(ValueError):
            model = BaseModel(**dummy_params, warnings=False)
        # Add a regularization rate for next tests
        dummy_params["regularization_rate"] = 0.001
        # Test regularizer l2
        dummy_params["regularizer"] = "l2"
        model = BaseModel(**dummy_params, warnings=False)
        assert isinstance(model._regularizer(), tf.keras.regularizers.L2)
        # Test regularizer l1_l2
        dummy_params["regularizer"] = "l1_l2"
        model = BaseModel(**dummy_params, warnings=False)
        assert isinstance(model._regularizer(), tf.keras.regularizers.L1L2)
        # Test regularizer unknown
        dummy_params["regularizer"] = "unknown"
        with pytest.raises(ValueError):
            model = BaseModel(**dummy_params, warnings=False)
            model._regularizer()

    def test_summary_before_fit(self):
        # Test calling summary before the model is fitted
        dummy_params = {"attributes": ["a", "b", "c"], "n_alt": 3, "layers_dim": [10, 5]}
        model = BaseModel(**dummy_params, warnings=False)
        with pytest.raises(Exception):
            model.summary()

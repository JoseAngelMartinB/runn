"""
Useful econometric indicators that can be extracted from the models.
"""
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from runn.models.base import BaseModel


def willingness_to_pay(
    model: BaseModel,
    x: Union[tf.Tensor, np.ndarray, pd.DataFrame],
    analysed_attribute: Union[int, str],
    cost_attribute: Union[int, str],
    alt: int,
    scaler: Optional[object] = None,
) -> np.ndarray:
    """Calculate the willingness to pay (WTP) for a given attribute and alternative. The WTP is calculated for all
    the observations in the input data.

    Args:
        model: The model to be used. It should be a model defined in the runn.models module.
        x: The input data. It can be a tf.Tensor, np.ndarray or pd.DataFrame.
        analysed_attribute: The index or name of the attribute to be analysed.
        cost_attribute: The index or name of the cost attribute.
        alt: The index of the alternative to be analysed.
        scaler: If the data was scaled before training the model, the scaler object should be provided. Currently,
            only the StandardScaler and MinMaxScaler from sklearn.preprocessing are supported. Default: None.

    Returns:
        Numpy array with the WTP for each observation in the input data.
    """
    if scaler is not None and not isinstance(scaler, (StandardScaler, MinMaxScaler)):
        raise ValueError(
            "The scaler object should be either a StandardScaler or a MinMaxScaler from " "sklearn.preprocessing."
        )

    if isinstance(analysed_attribute, str) and isinstance(x, pd.DataFrame):
        if analysed_attribute not in model.params["attributes"]:
            raise ValueError("The analysed attribute is not present in the model.")
        analysed_attribute = x.columns.get_loc(analysed_attribute)
    elif not isinstance(analysed_attribute, int):
        raise ValueError(
            "The analysed attribute should be either an integer indicating the index of the attribute "
            "or a string with the name of the attribute."
        )

    if isinstance(cost_attribute, str) and isinstance(x, pd.DataFrame):
        if cost_attribute not in model.params["attributes"]:
            raise ValueError("The cost attribute is not present in the model.")
        cost_attribute = x.columns.get_loc(cost_attribute)
    elif not isinstance(cost_attribute, int):
        raise ValueError(
            "The cost attribute should be either an integer indicating the index of the attribute "
            "or a string with the name of the attribute."
        )

    if analysed_attribute == cost_attribute:
        raise ValueError("The analysed attribute cannot be the same as the cost attribute.")
    if analysed_attribute >= len(model.params["attributes"]):
        raise ValueError("The analysed attribute index is out of range.")
    if cost_attribute >= len(model.params["attributes"]):
        raise ValueError("The cost attribute index is out of range.")

    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x)

    if alt >= model.params["n_alt"]:
        raise ValueError("The alternative index is out of range.")
    if alt < 0:
        raise ValueError("The alternative index cannot be negative.")

    # Compute the gradient of the utility function with respect to the analysed attributes using the tensorflow
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred_utility = model.get_utility(x)
        pred_utility = pred_utility[:, alt]
    grad = tape.gradient(pred_utility, x)

    grad_cost = grad[:, cost_attribute]
    grad_analysed_attr = grad[:, analysed_attribute]

    # Undo the scaling effect on the WTP
    if scaler is not None:
        if type(scaler) is StandardScaler:
            if isinstance(analysed_attribute, str):
                analysed_attr_scale = scaler.scale_[list(scaler.feature_names_in_).index(analysed_attribute)]
            else:
                analysed_attr_scale = scaler.scale_[
                    list(scaler.feature_names_in_).index(model.params["attributes"][analysed_attribute])
                ]
            if isinstance(cost_attribute, str):
                cost_attr_scale = scaler.scale_[list(scaler.feature_names_in_).index(cost_attribute)]
            else:
                cost_attr_scale = scaler.scale_[
                    list(scaler.feature_names_in_).index(model.params["attributes"][cost_attribute])
                ]
            grad_analysed_attr = grad_analysed_attr / analysed_attr_scale
            grad_cost = grad_cost / cost_attr_scale
        elif type(scaler) is MinMaxScaler:
            raise NotImplementedError("MinMaxScaler not implemented yet.")  # TODO: Implement

    # Compute the WTP
    wtp = -grad_analysed_attr / grad_cost
    wtp = wtp.numpy()

    return wtp


def value_of_time(
    model: BaseModel,
    x: Union[tf.Tensor, np.ndarray, pd.DataFrame],
    time_attribute: Union[int, str],
    cost_attribute: Union[int, str],
    alt: int,
    scaler: Optional[object] = None,
) -> np.ndarray:
    """Calculate the value of time (VOT) for a given alternative. The VOT is calculated for all
    the observations in the input data.

    Args:
        model: The model to be used. It should be a model defined in the runn.models module.
        x: The input data. It can be a tf.Tensor, np.ndarray or pd.DataFrame.
        time_attribute: The index or name of the time attribute.
        cost_attribute: The index or name of the cost attribute.
        alt: The index of the alternative to be analysed.
        scaler: If the data was scaled before training the model, the scaler object should be provided. Currently,
            only the StandardScaler and MinMaxScaler from sklearn.preprocessing are supported. Default: None.

    Returns:
        Numpy array with the VOT for each observation in the input data.
    """
    return -willingness_to_pay(model, x, time_attribute, cost_attribute, alt, scaler)


def market_shares(model: BaseModel, x: Union[tf.Tensor, np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Calculate the market shares for each alternative.

    Args:
        model: The model to be used. It should be a model defined in the runn.models module.
        x: The input data. It can be a tf.Tensor, np.ndarray or pd.DataFrame.

    Returns:
        Numpy array with the market shares for each alternative.
    """
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x)

    # Compute the matrix of probabilities
    pred_probabilities = model.predict(x)

    # Compute the market shares
    market_shares = np.round(np.mean(pred_probabilities, axis=0) * 100, 4)

    return market_shares

from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import (
    SGD,
    Adadelta,
    Adafactor,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    Ftrl,
    Lion,
    Nadam,
    RMSprop,
)
from tensorflow.keras.regularizers import l1, l1_l2, l2

from runn.plot_model import plot_model
from runn.utils import WarningManager

# Initialize the warning manager
warning_manager = WarningManager()

optimizers = {
    "adadelta": Adadelta,
    "adafactor": Adafactor,
    "adagrad": Adagrad,
    "adam": Adam,
    "adamw": AdamW,
    "adamax": Adamax,
    "ftrl": Ftrl,
    "lion": Lion,
    "nadam": Nadam,
    "rmsprop": RMSprop,
    "sgd": SGD,
}


class BaseModel:
    """Abstract base class for all choice models."""

    __metaclass__ = ABCMeta

    def __init__(self, params: dict = None, filename: str = None, warnings: bool = True) -> None:
        self._initialize_base_variables(warnings=warnings)
        if filename is None:
            # Initialize new model
            self.params = params
            self._initialize_base_params()
        elif isinstance(filename, str):
            # Load model from file
            self.load(filename)
        else:
            raise ValueError("The 'filename' parameter should be a string.")

    def _initialize_base_variables(self, **kwargs) -> None:
        """Initialize the base variables of the model.

        Args:
            **kwargs: Additional arguments passed to the model.
        """
        self.keras_model = None
        self.fitted = False
        self.history = None
        self.attributes = []
        if "warnings" in kwargs:
            warning_manager.set_show_warnings(kwargs.get("warnings", True))

    def _initialize_base_params(self) -> None:
        """Initialize the base parameters of the model."""
        if self.params is None:
            raise ValueError("No parameters provided. Please provide a dictionary with the model parameters.")
        else:
            params = self.params.copy()

        if "attributes" not in params:
            msg = (
                "No 'attributes' parameter provided. Please provide a list with the attributes names, in the "
                "same order as in the input data."
            )
            raise ValueError(msg)
        if not isinstance(params["attributes"], list):
            msg = (
                "The 'attributes' parameter should be a list with the attributes names, in the same order as in "
                "the input data."
            )
            raise ValueError(msg)
        for a in params["attributes"]:
            if not isinstance(a, str):
                msg = "The elements of the 'attributes' list should be strings."
                raise ValueError(msg)

        if "n_alt" not in self.params:
            msg = "No 'n_alt' parameter provided. Please provide the number of output features."
            raise ValueError(msg)
        if not isinstance(self.params["n_alt"], int) or self.params["n_alt"] <= 1:
            msg = "The 'n_alt' parameter should be a positive integer greater than 1."
            raise ValueError(msg)

        if "layers_dim" not in self.params:
            self.params["layers_dim"] = [25, 25]
            msg = "No 'layers_dim' parameter provided. Using default value: [25, 25]."
            warning_manager.warn(msg)
        if not isinstance(self.params["layers_dim"], list):
            msg = "The 'layers_dim' parameter should be a list with the number of neurons in each hidden layer, \" \
                \"the length of the list is the number of hidden layers."
            raise ValueError(msg)

        if "regularizer" not in self.params:
            self.params["regularizer"] = None
            msg = "No 'regularizer' parameter provided. Using default value: None."
            warning_manager.warn(msg)
        if self.params["regularizer"] is not None:
            if not isinstance(self.params["regularizer"], str) or self.params["regularizer"] not in [
                "l1",
                "l2",
                "l1_l2",
            ]:
                msg = (
                    "The 'regularizer' parameter should be a string indicating the type of regularization: "
                    "'l1', 'l2' or 'l1_l2'."
                )
                raise ValueError(msg)
            if "regularization_rate" not in self.params or self.params["regularization_rate"] is None:
                self.params["regularization_rate"] = 0.001
                msg = "No 'regularization_rate' parameter provided. Using default value: 0.001."
                warning_manager.warn(msg)
            if not isinstance(self.params["regularization_rate"], float) or self.params["regularization_rate"] <= 0:
                msg = "The 'regularization_rate' parameter should be a positive float."
                raise ValueError(msg)

        if "learning_rate" not in self.params:
            self.params["learning_rate"] = 0.001
            msg = "No 'learning_rate' parameter provided. Using default value: 0.001."
            warning_manager.warn(msg)
        if not isinstance(self.params["learning_rate"], float) or self.params["learning_rate"] <= 0:
            msg = "The 'learning_rate' parameter should be a positive float."
            raise ValueError(msg)

        if "optimizer" not in self.params:
            self.params["optimizer"] = "adam"
            msg = "No 'optimizer' parameter provided. Using default value: 'adam'."
            warning_manager.warn(msg)
        if isinstance(self.params["optimizer"], str):
            # Search for the optimizer in the list of available optimizers, be case insensitive
            optimizer = self.params["optimizer"].lower()
            if optimizer not in optimizers:
                msg = (
                    "Optimizer '{}' not found in the list of available optimizers.\n"
                    "Available optimizers: {}".format(optimizer, list(optimizers.keys()))
                )
                raise ValueError(msg)
            else:
                self.params["optimizer"] = optimizers[optimizer]
        elif not issubclass(self.params["optimizer"], tf.keras.optimizers.Optimizer):
            msg = "The 'optimizer' parameter should be either a string or a tf.keras.optimizers.Optimizer."
            raise ValueError(msg)

        if "loss" not in self.params:
            self.params["loss"] = "categorical_crossentropy"
            msg = "No 'loss' parameter provided. Using default value: 'categorical_crossentropy'."
            warning_manager.warn(msg)
        if not isinstance(self.params["loss"], str) and not isinstance(self.params["loss"], tf.keras.losses.Loss):
            msg = "The 'loss' parameter should be either a string or a tf.keras.losses.Loss."
            raise ValueError(msg)

        if "metrics" not in self.params:
            self.params["metrics"] = ["accuracy"]
            msg = "No 'metrics' parameter provided. Using default value: ['accuracy']."
            warning_manager.warn(msg)
        else:
            if isinstance(self.params["metrics"], str):
                self.params["metrics"] = [self.params["metrics"]]

    def _compile(self) -> None:
        """Compile the keras model."""
        # Define the optimizer
        opt = self.params["optimizer"](learning_rate=self.params["learning_rate"])
        # Compile the model
        self.keras_model.compile(loss=self.params["loss"], optimizer=opt, metrics=["accuracy"])

    def _regularizer(self) -> tf.keras.regularizers.Regularizer:
        """Create a regularizer object based on the model parameters.

        Returns:
            Regularizer object.
        """
        if self.params["regularizer"] is None:
            return None
        elif self.params["regularizer"] == "l1":
            return l1(self.params["regularization_rate"])
        elif self.params["regularizer"] == "l2":
            return l2(self.params["regularization_rate"])
        elif self.params["regularizer"] == "l1_l2":
            return l1_l2(self.params["regularization_rate"])

    def summary(self) -> None:
        """Print a summary of the keras model."""
        if self.keras_model is None:
            raise Exception("Keras model is not initialized yet. Please call build() first.")
        self.keras_model.summary()

    def plot_model(self, filename: str = None, expand_nested=True, dpi: int = 96) -> None:
        """Generate a graphical representation of the model.

        Args:
            filename: File to which the plot will be saved. If None, the plot will only be displayed on screen. Default:
                None.
            expand_nested: Whether to expand nested models into clusters. Default: True.
            dpi: Resolution of the plot. Default: 96.
        """
        if self.keras_model is None:
            raise ValueError("Keras model is not initialized yet. Please call build() first.")
        if filename is None:
            filename = self.__class__.__name__ + ".png"
        return plot_model(
            self.keras_model,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=expand_nested,
            rankdir="TB",
            style=0,
            color=True,
            to_file=filename,
            dpi=dpi,
        )

    def fit(
        self,
        x: Union[tf.Tensor, np.ndarray, pd.DataFrame],
        y: Union[tf.Tensor, np.ndarray, pd.DataFrame],
        batch_size: Optional[int] = None,
        epochs: int = 1,
        verbose: int = 1,
        callbacks: Optional[list] = None,
        **kwargs,
    ) -> tf.keras.callbacks.History:
        """Train the model for a fixed number of epochs (iterations on a dataset).

        Args:
            x: Input data.
            y: Target data.
            batch_size: Number of samples per gradient update. If unspecified, batch_size will default to 32.
            epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data
                provided. Default: 1.
            verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default: 1.
            callbacks: List of tf.keras.callbacks.Callback instances. List of callbacks to apply during training.
                See tf.keras.callbacks for details. Default: None.
            **kwargs: Additional arguments passed to the keras model. See tf.keras.Model.fit() for details.

        Returns:
            A tf.keras.callbacks.History object. Its History.history attribute is a record of training loss values
            and metrics values at successive epochs, as well as validation loss values and validation metrics values
            (if applicable).
        """
        self.history = self.keras_model.fit(x, y, batch_size, epochs, verbose, callbacks, **kwargs)
        self.fitted = True
        return self.history

    def predict(self, x: Union[tf.Tensor, np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """Predict the choice probabilities for a given input.

        Args:
            x: Input data.
            **kwargs: Additional arguments passed to the keras model. See tf.keras.Model.predict() for details.

        Returns:
            Numpy array with the choice probabilities for each alternative.
        """
        if self.fitted is False:
            raise Exception("The model is not fitted yet. Please call fit() first.")
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)
        return self.keras_model.predict(x, **kwargs)

    def evaluate(
        self, x: Union[tf.Tensor, np.ndarray, pd.DataFrame], y: Union[tf.Tensor, np.ndarray, pd.DataFrame], **kwargs
    ) -> Union[float, list]:
        """Returns the loss value & metrics values for the model for a given input.

        Args:
            x: Input data.
            y: Target data.
            **kwargs: Additional arguments passed to the keras model. See tf.keras.Model.evaluate() for details.

        Returns:
            Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has
            multiple outputs and/or metrics). See tf.keras.Model.evaluate() for details.
        """
        if self.fitted is False:
            raise Exception("The model is not fitted yet. Please call fit() first.")
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)
        return self.keras_model.evaluate(x, y, **kwargs)

    @abstractmethod
    def _build(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

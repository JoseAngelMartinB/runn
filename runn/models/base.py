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
    """Abstract base class for all choice models.

    Args:
        attributes: List with the attributes names in the model, in the same order as in the input data. If None, the
            model cannot be initialized unless it is loaded from a file. Default: None.
        n_alt: Number of alternatives in the choice set. If None, the model cannot be initialized unless it is loaded
            from a file. Default: None.
        layers_dim: List with the number of neurons in each hidden layer, the length of the list is the number of
            hidden layers. Default: [25, 25].
        regularizer: Type of regularization to apply. Possible values: 'l1', 'l2' or 'l1_l2'. Default: None.
        regularization_rate: Regularization rate if regularizer is not None. Default: 0.001.
        learning_rate: Learning rate of the optimizer. Default: 0.001.
        optimizer: Optimizer to use. Can be either a string or a tf.keras.optimizers.Optimizer. Default: 'adam'.
        loss: Loss function to use. Can be either a string or a tf.keras.losses.Loss. Default:
            'categorical_crossentropy'.
        metrics: List of metrics to be evaluated by the model during training and testing. Each of this can be either
            a string or a tf.keras.metrics.Metric. Default: ['accuracy'].
        filename: Load a previously trained model from a file. If None, a new model will be initialized. When loading
            a model from a file, the previous parameters will be ignored. Default: None.
        warnings: Whether to show warnings or not. Default: True.
    """

    __metaclass__ = ABCMeta

    def __init__(
        self,
        attributes: Optional[list] = None,
        n_alt: Optional[int] = None,
        layers_dim: list = [25, 25],
        regularizer: Optional[str] = None,
        regularization_rate: float = 0.001,
        learning_rate: float = 0.001,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
        loss: Union[str, tf.keras.losses.Loss] = "categorical_crossentropy",
        metrics: list = ["accuracy"],
        filename: Optional[str] = None,
        warnings: bool = True,
    ) -> None:
        self._initialize_base_variables(warnings=warnings)
        if filename is None:
            # Initialize new model
            self._initialize_base_params(
                attributes=attributes,
                n_alt=n_alt,
                layers_dim=layers_dim,
                regularizer=regularizer,
                regularization_rate=regularization_rate,
                learning_rate=learning_rate,
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
            )
        elif isinstance(filename, str):
            # Load model from file
            self.load(filename)
        else:
            raise ValueError("The 'filename' parameter should be a string.")

    def _initialize_base_variables(self, **kwargs) -> None:
        """Initialize the base variables of the model.

        Args:
            **kwargs: Additional arguments passed to the model. See the documentation of the class for more details.
        """
        self.keras_model = None
        self.fitted = False
        self.history = None
        self.attributes = []
        if "warnings" in kwargs:
            warning_manager.set_show_warnings(kwargs.get("warnings", True))

    def _initialize_base_params(self, **kwargs) -> None:
        """Initialize the base parameters of the model.

        Args:
            **kwargs: Additional arguments passed to the model. See the documentation of the class for more details.
        """
        self.attributes = kwargs["attributes"]
        if self.attributes is None:
            msg = "The 'attributes' parameter is required to initialize a new model."
            raise ValueError(msg)
        if not isinstance(self.attributes, list):
            msg = (
                "The 'attributes' parameter should be a list with the attributes names, in the same order as in "
                "the input data."
            )
            raise ValueError(msg)
        for a in self.attributes:
            if not isinstance(a, str):
                msg = "The elements of the 'attributes' list should be strings."
                raise ValueError(msg)

        self.n_alt = kwargs["n_alt"]
        if self.n_alt is None:
            msg = "The 'n_alt' parameter is required to initialize a new model."
            raise ValueError(msg)
        if not isinstance(self.n_alt, int) or self.n_alt <= 1:
            msg = "The 'n_alt' parameter should be a positive integer greater than 1."
            raise ValueError(msg)

        self.layers_dim = kwargs["layers_dim"]
        if not isinstance(self.layers_dim, list):
            msg = (
                "The 'layers_dim' parameter should be a list with the number of neurons in each hidden layer, "
                "the length of the list is the number of hidden layers."
            )
            raise ValueError(msg)

        self.regularizer = kwargs["regularizer"]
        if self.regularizer is not None:
            if not isinstance(self.regularizer, str) or self.regularizer not in ["l1", "l2", "l1_l2"]:
                msg = (
                    "The 'regularizer' parameter should be a string indicating the type of regularization: "
                    "'l1', 'l2' or 'l1_l2'."
                )
                raise ValueError(msg)

        self.regularization_rate = kwargs["regularization_rate"]
        if not isinstance(self.regularization_rate, float) or self.regularization_rate <= 0:
            msg = "The 'regularization_rate' parameter should be a positive float."
            raise ValueError(msg)

        self.learning_rate = kwargs["learning_rate"]
        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            msg = "The 'learning_rate' parameter should be a positive float."
            raise ValueError(msg)

        self.optimizer = kwargs["optimizer"]
        if isinstance(self.optimizer, str):
            # Search for the optimizer in the list of available optimizers, be case insensitive
            optimizer = self.optimizer.lower()
            if optimizer not in optimizers:
                msg = (
                    "Optimizer '{}' not found in the list of available optimizers.\n"
                    "Available optimizers: {}".format(optimizer, list(optimizers.keys()))
                )
                raise ValueError(msg)
            else:
                self.optimizer = optimizers[optimizer]
        elif not issubclass(self.optimizer, tf.keras.optimizers.Optimizer):
            msg = "The 'optimizer' parameter should be either a string or a tf.keras.optimizers.Optimizer."
            raise ValueError(msg)

        self.loss = kwargs["loss"]
        if not isinstance(self.loss, str) and not isinstance(self.loss, tf.keras.losses.Loss):
            msg = "The 'loss' parameter should be either a string or a tf.keras.losses.Loss."
            raise ValueError(msg)

        self.metrics = kwargs["metrics"]
        if isinstance(self.metrics, str):
            self.metrics = [self.metrics]
            msg = "The 'metrics' parameter should be a list of strings. Converting to list with one element."
            warning_manager.warn(msg)

    def _compile(self) -> None:
        """Compile the keras model."""
        # Define the optimizer
        opt = self.optimizer(learning_rate=self.learning_rate)
        # Compile the model
        self.keras_model.compile(loss=self.loss, optimizer=opt, metrics=["accuracy"])

    def _regularizer(self) -> tf.keras.regularizers.Regularizer:
        """Create a regularizer object based on the model parameters.

        Returns:
            Regularizer object.
        """
        if self.regularizer is None:
            return None
        elif self.regularizer == "l1":
            return l1(self.regularization_rate)
        elif self.regularizer == "l2":
            return l2(self.regularization_rate)
        elif self.regularizer == "l1_l2":
            return l1_l2(self.regularization_rate)

    def summary(self, line_length: int = 100, **kwargs) -> None:
        """Print a summary of the keras model.

        Args:
            line_length: Total length of printed lines. Default: 100.
            **kwargs: Additional arguments passed to the keras model. See tf.keras.Model.summary() for details.
        """
        if self.keras_model is None:
            raise Exception("Keras model is not initialized yet. Please initialize the model first.")
        print("------ {} ------".format(self.__class__.__name__))
        self._print_data_summary(line_length=line_length)
        print("\nSummary of the keras model:")
        self.keras_model.summary(line_length=line_length, **kwargs)

    def _print_data_summary(self, line_length: int = 100) -> None:
        """Print a summary of the data used in the model.

        Args:
            line_length: Total length of printed lines. Default: 100.
        """
        print("\nSummary of the data used in the model:")
        print(" - Attributes used in the model:")
        # Break the attributes list into multiple lines if the line is too long (more than 100 characters).
        # Try that every line is as close as possible to 80 characters.
        attributes_str = ""
        char_count = 0
        for i, attr in enumerate(self.attributes):
            if i == 0:
                attributes_str += "    ["
                attributes_str += attr
                char_count += len(attr)
            elif char_count + len(attr) + 6 < line_length:
                attributes_str += ", " + attr
                char_count += len(attr) + 2
            else:
                attributes_str += ",\n    " + attr
                char_count = len(attr)
        attributes_str += "]"
        print(attributes_str)
        print(" - Number of alternatives in the choice set: %d" % self.n_alt)

    def plot_model(self, filename: str = None, expand_nested=True, dpi: int = 96) -> None:
        """Generate a graphical representation of the model.

        Args:
            filename: File to which the plot will be saved. If None, the plot will only be displayed on screen. Default:
                None.
            expand_nested: Whether to expand nested models into clusters. Default: True.
            dpi: Resolution of the plot. Default: 96.
        """
        if self.keras_model is None:
            raise ValueError("Keras model is not initialized yet. Please initialize the model first.")
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
    def save(self, path: str = "model.zip") -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_utility(self, x: Union[tf.Tensor, np.ndarray, pd.DataFrame]) -> np.ndarray:
        raise NotImplementedError

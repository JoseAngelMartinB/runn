import datetime
import json
import os
import pickle
from typing import Optional, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Model

import runn
from runn.models.base import BaseModel
from runn.utils import WarningManager

# Initialize the warning manager
warning_manager = WarningManager()


class DNN(BaseModel):
    """Deep neural network model for choice modeling.

    Args:
        attributes: List with the attributes names in the model, in the same order as in the input data. If None, the
            model cannot be initialized unless it is loaded from a file. Default: None.
        n_alt: Number of alternatives in the choice set. If None, the model cannot be initialized unless it is loaded
            from a file. Default: None.
        layers_dim: List with the number of neurons in each hidden layer, the length of the list is the number of
            hidden layers. Default: [25, 25].
        activation: Activation function to use in the hidden layers. Can be either a string or a list of strings.
            See https://keras.io/api/layers/activations/ for the available activations. Default: 'relu'.
        regularizer: Type of regularization to apply. Possible values: 'l1', 'l2' or 'l1_l2'. Default: None.
        regularization_rate: Regularization rate if regularizer is not None. Default: 0.001.
        learning_rate: Learning rate of the optimizer. Default: 0.001.
        dropout: Dropout rate to use in the hidden layers. Can be either a float or a list of floats. If a float is
            provided, the same dropout rate will be used in all the hidden layers. Default: 0.0.
        batch_norm: Whether to use batch normalization or not. Default: False.
        optimizer: Optimizer to use. Can be either a string or a tf.keras.optimizers.Optimizer. Default: 'adam'.
        loss: Loss function to use. Can be either a string or a tf.keras.losses.Loss. Default:
            'categorical_crossentropy'.
        metrics: List of metrics to be evaluated by the model during training and testing. Each of this can be either
            a string or a tf.keras.metrics.Metric. Default: ['accuracy'].
        filename: Load a previously trained model from a file. If None, a new model will be initialized. When loading
            a model from a file, the previous parameters will be ignored. Default: None.
        warnings: Whether to show warnings or not. Default: True.
    """

    def __init__(
        self,
        attributes: Optional[list] = None,
        n_alt: Optional[int] = None,
        layers_dim: list = [25, 25],
        activation: Union[str, list] = "relu",
        regularizer: Optional[str] = None,
        regularization_rate: float = 0.001,
        dropout: Union[float, list] = 0.0,
        batch_norm: bool = False,
        learning_rate: float = 0.001,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
        loss: Union[str, tf.keras.losses.Loss] = "categorical_crossentropy",
        metrics: list = ["accuracy"],
        filename: Optional[str] = None,
        warnings: bool = True,
    ) -> None:
        super().__init__(
            attributes=attributes,
            n_alt=n_alt,
            layers_dim=layers_dim,
            regularizer=regularizer,
            regularization_rate=regularization_rate,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            warnings=warnings,
        )
        if filename is None:
            self._initialize_dnn_params(activation=activation, dropout=dropout, batch_norm=batch_norm)
            self._build()
        self._compile()

    def _initialize_dnn_params(self, **kwargs) -> None:
        """Initialize the parameters of the DNN model.

        Args:
            **kwargs: Keyword arguments with the parameters to initialize. See the documentation of the class for more
                details.
        """
        self.activation = kwargs["activation"]
        if isinstance(self.activation, str):
            self.activation = [self.activation for _ in range(len(self.layers_dim))]
        elif isinstance(self.activation, list):
            if len(self.activation) != len(self.layers_dim):
                msg = "The length of the 'activation' list should be equal to the number of hidden layers."
                raise ValueError(msg)
            for a in self.activation:
                if not isinstance(a, str):
                    msg = "The elements of the 'activation' list should be strings."
                    raise ValueError(msg)
        else:
            msg = "The 'activation' parameter should be either a string or a list of strings."
            raise ValueError(msg)

        self.dropout = kwargs["dropout"]
        if isinstance(self.dropout, float) or self.dropout == 0:
            self.dropout = [self.dropout for _ in range(len(self.layers_dim))]
        elif isinstance(self.dropout, list):
            if len(self.dropout) != len(self.layers_dim):
                msg = "The length of the 'dropout' list should be equal to the number of hidden layers."
                raise ValueError(msg)

        self.batch_norm = kwargs["batch_norm"]
        if not isinstance(self.batch_norm, bool):
            msg = "The 'batch_norm' parameter should be a boolean."
            raise ValueError(msg)

    def _build(self) -> None:
        """Build the architecture of the DNN model."""
        # Input layer
        input_shape = (len(self.attributes),)
        inputs = Input(shape=input_shape, name="features")
        # Hidden layers (fully-connected)
        x = inputs
        for L in range(0, len(self.layers_dim)):
            x = Dense(
                self.layers_dim[L],
                activation=self.activation[L],
                kernel_regularizer=self._regularizer(),
                name="dense_{}".format(L + 1),
            )(x)
            if self.dropout[L] > 0:
                x = Dropout(self.dropout[L], name="dropout_{}".format(L + 1))(x)
            if self.batch_norm:
                x = BatchNormalization(name="batch_norm_{}".format(L + 1))(x)
        # Output layer (fully-connected). Represents the pseudo-utilities of the alternatives
        u = Dense(self.n_alt, activation="linear", kernel_regularizer=self._regularizer(), name="U")(x)
        # Softmax activation function to obtain the choice probabilities
        outputs = Activation("softmax", name="P")(u)
        # Create the model
        self.keras_model = Model(inputs=inputs, outputs=outputs, name="DNN")

    def save(self, path: str = "model.zip") -> None:
        """Save the model to a file.

        Args:
            path: Path to the file where the model will be saved. Default: 'model.zip'.
        """
        if not isinstance(path, str):
            raise ValueError("The 'path' parameter should be a string.")
        if path[-3:] != ".zip":
            path += ".zip"
        aux_files = path[:-4]

        files = []
        # Save model info as json
        model_info = {
            "model": "DNN",
            "runn_version": runn.__version__,
            "creation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fitted": self.fitted,
        }
        with open(aux_files + "_info.json", "w") as f:
            json.dump(model_info, f)
        files.append(aux_files + "_info.json")

        # Save the parameters of the model
        # Save all the parameters of the model in a pickle file except the keras model
        pickle.dump(
            [
                self.attributes,
                self.n_alt,
                self.layers_dim,
                self.activation,
                self.regularizer,
                self.regularization_rate,
                self.dropout,
                self.batch_norm,
                self.learning_rate,
                self.optimizer,
                self.loss,
                self.metrics,
            ],
            open(aux_files + "_params.pkl", "wb"),
        )
        files.append(aux_files + "_params.pkl")

        # Save the keras model
        self.keras_model.save_weights(aux_files + "_model.h5")
        files.append(aux_files + "_model.h5")

        # Save the history
        pickle.dump(self.history, open(aux_files + "_history.pkl", "wb"))
        files.append(aux_files + "_history.pkl")

        # Compress all the files
        with ZipFile(path, "w") as zip:
            for file in files:
                zip.write(file, os.path.basename(file))

        # Delete the auxiliary files
        for file in files:
            os.remove(file)

    def load(self, path: str) -> None:
        """Load the model from a file.

        Args:
            path: Path to the file where the model is saved.
        """
        if not isinstance(path, str):
            raise ValueError("The 'path' parameter should be a string.")
        # Check that the str ends with .zip
        if not path.endswith(".zip"):
            raise ValueError("The 'path' parameter should be a .zip file.")
        else:
            # Remove the .zip extension
            aux_files = path[:-4]
            # Get the last index of the '/' character
            idx = aux_files.rfind("/")
            # Get the name of the file without the path
            aux_name = aux_files[idx + 1 :]

        try:
            # Extract the files inside an temporal auxiliary folder
            os.mkdir(aux_files)
            with ZipFile(path, "r") as zip:
                zip.extractall(path=aux_files)

            # Load model info
            with open(aux_files + "/" + aux_name + "_info.json", "r") as f:
                model_info = json.load(f)
            if model_info["model"] != "DNN":
                raise ValueError("The model in the file is not a DNN model.")

            # Check runn version
            major, minor, patch = model_info["runn_version"].split(".")
            if (
                int(major) > int(runn.__version__.split(".")[0])
                or (
                    int(major) == int(runn.__version__.split(".")[0])
                    and int(minor) > int(runn.__version__.split(".")[1])
                )
                or (
                    int(major) == int(runn.__version__.split(".")[0])
                    and int(minor) == int(runn.__version__.split(".")[1])
                    and int(patch) > int(runn.__version__.split(".")[2])
                )
            ):
                msg = (
                    "The model was created with a newer version of runn ({}). "
                    "Please update runn to version {} or higher.".format(model_info["runn_version"], runn.__version__)
                )
                warning_manager.warn(msg)

            # Load the parameters of the model
            (
                self.attributes,
                self.n_alt,
                self.layers_dim,
                self.activation,
                self.regularizer,
                self.regularization_rate,
                self.dropout,
                self.batch_norm,
                self.learning_rate,
                self.optimizer,
                self.loss,
                self.metrics,
            ) = pickle.load(open(aux_files + "/" + aux_name + "_params.pkl", "rb"))

            # Load the keras model
            self._build()
            self.keras_model.load_weights(aux_files + "/" + aux_name + "_model.h5")

            # Load the history
            self.history = pickle.load(open(aux_files + "/" + aux_name + "_history.pkl", "rb"))
            self.fitted = model_info["fitted"]
        except Exception as e:
            raise e
        finally:
            # Delete the auxiliary folder
            for file in os.listdir(aux_files):
                os.remove(aux_files + "/" + file)
            os.rmdir(aux_files)

    def get_utility(self, x: Union[tf.Tensor, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get the utility of each alternative for a given set of observations.

        Args:
            x: The input data. It can be a tf.Tensor, np.ndarray or pd.DataFrame.

        Returns:
            Numpy array with the utility of each alternative for each observation in the input data.
        """
        if self.fitted is False:
            raise Exception("The model is not fitted yet. Please call fit() first.")

        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)

        utility_model = Model(inputs=self.keras_model.input, outputs=self.keras_model.get_layer("U").output)
        return utility_model(x)

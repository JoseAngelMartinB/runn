import datetime
import json
import os
import pickle
from typing import Dict, Optional, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model

import runn
from runn.keras.layers import Gather
from runn.models.dnn import DNN
from runn.utils import IncompatibleVersionError, WarningManager

# Initialize the warning manager
warning_manager = WarningManager()


class AltSpecNN(DNN):
    """Alternative-specific neural network model for choice modeling.

    Args:
        attributes: List with the attributes names in the model, in the same order as in the input data. If None, the
            model cannot be initialized unless it is loaded from a file. Default: None.
        n_alt: Number of alternatives in the choice set. If None, the model cannot be initialized unless it is loaded
            from a file. Default: None.
        alt_spec_attrs: Dictionary with the alternative-specific attributes. The keys are the index of the alternative
            and the values are lists with the names of the alternative-specific attributes. The alternative-specific
            attributes must be a subset of the attributes defined in the model. If None, the model cannot be initialized
            unless it is loaded from a file. Default: None.
        shared_attrs: List with the names of the attributes that are shared across all alternatives. The shared
            attributes must be a subset of the attributes defined in the model. If None, the model cannot be initialized
            unless it is loaded from a file. Default: None.
        socioec_attrs: List with the names of the socio-economic attributes of each decision maker. The socio-economic
            attributes must be a subset of the attributes defined in the model.  If None, the model cannot be initialized
            unless it is loaded from a file. Default: None.
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
        alt_spec_attrs: Optional[Dict[int, list]] = None,
        shared_attrs: Optional[list] = None,
        socioec_attrs: Optional[list] = None,
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
        # Initialize the parameters of the base class
        super(DNN, self).__init__(
            attributes=attributes,
            n_alt=n_alt,
            layers_dim=layers_dim,
            regularizer=regularizer,
            regularization_rate=regularization_rate,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            filename=filename,
            warnings=warnings,
        )
        if filename is None:
            self._initialize_dnn_params(activation=activation, dropout=dropout, batch_norm=batch_norm)
            self._initialize_AltSpecNN_params(
                alt_spec_attrs=alt_spec_attrs, shared_attrs=shared_attrs, socioec_attrs=socioec_attrs
            )
            self._build()
        self._compile()

    def _initialize_AltSpecNN_params(self, **kwargs) -> None:
        """Initialize the parameters of the AltSpecNN model.

        Args:
            **kwargs: Keyword arguments with the parameters to initialize. See the documentation of the class for more
                details.
        """
        self.alt_spec_attrs = kwargs["alt_spec_attrs"]
        if not isinstance(self.alt_spec_attrs, dict):
            raise TypeError("The alternative-specific attributes 'alt_spec_attrs' must be defined in a dictionary.")
        for alt, attrs in self.alt_spec_attrs.items():
            if not isinstance(alt, int):
                raise TypeError("The alternative index in 'alt_spec_attrs' must be an integer.")
            if not 0 <= alt < self.n_alt:
                raise ValueError("The alternative index in 'alt_spec_attrs' must be between 0 and n_alt-1.")
            if not isinstance(attrs, list):
                raise TypeError("The attributes specified for each alternative in 'alt_spec_attrs' must be a list.")
            if not set(attrs).issubset(self.attributes):
                raise ValueError(
                    "The attributes specified for each alternative in 'alt_spec_attrs' must be a subset of the attributes defined in the model."
                )

        self.shared_attrs = kwargs["shared_attrs"]
        if not isinstance(self.shared_attrs, list):
            raise TypeError("The shared attributes 'shared_attrs' must be defined in a list.")
        if not set(self.shared_attrs).issubset(self.attributes):
            raise ValueError(
                "The shared attributes 'shared_attrs' must be a subset of the attributes defined in the model."
            )

        self.socioec_attrs = kwargs["socioec_attrs"]
        if not isinstance(self.socioec_attrs, list):
            raise TypeError("The socio-economic attributes 'socioec_attrs' must be defined in a list.")
        if not set(self.socioec_attrs).issubset(self.attributes):
            raise ValueError(
                "The socio-economic attributes 'socioec_attrs' must be a subset of the attributes defined in the model."
            )

        # Get the position of the attributes in the dataset
        self.alt_spec_attrs_idx = {
            alt: [self.attributes.index(attr) for attr in attrs] for alt, attrs in self.alt_spec_attrs.items()
        }
        self.shared_attrs_idx = [self.attributes.index(attr) for attr in self.shared_attrs]
        self.socioec_attrs_idx = [self.attributes.index(attr) for attr in self.socioec_attrs]

    def _build(self) -> None:
        """Build the architecture of the AltSpecNN model."""
        # Input layer
        input_shape = (len(self.attributes),)
        inputs = Input(shape=input_shape, name="features")
        # Shared attributes layer (shared and socio-economic attributes)
        x_shared = Gather(self.shared_attrs_idx + self.socioec_attrs_idx, axis=1, name="shared_attrs")(inputs)

        # Construct the utility of each alternative
        utilities = []
        for alt in range(0, self.n_alt):
            # Alternative-specific attributes layer
            x_alt_spec = Gather(self.alt_spec_attrs_idx[alt], axis=1, name="alt_spec_attrs_{}".format(alt))(inputs)

            # Define the alternative-specific utility block for the current alternative
            utilities.append(self._build_alt_utility_block(x_shared, x_alt_spec, alt))

        # Concatenate the utilities of all alternatives
        u = concatenate(utilities, axis=1, name="U")
        # Softmax activation function to obtain the choice probabilities
        outputs = Activation("softmax", name="P")(u)
        # Create the model
        self.keras_model = Model(inputs=inputs, outputs=outputs, name="AltSpecNN")

    def _build_alt_utility_block(self, x_shared: tf.Tensor, x_alt_spec: tf.Tensor, alt: int) -> None:
        """Build the architecture of the alternative-specific utility block.

        Args:
            x_shared: Tensor with the shared attributes.
            x_alt_spec: Tensor with the alternative-specific attributes.
            alt: Index of the alternative for which the utility block will be built.
        """
        # Concatenate the shared and alternative-specific attributes
        x = concatenate([x_shared, x_alt_spec], axis=1, name="Attrs_alt_{}".format(alt))

        # Hidden layers
        for L in range(0, len(self.layers_dim)):
            x = Dense(
                self.layers_dim[L],
                activation=self.activation[L],
                kernel_regularizer=self._regularizer(),
                name="dense_{}_alt_{}".format(L + 1, alt),
            )(x)
            if self.dropout[L] > 0:
                x = Dropout(self.dropout[L], name="dropout_{}_alt_{}".format(L + 1, alt))(x)
            if self.batch_norm:
                x = BatchNormalization(name="batch_norm_{}_alt_{}".format(L + 1, alt))(x)

        # Compute the pseudo-utility of the alternative
        utility = Dense(1, activation="linear", kernel_regularizer=self._regularizer(), name="U_alt_{}".format(alt))(x)
        return utility

    def save(self, path: str = "model.zip") -> None:
        """Save the model to a file.

        Args:
            path: Path to the file where the model will be saved. Default: 'model.zip'.
        """
        if not isinstance(path, str):
            raise ValueError("The 'path' parameter should be a string.")
        if path[-4:] != ".zip":
            path += ".zip"
        aux_files = path[:-4]

        files = []
        # Save model info as json
        model_info = {
            "model": "AltSpecNN",
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
                self.alt_spec_attrs,
                self.shared_attrs,
                self.socioec_attrs,
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
            if model_info["model"] != "AltSpecNN":
                msg = (
                    "The model in the file is not a 'AltSpecNN' model. The model cannot be loaded.",
                    "Please try using the '{}' model instead.".format(model_info["model"]),
                )
                raise ValueError(msg)

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
                raise IncompatibleVersionError(msg)

            # Load the parameters of the model
            (
                self.attributes,
                self.n_alt,
                self.alt_spec_attrs,
                self.shared_attrs,
                self.socioec_attrs,
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
            self._initialize_AltSpecNN_params(
                alt_spec_attrs=self.alt_spec_attrs, shared_attrs=self.shared_attrs, socioec_attrs=self.socioec_attrs
            )

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

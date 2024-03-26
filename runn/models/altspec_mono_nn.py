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
from runn.keras.layers import Gather, MonoDense
from runn.models.dnn import DNN
from runn.utils import IncompatibleVersionError, WarningManager

# Initialize the warning manager
warning_manager = WarningManager()


class AltSpecMonoNN(DNN):
    """Alternative-specific monotonic neural network model for choice modeling.

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
        monotonicity_constraints: Dictionary with the attributes that must be monotonic. The keys are the attribute
            names and the values are the monotonicity constraints. The constraints can be either an integer or a
            dictionary. If an integer is provided, the possible values are -1, 0 or 1, which represent decreasing, no
            monotonicity and increasing monotonicity, respectively. If a dictionary is provided, the keys are the index
            of the alternative and the values are the monotonicity constraints for the attribute in that alternative. If
            no monotonicity constraints are provided for an attribute, the default value is 0 (no monotonicity). If
            None, no monotonicity constraints will be applied. Default: None.
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
        monotonicity_constraints: Optional[Dict[str, Union[int, Dict[int, int]]]] = None,
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
        # Initialize the parameters of the base class (calls the __init__ method of the BaseModel class).
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
            self._initialize_AltSpecMonoNN_params(
                alt_spec_attrs=alt_spec_attrs,
                shared_attrs=shared_attrs,
                socioec_attrs=socioec_attrs,
                monotonicity_constraints=monotonicity_constraints,
            )
            self._build()
        self._compile()

    def _initialize_AltSpecMonoNN_params(self, **kwargs) -> None:
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

        self.monotonicity_constraints = kwargs["monotonicity_constraints"]
        if self.monotonicity_constraints is not None:
            if not isinstance(self.monotonicity_constraints, dict):
                raise TypeError("The monotonic attributes 'monotonicity_constraints' must be defined in a dictionary.")
            for attr, constraint in self.monotonicity_constraints.items():  # Check the monotonicity constraints
                if not isinstance(attr, str):
                    raise TypeError("The attribute name in 'monotonicity_constraints' must be a string.")
                if attr not in self.attributes:
                    raise ValueError(
                        "The attribute '{}' in 'monotonicity_constraints' must be defined in the model's attributes.".format(
                            attr
                        )
                    )
                if isinstance(constraint, int):
                    if constraint not in [-1, 0, 1]:
                        raise ValueError("The monotonicity constraint must be -1, 0 or 1.")
                elif isinstance(constraint, dict):  # Alternative-specific monotonicity
                    for alt, c in constraint.items():
                        if not isinstance(alt, int):
                            raise TypeError("The alternative index in 'monotonicity_constraints' must be an integer.")
                        if not 0 <= alt < self.n_alt:
                            raise ValueError(
                                "The alternative index in 'monotonicity_constraints' must be between 0 and n_alt-1."
                            )
                        if not isinstance(c, int):
                            raise TypeError("The monotonicity constraint must be an integer.")
                        if c not in [-1, 0, 1]:
                            raise ValueError("The monotonicity constraint must be -1, 0 or 1.")
                else:
                    raise TypeError(
                        (
                            "The monotonicity constraint must be an integer in the range [-1, 0, 1] or a dictionary ",
                            "with alternative-specific monotonicity constraints.",
                        )
                    )
        self.monotonicity_constraints_idx = self._define_monotonicity(
            self.socioec_attrs, self.shared_attrs, self.alt_spec_attrs, self.monotonicity_constraints
        )

    def _define_monotonicity(
        self, socioec_attrs: list, shared_attrs: list, alt_spec_attrs: dict, monotonicity_constraints: dict
    ):
        """Define the monotonicity constraints for the model."""
        # Define a dictionary with the monotonicity constraints for each attribute. Default: 0 (no monotonicity)
        monotonic_idx = {
            "alt_spec": {alt: np.zeros(len(attrs), dtype=int) for alt, attrs in alt_spec_attrs.items()},
            "shared": {alt: np.zeros(len(shared_attrs), dtype=int) for alt in range(self.n_alt)},
            "socioec": {alt: np.zeros(len(socioec_attrs), dtype=int) for alt in range(self.n_alt)},
        }
        # Define the monotonicity constraints for the alternative-specific attributes
        if alt_spec_attrs is not None:
            for alt, attrs in alt_spec_attrs.items():
                for i, attr in enumerate(attrs):
                    if attr in monotonicity_constraints.keys():
                        if isinstance(monotonicity_constraints[attr], int):
                            monotonic_idx["alt_spec"][alt][i] = monotonicity_constraints[attr]
                        elif isinstance(monotonicity_constraints[attr], dict):
                            if alt in monotonicity_constraints[attr].keys():
                                monotonic_idx["alt_spec"][alt][i] = monotonicity_constraints[attr][alt]
                        else:
                            raise ValueError(
                                "The monotonicity constraint must be an integer in the range [-1, 0, 1] or a ",
                                "dictionary with alternative-specific monotonicity constraints.",
                            )
        # Define the monotonicity constraints for the shared attributes
        if shared_attrs is not None:
            for i, attr in enumerate(shared_attrs):
                if attr in monotonicity_constraints.keys():
                    if isinstance(monotonicity_constraints[attr], int):
                        for alt in range(self.n_alt):
                            monotonic_idx["shared"][alt][i] = monotonicity_constraints[attr]
                    elif isinstance(monotonicity_constraints[attr], dict):
                        for alt in monotonicity_constraints[attr].keys():
                            if alt in range(self.n_alt):
                                monotonic_idx["shared"][alt][i] = monotonicity_constraints[attr][alt]
                    else:
                        raise ValueError(
                            "The monotonicity constraint must be an integer in the range [-1, 0, 1] or a ",
                            "dictionary with alternative-specific monotonicity constraints.",
                        )
        # Define the monotonicity constraints for the socio-economic attributes
        if socioec_attrs is not None:
            for i, attr in enumerate(socioec_attrs):
                if attr in monotonicity_constraints.keys():
                    if isinstance(monotonicity_constraints[attr], int):
                        for alt in range(self.n_alt):
                            monotonic_idx["socioec"][alt][i] = monotonicity_constraints[attr]
                    elif isinstance(monotonicity_constraints[attr], dict):
                        for alt in monotonicity_constraints[attr].keys():
                            if alt in range(self.n_alt):
                                monotonic_idx["socioec"][alt][i] = monotonicity_constraints[attr][alt]
                    else:
                        raise ValueError(
                            "The monotonicity constraint must be an integer in the range [-1, 0, 1] or a ",
                            "dictionary with alternative-specific monotonicity constraints.",
                        )
        return monotonic_idx

    def _build(self) -> None:
        """Build the architecture of the AltSpecNN model."""
        # Input layer
        input_shape = (len(self.attributes),)
        inputs = Input(shape=input_shape, name="features")
        # Extract the shared attributes
        x_shared = Gather(self.shared_attrs_idx, axis=1, name="shared_attrs")(inputs)
        # Extract the socio-economic attributes
        x_socioec = Gather(self.socioec_attrs_idx, axis=1, name="socioec_attrs")(inputs)

        # Construct the utility of each alternative
        utilities = []
        for alt in range(0, self.n_alt):
            # Extract the alternative-specific attributes
            x_alt_spec = Gather(self.alt_spec_attrs_idx[alt], axis=1, name="alt_spec_attrs_{}".format(alt))(inputs)

            # Define the alternative-specific utility block for the current alternative
            utilities.append(self._build_alt_utility_block(x_alt_spec, x_shared, x_socioec, alt))

        # Concatenate the utilities of all alternatives
        u = concatenate(utilities, axis=1, name="U")
        # Softmax activation function to obtain the choice probabilities
        outputs = Activation("softmax", name="P")(u)
        # Create the model
        self.keras_model = Model(inputs=inputs, outputs=outputs, name="AltSpecMonoNN")

    def _build_alt_utility_block(
        self, x_alt_spec: tf.Tensor, x_shared: tf.Tensor, x_socioec: tf.Tensor, alt: int
    ) -> tf.Tensor:
        """Build the architecture of the alternative-specific utility block.

        Args:
            x_alt_spec: Tensor with the alternative-specific attributes.
            x_shared: Tensor with the shared attributes.
            x_socioec: Tensor with the socio-economic attributes.
            alt: Index of the alternative for which the utility block will be built.
        """
        # Concatenate the alternative-specific, shared, and socio-economic attributes
        x = concatenate([x_alt_spec, x_shared, x_socioec], axis=1, name="Attrs_alt_{}".format(alt))

        # Construct the monotonicity indicator for the first layer. It combines the monotonicity constraints of the
        # alternative-specific, shared, and socio-economic attributes
        monotonicity_indicator = np.concatenate(
            [
                self.monotonicity_constraints_idx["alt_spec"][alt],
                self.monotonicity_constraints_idx["shared"][alt],
                self.monotonicity_constraints_idx["socioec"][alt],
            ]
        )

        # Hidden layers
        for L in range(0, len(self.layers_dim)):
            # Set the monotonicity indicator to 1 for all layers except the first one
            if L > 0:
                monotonicity_indicator = 1
            x = MonoDense(
                self.layers_dim[L],
                activation=self.activation[L],
                monotonicity_indicator=monotonicity_indicator,
                kernel_regularizer=self._regularizer(),
                name="dense_{}_alt_{}".format(L + 1, alt),
            )(x)
            if self.dropout[L] > 0:
                x = Dropout(self.dropout[L], name="dropout_{}_alt_{}".format(L + 1, alt))(x)
            if self.batch_norm:
                x = BatchNormalization(name="batch_norm_{}_alt_{}".format(L + 1, alt))(x)

        # Compute the pseudo-utility of the alternative
        utility = MonoDense(
            1,
            activation="linear",
            monotonicity_indicator=1,
            kernel_regularizer=self._regularizer(),
            name="U_alt_{}".format(alt),
        )(x)
        return utility

    def save(self, path: str = "model.zip") -> None:
        """Save the model to a file. The model must be fitted before saving it.

        Args:
            path: Path to the file where the model will be saved. Default: 'model.zip'.
        """
        if not isinstance(path, str):
            raise ValueError("The 'path' parameter should be a string.")
        if path[-4:] != ".zip":
            path += ".zip"
        aux_files = path[:-4]
        if not self.fitted or self.keras_model is None:
            msg = "The model has not been fitted yet. Please call the 'fit' method first."
            raise ValueError(msg)

        files = []
        # Save model info as json
        model_info = {
            "model": "AltSpecMonoNN",
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
                self.monotonicity_constraints,
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
            if model_info["model"] != "AltSpecMonoNN":
                msg = (
                    "The model in the file is not a 'AltSpecMonoNN' model. The model cannot be loaded.",
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
                self.monotonicity_constraints,
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
            self._initialize_AltSpecMonoNN_params(
                alt_spec_attrs=self.alt_spec_attrs,
                shared_attrs=self.shared_attrs,
                socioec_attrs=self.socioec_attrs,
                monotonicity_constraints=self.monotonicity_constraints,
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

    def get_utility(
        self,
        x: Union[tf.Tensor, np.ndarray, pd.DataFrame],
        name: str = "AltSpecMonoNN_Utility",
    ) -> np.ndarray:
        """Get the utility of each alternative for a given set of observations.

        Args:
            x: The input data. It can be a tf.Tensor, np.ndarray or pd.DataFrame.
            name: Name of the utility model. Default: 'AltSpecMonoNN_Utility'.

        Returns:
            Numpy array with the utility of each alternative for each observation in the input data.
        """
        if self.fitted is False:
            raise Exception("The model is not fitted yet. Please call the 'fit' method first.")

        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)

        utility_model = Model(inputs=self.keras_model.input, outputs=self.keras_model.get_layer("U").output, name=name)
        return utility_model(x)

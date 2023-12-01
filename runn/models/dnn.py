import datetime
import json
import os
import pickle
import warnings
from typing import Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Model

import runn
from runn.models.base import BaseModel


class DNN(BaseModel):
    def __init__(self, params: dict = None, filename: str = None) -> None:
        """Deep neural network model for choice modeling.

        Args:
            params: Dictionary with the model parameters. Default: None.
            filename: Load a previously trained model from a file. Default: None.
        """
        super().__init__(params, filename)
        if filename is None:
            self._initilize_dnn_params()
            self._build()
        self._compile()

    def _initilize_dnn_params(self) -> None:
        """Initialize the parameters of the DNN model."""
        if "activation" not in self.params:
            self.params["activation"] = "relu"
            msg = "No 'activation' parameter provided. Using default value: 'relu'."
            warnings.warn(msg)
        if isinstance(self.params["activation"], str):
            self.params["activation"] = [self.params["activation"] for _ in range(len(self.params["layers_dim"]))]
        elif isinstance(self.params["activation"], list):
            if len(self.params["activation"]) != len(self.params["layers_dim"]):
                msg = "The length of the 'activation' list should be equal to the number of hidden layers."
                raise ValueError(msg)
            for a in self.params["activation"]:
                if not isinstance(a, str):
                    msg = "The elements of the 'activation' list should be strings."
                    raise ValueError(msg)
        else:
            msg = "The 'activation' parameter should be either a string or a list of strings."
            raise ValueError(msg)

        if "dropout" not in self.params or self.params["dropout"] is None:
            self.params["dropout"] = 0.0
            msg = "No 'dropout' parameter provided. Using default value: 0.0."
            warnings.warn(msg)
        if isinstance(self.params["dropout"], float) or self.params["dropout"] == 0:
            self.params["dropout"] = [self.params["dropout"] for _ in range(len(self.params["layers_dim"]))]
        elif isinstance(self.params["dropout"], list):
            if len(self.params["dropout"]) != len(self.params["layers_dim"]):
                msg = "The length of the 'dropout' list should be equal to the number of hidden layers."
                raise ValueError(msg)

        if "batch_norm" not in self.params:
            self.params["batch_norm"] = False
            msg = "No 'batch_norm' parameter provided. Using default value: False."
            warnings.warn(msg)
        if not isinstance(self.params["batch_norm"], bool):
            msg = "The 'batch_norm' parameter should be a boolean."
            raise ValueError(msg)

    def _build(self) -> None:
        """Build the architecture of the DNN model."""
        # Input layer
        input_shape = (len(self.params["attributes"]),)
        inputs = Input(shape=input_shape, name="features")
        # Hidden layers (fully-connected)
        x = inputs
        for L in range(0, len(self.params["layers_dim"])):
            x = Dense(
                self.params["layers_dim"][L],
                activation=self.params["activation"][L],
                kernel_regularizer=self._regularizer(),
                name="dense_{}".format(L + 1),
            )(x)
            if self.params["dropout"][L] > 0:
                x = Dropout(self.params["dropout"][L], name="dropout_{}".format(L + 1))(x)
            if self.params["batch_norm"]:
                x = BatchNormalization(name="batch_norm_{}".format(L + 1))(x)
        # Output layer (fully-connected). Represents the pseudo-utilities of the alternatives
        u = Dense(self.params["n_alt"], activation="linear", kernel_regularizer=self._regularizer(), name="U")(x)
        # Softmax activation function to obtain the choice probabilities
        outputs = Activation("softmax", name="P")(u)
        # Create the model
        self.keras_model = Model(inputs=inputs, outputs=outputs)

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

        # Save the parameters
        pickle.dump(self.params, open(aux_files + "_params.pkl", "wb"))
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
                warnings.warn(msg)

            # Load the parameters
            self.params = pickle.load(open(aux_files + "/" + aux_name + "_params.pkl", "rb"))

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

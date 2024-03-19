import datetime
import json
import os
import pickle
from typing import Optional, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Average, Input
from tensorflow.keras.models import Model

import runn
from runn.models.dnn import DNN
from runn.utils import IncompatibleVersionError, NotSupportedError, ProgressBar, WarningManager

# Initialize the warning manager
warning_manager = WarningManager()


class EnsembleDNN(DNN):
    """Ensemble of deep neural network models for choice modeling.

    Args:
        attributes: List with the attributes names in the model, in the same order as in the input data. If None, the
            model cannot be initialized unless it is loaded from a file. Default: None.
        n_alt: Number of alternatives in the choice set. If None, the model cannot be initialized unless it is loaded
            from a file. Default: None.
        n_ensembles: Number of base DNN models in the ensemble. This value should be greater than 1. Default: 5.
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
        n_jobs: Number of parallel jobs to run. If -1, all CPUs are used. If 1 is given, no parallel computing code
            is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
        filename: Load a previously trained model from a file. If None, a new model will be initialized. When loading
            a model from a file, the previous parameters will be ignored. Default: None.
        warnings: Whether to show warnings or not. Default: True.
    """

    def __init__(
        self,
        attributes: Optional[list] = None,
        n_alt: Optional[int] = None,
        n_ensembles: int = 5,
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
        n_jobs: int = 1,
        filename: Optional[str] = None,
        warnings: bool = True,
    ) -> None:
        self._initialize_base_variables(warnings=warnings)
        if filename is None:
            # Initialize the model parameters of a new ensemble model
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
            self._initialize_dnn_params(activation=activation, dropout=dropout, batch_norm=batch_norm)
            self._initialize_ensemble_params(n_ensembles=n_ensembles, n_jobs=n_jobs)
            # Initialize the DNN models and store them in a list
            self.ensemble_pool = []
            for i in range(self.n_ensembles):
                self.ensemble_pool.append(
                    DNN(
                        attributes=self.attributes,
                        n_alt=self.n_alt,
                        layers_dim=self.layers_dim,
                        activation=self.activation,
                        regularizer=self.regularizer,
                        regularization_rate=self.regularization_rate,
                        dropout=self.dropout,
                        batch_norm=self.batch_norm,
                        learning_rate=self.learning_rate,
                        optimizer=self.optimizer,
                        loss=self.loss,
                        metrics=self.metrics,
                    )
                )
        elif isinstance(filename, str):
            # Load model from file
            self.load(filename)
            self._compile()
        else:
            raise ValueError("The 'filename' parameter should be a string.")

    def _initialize_ensemble_params(self, **kwargs) -> None:
        """Initialize the parameters of the ensemble model.

        Args:
            **kwargs: Keyword arguments with the parameters to initialize. See the documentation of the class for more
                details.
        """
        self.n_ensembles = kwargs["n_ensembles"]
        if self.n_ensembles < 2:
            msg = "The 'n_ensembles' parameter should be greater than 1."
            raise ValueError(msg)

        self.n_jobs = kwargs["n_jobs"]
        n_cpus = os.cpu_count()
        if not isinstance(self.n_jobs, int):
            msg = "The 'n_jobs' parameter should be an integer."
            raise ValueError(msg)
        if self.n_jobs < -1:
            self.n_jobs = n_cpus + 1 + self.n_jobs
            if self.n_jobs < 1:
                self.n_jobs = 1
                msg = "The 'n_jobs' parameter should be greater than %d. Setting 'n_jobs' to 1." % (-n_cpus - 1)
                warning_manager.warn(msg)
        elif self.n_jobs == 0:
            self.n_jobs = 1
            msg = "The 'n_jobs' parameter cannot be 0. Setting 'n_jobs' to 1."
            warning_manager.warn(msg)
        elif self.n_jobs > n_cpus:
            self.n_jobs = n_cpus
            msg = (
                "The 'n_jobs' parameter cannot be greater than the number of CPUs available (%d). "
                "Setting 'n_jobs' to %d." % (n_cpus, n_cpus)
            )
            warning_manager.warn(msg)

    def _build(self) -> None:
        """Build the architecture of the ensemble DNN model."""
        # Input layer
        input_shape = (len(self.attributes),)
        inputs = Input(shape=input_shape, name="features")
        # Average the output of the individual DNN models
        average_outputs = Average(name="P")(
            [
                Model(
                    inputs=self.ensemble_pool[i].keras_model.input,
                    outputs=self.ensemble_pool[i].keras_model.output,
                    name="DNN_{}".format(i + 1),
                )(inputs)
                for i in range(self.n_ensembles)
            ]
        )
        # Create the ensemble model
        self.keras_model = Model(inputs=inputs, outputs=average_outputs, name="EnsembleDNN")

    def summary(self, ensemble: bool = True, line_length: int = 100, **kwargs) -> None:
        """Print a summary of the ensemble model.

        Args:
            ensemble: If True, print the summary of the ensemble model. If False, print the summary of an individual
                DNN model. Default: True.
            line_length: Total length of printed lines. Default: 100.
            **kwargs: Additional arguments passed to the keras model. See tf.keras.Model.summary() for details.
        """
        if not ensemble:
            # Print the summary of an individual DNN model
            print("------ {} ------".format(self.__class__.__name__))
            print("Number of DNN models in the ensemble: %d" % self.n_ensembles)
            self._print_data_summary(line_length=line_length)
            if self.ensemble_pool is None or len(self.ensemble_pool) == 0 or self.ensemble_pool[0].keras_model is None:
                msg = (
                    "The individual DNN models have not been initialized yet. "
                    "Please initialize the ensemble model first."
                )
                raise ValueError(msg)
            print("\nSummary of the individual DNN model used in the ensemble:")
            self.ensemble_pool[0].keras_model.summary()
        elif ensemble:
            # Print the summary of the ensemble model
            if self.keras_model is None:
                msg = "The ensemble model has not been constructed yet. Please call the 'fit' method first."
                raise ValueError(msg)
            super().summary()

    def plot_model(self, ensemble: bool = True, **kwargs) -> None:
        """Generate a graphical representation of the ensemble model.

        Args:
            ensemble: Whether to plot the ensemble model or the individual DNN models. Default: True.
            kwargs: Additional arguments passed to the 'plot_model' function. See the documentation of the
                base class for more details.
        """
        if not ensemble:
            # Plot an individual DNN model
            if self.ensemble_pool is None or len(self.ensemble_pool) == 0 or self.ensemble_pool[0].keras_model is None:
                msg = (
                    "The individual DNN models have not been initialized yet. "
                    "Please initialize the ensemble model first."
                )
                raise ValueError(msg)
            return self.ensemble_pool[0].plot_model(**kwargs)
        elif ensemble:
            # Plot the ensemble model
            if self.keras_model is None:
                msg = "The ensemble model has not been constructed yet. Please call the 'fit' method first."
                raise ValueError(msg)
            return super().plot_model(**kwargs)

    def fit(
        self,
        x: Union[tf.Tensor, np.ndarray, pd.DataFrame],
        y: Union[tf.Tensor, np.ndarray],
        batch_size: Optional[int] = None,
        epochs: int = 1,
        verbose: int = 1,
        callbacks: Optional[list] = None,
        validation_split: float = 0.0,
        validation_data: Optional[tuple] = None,
        bagging: Optional[float] = None,
        **kwargs,
    ) -> tf.keras.callbacks.History:
        """Train the ensemble model.

         Args:
            x: Input data. It can be a tf.Tensor, np.ndarray or pd.DataFrame.
            y: The alternative selected by each decision maker in the sample x. Can be either a tf.Tensor or np.ndarray.
                It should be a 1D array with integers in the range [0, n_alt-1] or a 2D array with one-hot encoded
                alternatives.
            batch_size: Number of samples per gradient update. If unspecified, batch_size will default to 32.
            epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data
                provided. Default: 1.
            verbose: Verbosity mode. 0 = silent, 1 = ensemble progress bar, 2 = one progress bar per individual model.
                3 = for each individual model, show one line per epoch. Default: 1.
            callbacks: List of tf.keras.callbacks.Callback instances. List of callbacks to apply during training.
                See tf.keras.callbacks for details. Default: None.
            validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data, will not train on it, and will evaluate
                the loss and any model metrics on this data at the end of each epoch. The validation data is selected
                from the last samples in the x and y data provided, before shuffling. Default: 0.0.
            validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch. The
                model will not be trained on this data. This could be a tuple (x_val, y_val) or a tuple (x_val, y_val,
                val_sample_weights). Default: None.
            bagging: Whether to use bagging or not. If None, bagging will not be used. If a float is provided, it
                indicates the percentage of samples to use in each bootstrap sample. The value should be between 0.0
                and 1.0. Default: None.
            **kwargs: Additional arguments passed to the keras model. See tf.keras.Model.fit() for details.

        Returns:
            A list of tf.keras.callbacks.History objects, one for each individual DNN model. Each History object is a
            record of training loss values and metrics values at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        # Check if the ensemble model has been initialized
        if self.ensemble_pool is None or len(self.ensemble_pool) == 0:
            msg = "The individual DNN models have not been initialized yet. Please initialize the ensemble model first."
            raise ValueError(msg)
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)

        # Check if y is one-hot encoded or a 1D array with integers in the range [0, n_alt-1]
        if isinstance(y, tf.Tensor):
            y = y.numpy()
        if not (len(y.shape) == 2 and y.shape[1] == self.n_alt):
            # y is not one-hot encoded, hence it should be a 1D array with integers in the range [0, n_alt-1]
            if np.any(y < 0) or np.any(y >= self.n_alt):
                raise ValueError("The input parameter 'y' should contain integers in the range [0, n_alt-1].")

        if bagging is not None:
            # Use bagging
            if not isinstance(bagging, float):
                msg = "The 'bagging' parameter should be a float."
                raise ValueError(msg)
            if bagging <= 0.0 or bagging > 1.0:
                msg = "The 'bagging' parameter should be between 0.0 and 1.0."
                raise ValueError(msg)
            # Split the data into bootstrap samples
            bootstrap_x, bootstrap_y = [], []
            idx = np.arange(len(x))
            for i in range(self.n_ensembles):
                # Select random samples with replacement from the data using the bootstrap sample size
                bootstrap_idx = np.random.choice(idx, size=int(len(x) * bagging), replace=True)
                bootstrap_x.append(tf.gather(x, bootstrap_idx))
                bootstrap_y.append(y[bootstrap_idx])

        if verbose == 1:
            pb = ProgressBar(total=self.n_ensembles)
            pb.update(0)
        elif verbose > 1:
            print("Estimating the individual DNN models...")

        # Fit the ensemble model
        for i in range(self.n_ensembles):
            if verbose > 1:
                print("\n------ DNN model {} ------".format(i + 1))
            if bagging is not None:
                x_i, y_i = bootstrap_x[i], bootstrap_y[i]
            else:
                x_i, y_i = x, y
            # Fit the individual DNN model
            self.ensemble_pool[i].fit(
                x=x_i,
                y=y_i,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose - 1,
                callbacks=callbacks,
                validation_split=validation_split,
                validation_data=validation_data,
                **kwargs,
            )

            # Update the progress bar or print the verbose output
            if verbose == 1:
                pb_value_dict = {"loss": "{:.4f}".format(self.ensemble_pool[i].get_history()["loss"][-1])}
                for metric in self.metrics:
                    pb_value_dict[metric] = "{:.4f}".format(self.ensemble_pool[i].get_history()[metric][-1])
                pb.update(i + 1, value_dict=pb_value_dict)
            elif verbose > 1:
                verbose_output = "Training - loss: {:.4f}".format(self.ensemble_pool[i].get_history()["loss"][-1])
                for metric in self.metrics:
                    verbose_output += " - {}: {:.4f}".format(metric, self.ensemble_pool[i].get_history()[metric][-1])
                print(verbose_output)
                if validation_data is not None or validation_split > 0.0:
                    verbose_output = "Validation - loss: {:.4f}".format(
                        self.ensemble_pool[i].get_history()["val_loss"][-1]
                    )
                    for metric in self.metrics:
                        verbose_output += " - {}: {:.4f}".format(
                            metric, self.ensemble_pool[i].get_history()["val_" + metric][-1]
                        )
                    print(verbose_output)

        # Build the ensemble model
        self._build()
        self.fitted = True
        # Compile the ensemble model
        self._compile()

        # Print the verbose output of the ensemble model
        if verbose >= 1:
            # Print the final loss and metrics values
            metrics = ["loss"] + self.metrics
            print("\n------ Ensemble ------")
            verbose_output = "Training"
            ensemble_metrics = dict(zip(metrics, self.evaluate(x, y, verbose=0)))
            for metric in ensemble_metrics:
                verbose_output += " - {}: {:.4f}".format(metric, ensemble_metrics[metric])
            print(verbose_output)
            if validation_data is not None:
                verbose_output = "Validation"
                ensemble_metrics = dict(zip(metrics, self.evaluate(validation_data[0], validation_data[1], verbose=0)))
                for metric in ensemble_metrics:
                    verbose_output += " - {}: {:.4f}".format(metric, ensemble_metrics[metric])
                print(verbose_output)

    def get_history(self) -> list[dict]:
        """Return the history of the model training for each individual DNN model.

        Returns:
            List of dictionaries with the history of the training of each individual DNN model.
        """
        if not self.fitted:
            msg = "The model has not been fitted yet. Please call the 'fit' method first."
            raise ValueError(msg)
        history = []
        for i in range(self.n_ensembles):
            history.append(self.ensemble_pool[i].get_history())
        return history

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
            "model": "EnsembleDNN",
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
                self.n_ensembles,
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
                self.n_jobs,
            ],
            open(aux_files + "_params.pkl", "wb"),
        )
        files.append(aux_files + "_params.pkl")

        # Save the keras ensemble model
        self.keras_model.save_weights(aux_files + "_model.h5")
        files.append(aux_files + "_model.h5")

        # Save the individual DNN models
        for i in range(self.n_ensembles):
            self.ensemble_pool[i].save(aux_files + "_DNN_model_{}.zip".format(i + 1))
            files.append(aux_files + "_DNN_model_{}.zip".format(i + 1))

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
            if model_info["model"] != "EnsembleDNN":
                msg = (
                    "The model in the file is not a EnsembleDNN model. The model cannot be loaded.",
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
                self.n_ensembles,
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
                self.n_jobs,
            ) = pickle.load(open(aux_files + "/" + aux_name + "_params.pkl", "rb"))

            # Load the individual DNN models
            self.ensemble_pool = []
            for i in range(self.n_ensembles):
                self.ensemble_pool.append(
                    DNN(filename=aux_files + "/" + aux_name + "_DNN_model_{}.zip".format(i + 1), warnings=False)
                )

            # Load the ensemble keras model
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
            name: str = "EnsembleDNN_Utility",
        ) -> np.ndarray:
        """Get the utility of each alternative for a given set of observations.

        Args:
            x: The input data. It can be a tf.Tensor, np.ndarray or pd.DataFrame.
            name: Name of the utility model. Default: 'EnsembleDNN_Utility'.

        Returns:
            Numpy array with the utility of each alternative for each observation in the input data.
        """
        # This method is not supported for ensemble models
        raise NotSupportedError("This method is not supported for ensemble models.")

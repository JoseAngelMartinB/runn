from runn.models.dnn import DNN
from runn.utils import WarningManager

# Initialize the warning manager
warning_manager = WarningManager()


class EnsembleDNN(DNN):
    def __init__(self, params: dict = None) -> None:
        """Ensemble of deep neural network models for choice modeling.

        Args:
            params: Dictionary with the model parameters.
        """

        super().__init__(params)

    def _initialize_ensemble_params(self) -> None:
        """Initialize the parameters of the ensemble model."""
        if "n_models" not in self.params:
            self.params["n_models"] = 5
            msg = "No 'n_models' parameter provided. Using default value: 5."
            warning_manager.warn(msg)
        if not isinstance(self.params["n_models"], int):
            msg = "The 'n_models' parameter should be an integer."
            raise ValueError(msg)
        if self.params["n_models"] < 2:
            msg = "The 'n_models' parameter should be greater than 1."
            raise ValueError(msg)

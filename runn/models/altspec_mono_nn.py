from runn.models.base import BaseModel
from runn.utils import WarningManager

# Initialize the warning manager
warning_manager = WarningManager()


class AltSpecMonoNN(BaseModel):
    def __init__(self, params: dict = None) -> None:
        """Alternative-specific monotonic neural network model for choice modeling.

        Args:
            params: Dictionary with the model parameters.
        """

        super().__init__(params)

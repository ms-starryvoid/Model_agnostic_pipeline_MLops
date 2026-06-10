# contracts/model_contract.py
from abc import ABC, abstractmethod
import numpy as np
import mlflow

class ModelContract(ABC):

    @abstractmethod
    def build(self, input_dim: int, **kwargs) -> "ModelContract":
        ...

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> dict:
        """Returns evaluation metrics dict."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def log_to_mlflow(self, run: mlflow.ActiveRun, artifact_path: str, 
                      processor=None, **kwargs) -> str:
        """
        Logs model + processor bundle to MLflow.
        processor: FeatureContract instance (optional, but should be provided)
        Returns model URI.
        """
        ...

    @property
    @abstractmethod
    def loader_tag(self) -> str:
        """Value for the MLflow 'loader' tag: 'sklearn', 'pytorch', etc."""
        ...

    @property
    @abstractmethod
    def adapter_tag(self) -> str:
        """Value for the MLflow 'adapter' tag: matches a key in AdapterRegistry."""
        ...
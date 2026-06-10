# serving/usecase_adapter.py
from abc import ABC, abstractmethod
from contracts.usecase_groups import InputSchema, OutputSchema
import numpy as np

class UsecaseAdapter(ABC):
    """
    One adapter instance lives per usecase in the registry.
    On champion swap, a new adapter is instantiated and replaces the old one.
    Preprocess + predict + postprocess are the only methods the router calls.
    """

    @property
    @abstractmethod
    def usecase_name(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> InputSchema: ...

    @property
    @abstractmethod
    def output_schema(self) -> OutputSchema: ...

    @property
    @abstractmethod
    def target_column(self) -> str: ...

    @property
    @abstractmethod
    def training_schema(self) -> dict: ...

    @abstractmethod
    def load_champion(self) -> None:
        """Pull champion from MLflow. Called on init and on hot-swap."""
        ...

    @abstractmethod
    def preprocess(self, raw_record: dict) -> np.ndarray:
        """Validate + transform one raw dict into feature array."""
        ...

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def postprocess(self, raw_output: np.ndarray) -> dict:
        """Map raw model output → typed response dict matching output_schema."""
        ...

    def run(self, raw_record: dict) -> dict:
        """Full inference pipeline. Called by the router."""
        features = self.preprocess(raw_record)
        raw_out  = self.predict(features)
        return self.postprocess(raw_out)
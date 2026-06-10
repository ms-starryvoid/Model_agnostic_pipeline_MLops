from abc import ABC, abstractmethod
from contracts.usecase_groups import InputSchema, OutputSchema
from contracts.feature_contract import FeatureContract
from contracts.model_contract import ModelContract
import numpy as np


class UsecaseAdapter(ABC):

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
    def target_column(self) -> str:
        """Column name the pipeline uses as label during training."""
        ...

    @property
    @abstractmethod
    def models(self) -> list[ModelContract]:
        """All model architectures that compete for this usecase."""
        ...

    @property
    @abstractmethod
    def processor(self) -> FeatureContract:
        """Fresh (unfitted) processor instance for this usecase."""
        ...

    @abstractmethod
    def load_champion(self) -> None: ...

    @abstractmethod
    def preprocess(self, raw_record: dict) -> np.ndarray: ...

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def postprocess(self, raw_output: np.ndarray) -> dict: ...

    def run(self, raw_record: dict) -> dict:
        features = self.preprocess(raw_record)
        raw_out  = self.predict(features)
        return self.postprocess(raw_out)
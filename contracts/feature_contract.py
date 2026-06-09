# contracts/feature_contract.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class FeatureContract(ABC):
    """
    A feature processor is fit once (on training data) and
    transforms one record at a time at inference.
    It must be serialisable (pickle/joblib) so it can be
    bundled into the MLflow artifact alongside the model.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "FeatureContract":
        """Fit scalers, encoders, etc. Returns self for chaining."""
        ...

    @abstractmethod
    def transform(self, record: dict) -> np.ndarray:
        """
        Transform ONE raw record dict → feature array.
        Shape must match what the model was trained on.
        This is called at inference time — must be stateless after fit.
        """
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Ordered list of feature names after transformation."""
        ...
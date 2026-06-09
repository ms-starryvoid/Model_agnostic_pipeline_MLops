import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch

from contracts.usecase_groups import InputSchema, OutputSchema
from usecases.classification.groups import TitanicClassificationGroup
from usecases.classification.processor import TitanicFeatureProcessor
from serving.usecase_adapter import UsecaseAdapter


_GROUP = TitanicClassificationGroup()


class TitanicClassificationAdapter(UsecaseAdapter):

    @property
    def usecase_name(self) -> str:
        return _GROUP.usecase_name

    @property
    def input_schema(self) -> InputSchema:
        return _GROUP.input_schema

    @property
    def output_schema(self) -> OutputSchema:
        return _GROUP.output_schema

    # ------------------------------------------------------------------
    # Champion loading  (called on init + every hot-swap)
    # ------------------------------------------------------------------

    def load_champion(self) -> None:
        model_name = f"{self.usecase_name}"
        artifact_uri = f"models:/{model_name}@champion"

        # Download artifact directory to a temp location
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri, dst_path=tmp
            )
            local_dir = Path(local_dir)

            # Load processor (fit state baked in at training time)
            self._processor: TitanicFeatureProcessor = joblib.load(
                local_dir / "processor.joblib"
            )

            # Load PyTorch weights — champion tag tells us the class
            state = torch.load(
                local_dir / "model.pt", map_location="cpu", weights_only=True
            )
            input_dim = state["net.0.weight"].shape[1]
            hidden_dim = state["net.0.weight"].shape[0]

            # Import here to avoid circular deps at module level
            from usecases.classification.models import ShallowMLPModel
            net = ShallowMLPModel(input_dim=input_dim, hidden_dim=hidden_dim)
            net.model.load_state_dict(state)
            net.model.eval()
            self._model = net

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------

    def preprocess(self, raw_record: dict) -> np.ndarray:
        return self._processor.transform(raw_record)   # (1, 7) float32

    def predict(self, features: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(features)             # (1, 7)
        with torch.no_grad():
            prob = self._model.model(tensor)            # (1, 1)
        return prob.numpy()                             # (1, 1) float32

    def postprocess(self, raw_output: np.ndarray) -> dict:
        probability = float(raw_output[0, 0])
        return {
            "survived":    probability >= 0.5,
            "probability": round(probability, 4),
        }
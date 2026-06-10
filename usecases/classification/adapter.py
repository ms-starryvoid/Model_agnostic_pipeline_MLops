import tempfile
from pathlib import Path
import logging

import joblib
import mlflow
import numpy as np
import torch

from contracts.usecase_groups import InputSchema, OutputSchema
from usecases.classification.groups import TitanicClassificationGroup
from usecases.classification.processor import TitanicFeatureProcessor
from serving.usecase_adapter import UsecaseAdapter

logger = logging.getLogger(__name__)

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

    @property
    def target_column(self) -> str:
        return "survived"

    @property
    def training_schema(self) -> dict:
        schema = self.input_schema.to_json_schema()
        schema["properties"][self.target_column] = {
            "anyOf": [{"type": "boolean"}, {"type": "integer"}]
        }
        if self.target_column not in schema["required"]:
            schema["required"].append(self.target_column)
        schema["description"] = (
            "CSV training row with input feature columns plus the target label. "
            "The target may be encoded as 0/1 or true/false."
        )
        return schema

    # ------------------------------------------------------------------
    # Champion loading  (called on init + every hot-swap)
    # ------------------------------------------------------------------

    def load_champion(self) -> None:
        model_name = f"{self.usecase_name}_models"
        artifact_uri = f"models:/{model_name}@champion"

        # Download artifact directory to a temp location
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri, dst_path=tmp
            )
            local_dir = Path(local_dir)

            # MLflow pytorch artifacts are nested: model/data/model.pt and model/processor.joblib
            # depending on how they were logged
            model_dir = local_dir / "model" if (local_dir / "model").exists() else local_dir
            
            # Load processor (fit state baked in at training time)
            processor_path = model_dir / "processor.joblib"
            if not processor_path.exists():
                raise FileNotFoundError(
                    f"processor.joblib not found at {processor_path}. "
                    f"Contents: {list(model_dir.glob('*'))}"
                )
            
            self._processor: TitanicFeatureProcessor = joblib.load(processor_path)
            logger.info(f"Loaded processor from {processor_path}")

            # Load PyTorch weights — MLflow saves as model.pt in the artifact directory
            model_pt_path = model_dir / "model.pt"
            # Also check for PyTorch serialized format
            if not model_pt_path.exists():
                # Try data directory (MLflow PyTorch standard structure)
                model_pt_path = model_dir / "data" / "model.pt"
            
            if not model_pt_path.exists():
                raise FileNotFoundError(
                    f"model.pt not found. Tried: {model_dir / 'model.pt'}, "
                    f"{model_dir / 'data' / 'model.pt'}. "
                    f"Contents: {list(model_dir.glob('**/*'))}"
                )

            state = torch.load(
                model_pt_path, map_location="cpu", weights_only=True
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
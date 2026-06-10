import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import joblib
import logging
import tempfile
from pathlib import Path

from contracts.model_contract import ModelContract

logger = logging.getLogger(__name__)


class ShallowMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ShallowMLPModel(ModelContract):

    def __init__(self):
        self.model = None

    def build(self, input_dim: int, hidden_dim: int = 64, **kwargs):
        self.model = ShallowMLP(input_dim, hidden_dim)
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        lr: float = 1e-3,
        **kwargs
    ) -> dict:
        print("Any NaNs in X?", np.isnan(X_train).any())
        print("NaN count:", np.isnan(X_train).sum())
        print("Any inf?", np.isinf(X_train).any())
        X = torch.tensor(X_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()

        for _ in range(epochs):
            optimizer.zero_grad()

            outputs = self.model(X)
            print("Output range:", outputs.min().item(), outputs.max().item())
            print("Target range:", y.min().item(), y.max().item())
            print("Unique targets:", torch.unique(y))
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

        return {"loss": loss.item()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            preds = self.model(X_tensor)

        return preds.numpy()

    def log_to_mlflow(
        self,
        run: mlflow.ActiveRun,
        artifact_path: str,
        processor=None,
        **kwargs
    ) -> str:
        """Log both model weights AND processor to MLflow."""
        
        # Save model weights using MLflow PyTorch
        scripted_model = torch.jit.script(self.model)
        mlflow.pytorch.log_model(
            pytorch_model=scripted_model,
            artifact_path=artifact_path
        )

        # Save processor if provided (CRITICAL for inference)
        if processor is not None:
            # Create temp file for processor, then log as artifact
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                joblib.dump(processor, tmp_path)
                # Log the file as an artifact in the artifact_path
                mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
                logger.info(f"Logged processor to MLflow artifact: {artifact_path}/processor.joblib")
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
        else:
            logger.warning("No processor provided to log_to_mlflow()")

        return f"runs:/{run.info.run_id}/{artifact_path}"

    @property
    def loader_tag(self) -> str:
        return "pytorch"

    @property
    def adapter_tag(self) -> str:
        return "shallow_mlp"


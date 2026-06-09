# pipeline/group_training_pipeline.py
import mlflow
from contracts.usecase_groups import UsecaseGroupContract
from contracts.model_contract import ModelContract
from contracts.feature_contract import FeatureContract
import pandas as pd
import numpy as np

class GroupTrainingPipeline:
    """
    Generic pipeline for any UsecaseGroupContract.
    Trains all provided model variants, evaluates each,
    promotes the best to champion under the group's usecase alias.
    """

    def __init__(
        self,
        group: UsecaseGroupContract,
        experiment_name: str | None = None,
    ):
        self.group = group
        self.experiment_name = experiment_name or f"group_{group.usecase_name}"

    def run(
        self,
        models: list[ModelContract],
        processor: FeatureContract,
        raw_df: pd.DataFrame,
        target_col: str,
        metric_to_maximize: str = "f1",
    ) -> str:
        """
        Returns the MLflow model URI of the promoted champion.
        """
        mlflow.set_experiment(self.experiment_name)
        X = processor.fit(raw_df.drop(columns=[target_col]))
        # transform full frame for train/val split
        X_arr = np.vstack([processor.transform(row) for row in raw_df.drop(columns=[target_col]).to_dict("records")])
        y_arr = raw_df[target_col].values

        best_score, best_uri, best_run_id = -np.inf, None, None

        for model in models:
            with mlflow.start_run(run_name=type(model).__name__) as run:
                mlflow.set_tags({
                    **self.group.mlflow_tags,
                    "loader":      model.loader_tag,
                    "adapter":     model.adapter_tag,
                    "model_class": type(model).__name__,
                })
                model.build(input_dim=X_arr.shape[1])
                metrics = model.train(X_arr, y_arr)
                mlflow.log_metrics(metrics)
                uri = model.log_to_mlflow(run, artifact_path="model")

                score = metrics.get(metric_to_maximize, -np.inf)
                if score > best_score:
                    best_score, best_uri, best_run_id = score, uri, run.info.run_id

        # Register + alias best model as champion for this usecase
        model_name = f"{self.group.usecase_name}_models"
        result = mlflow.register_model(best_uri, model_name)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(model_name, "champion", result.version)
        return best_uri
# api/routes.py
from fastapi import APIRouter, HTTPException, File, Form, Request, UploadFile
from serving.adapter_registry import UsecaseAdapterRegistry
import pandas as pd
import io


def _match_column_name(columns: list[str], expected: str) -> str | None:
    lower_expected = expected.lower()
    for col in columns:
        if col.lower() == lower_expected:
            return col
    return None


def build_router(registry: UsecaseAdapterRegistry) -> APIRouter:
    router = APIRouter()

    @router.get("/usecases")
    def list_usecases():
        """List all registered usecases with champion status."""
        result = []
        for uc in registry.all_usecases():
            status = "champion_loaded" if registry.is_champion_loaded(uc) else "no_champion"
            try:
                adapter = registry.get(uc)
                result.append({
                    "name": uc,
                    "status": status,
                    "input_schema":  adapter.input_schema.to_json_schema(),
                    "output_schema": adapter.output_schema.to_json_schema(),
                })
            except RuntimeError:
                # No champion loaded yet
                adapter = registry.get_unsafe(uc)
                result.append({
                    "name": uc,
                    "status": status,
                    "input_schema":  adapter.input_schema.to_json_schema(),
                    "output_schema": adapter.output_schema.to_json_schema(),
                    "message": "No champion loaded. POST /api/v1/train/{usecase} to train."
                })
        return {"usecases": result}

    @router.get("/schema/{usecase}")
    def get_schema(usecase: str):
        """Return the expected input + output schema for a specific usecase."""
        try:
            adapter = registry.get_unsafe(usecase)
            return {
                "usecase": usecase,
                "status": "champion_loaded" if registry.is_champion_loaded(usecase) else "no_champion",
                "input":  adapter.input_schema.to_json_schema(),
                "output": adapter.output_schema.to_json_schema(),
            }
        except KeyError:
            raise HTTPException(404, detail=f"Usecase '{usecase}' not found")

    @router.get("/train-schema/{usecase}")
    def get_training_schema(usecase: str):
        """Return the expected training data schema for a specific usecase."""
        try:
            adapter = registry.get_unsafe(usecase)
            return {
                "usecase": usecase,
                "status": "champion_loaded" if registry.is_champion_loaded(usecase) else "no_champion",
                "target_column": adapter.target_column,
                "training_schema": adapter.training_schema,
            }
        except KeyError:
            raise HTTPException(404, detail=f"Usecase '{usecase}' not found")

    @router.post("/predict/{usecase}")
    def predict(usecase: str, record: dict):
        """Run inference for a single record against the champion of a usecase."""
        try:
            adapter = registry.get(usecase)  # Will fail if no champion loaded
        except KeyError:
            raise HTTPException(404, detail=f"Usecase '{usecase}' not found")
        except RuntimeError as e:
            raise HTTPException(503, detail=str(e))
        try:
            return {"usecase": usecase, "prediction": adapter.run(record)}
        except Exception as e:
            raise HTTPException(422, detail=str(e))

    @router.post("/train/{usecase}")
    async def train_usecase(
        usecase: str,
        request: Request,
        csv_file: UploadFile | None = File(None),
        target_col: str | None = Form(None),
    ):
        """
        Train + register a champion model for a usecase.
        Expects CSV training data as multipart file upload or raw CSV body.
        If target column is omitted, the usecase default target column is used.
        """
        if usecase not in registry.all_usecases():
            raise HTTPException(404, detail=f"Usecase '{usecase}' not found")

        try:
            adapter = registry.get_unsafe(usecase)
            if target_col is None:
                target_col = adapter.target_column

            if csv_file is not None:
                contents = await csv_file.read()
            else:
                contents = await request.body()

            if not contents:
                raise HTTPException(422, detail="No CSV training data received.")

            try:
                df = pd.read_csv(io.BytesIO(contents))
            except Exception as parse_exc:
                raise HTTPException(422, detail=f"Could not parse CSV training data: {parse_exc}")

            matched_target = _match_column_name(list(df.columns), target_col)
            if matched_target is None:
                raise HTTPException(
                    422,
                    detail=(
                        f"Target column '{target_col}' not found in CSV. "
                        f"Available columns: {list(df.columns)}"
                    ),
                )
            target_col = matched_target

            # Import here to avoid circular dependency
            from usecases.classification.groups import TitanicClassificationGroup
            from usecases.classification.processor import TitanicFeatureProcessor
            from usecases.classification.models import ShallowMLPModel
            from pipeline.group_training_pipeline import GroupTrainingPipeline

            # Hardcoded for now (next step: parameterize by adapter_tag)
            group = TitanicClassificationGroup()
            processor = TitanicFeatureProcessor()
            models = [ShallowMLPModel()]

            pipeline = GroupTrainingPipeline(group)
            best_uri = pipeline.run(
                models=models,
                processor=processor,
                raw_df=df,
                target_col=target_col,
                metric_to_maximize="loss"  # minimize loss
            )

            # Reload champion into registry
            registry.hot_swap(usecase)

            return {
                "usecase": usecase,
                "target_column": target_col,
                "message": "Training completed successfully",
                "best_model_uri": best_uri,
                "champion_loaded": registry.is_champion_loaded(usecase),
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, detail=f"Training failed: {str(e)}")

    return router
# api/routes.py
from fastapi import APIRouter, HTTPException, File, Form, Request, UploadFile
from pipeline.group_training_pipeline import GroupTrainingPipeline
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
        if usecase not in registry.all_usecases():
            raise HTTPException(404, detail=f"Usecase '{usecase}' not found")

        try:
            # Adapter is the only thing the route knows about
            adapter    = registry.get_unsafe(usecase)
            target_col = target_col or adapter.target_column

            # --- parse CSV ---
            contents = await csv_file.read() if csv_file else await request.body()
            if not contents:
                raise HTTPException(422, detail="No CSV training data received.")
            try:
                df = pd.read_csv(io.BytesIO(contents))
            except Exception as e:
                raise HTTPException(422, detail=f"Could not parse CSV: {e}")

            matched = _match_column_name(list(df.columns), target_col)
            if matched is None:
                raise HTTPException(
                    422,
                    detail=f"Target column '{target_col}' not found. "
                           f"Available: {list(df.columns)}",
                )

            # --- training — group, processor, models all come from the adapter ---
            pipeline = GroupTrainingPipeline(adapter.group)
            best_uri = pipeline.run(
                models=adapter.models,          # all architectures for this usecase
                processor=adapter.processor,    # fresh unfitted instance
                raw_df=df,
                target_col=matched,
                metric_to_maximize="loss",
            )

            registry.hot_swap(usecase)

            return {
                "usecase":         usecase,
                "target_column":   matched,
                "models_trained":  [type(m).__name__ for m in adapter.models],
                "message":         "Training completed successfully",
                "best_model_uri":  best_uri,
                "champion_loaded": registry.is_champion_loaded(usecase),
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, detail=f"Training failed: {e}")

    return router
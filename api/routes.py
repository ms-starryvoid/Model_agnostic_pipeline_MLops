# api/routes.py
from fastapi import APIRouter, HTTPException
from serving.adapter_registry import UsecaseAdapterRegistry

def build_router(registry: UsecaseAdapterRegistry) -> APIRouter:
    router = APIRouter()

    @router.get("/usecases")
    def list_usecases():
        """List all registered usecases and their champion model info."""
        return {
            "usecases": [
                {
                    "name": uc,
                    "input_schema":  registry.get(uc).input_schema.to_json_schema(),
                    "output_schema": registry.get(uc).output_schema.to_json_schema(),
                }
                for uc in registry.all_usecases()
            ]
        }

    @router.get("/schema/{usecase}")
    def get_schema(usecase: str):
        """Return the expected input + output schema for a specific usecase."""
        try:
            adapter = registry.get(usecase)
        except KeyError:
            raise HTTPException(404, detail=f"Usecase '{usecase}' not found")
        return {
            "usecase": usecase,
            "input":  adapter.input_schema.to_json_schema(),
            "output": adapter.output_schema.to_json_schema(),
        }

    @router.post("/predict/{usecase}")
    def predict(usecase: str, record: dict):
        """Run inference for a single record against the champion of a usecase."""
        try:
            adapter = registry.get(usecase)
        except KeyError:
            raise HTTPException(404, detail=f"Usecase '{usecase}' not found")
        try:
            return {"usecase": usecase, "prediction": adapter.run(record)}
        except Exception as e:
            raise HTTPException(422, detail=str(e))

    return router
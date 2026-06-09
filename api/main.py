from fastapi import FastAPI
import mlflow
from registry import build_registry
from serving.champion_watcher import ChampionWatcher
from api.routes import build_router
from config import settings

def create_app() -> FastAPI:
    app = FastAPI(title="MLOps Serving API", version="1.0.0")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    registry = build_registry()

    watcher = ChampionWatcher(registry, poll_interval_seconds=30)
    watcher.start()

    app.include_router(build_router(registry), prefix="/api/v1")

    @app.get("/health")
    def health():
        return {"status": "ok", "usecases": registry.all_usecases()}

    return app


app = create_app()
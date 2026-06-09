import logging
import threading
import time

import mlflow

from serving.adapter_registry import UsecaseAdapterRegistry

logger = logging.getLogger(__name__)


class ChampionWatcher:
    """
    Background daemon thread. On every poll cycle it checks whether
    the champion version has changed for each registered usecase.
    If it has, it hot-swaps that adapter only — others are untouched.
    """

    def __init__(self, registry: UsecaseAdapterRegistry, poll_interval_seconds: int = 30):
        self._registry = registry
        self._interval = poll_interval_seconds
        self._client   = mlflow.MlflowClient()

        # Track last known champion version per usecase
        self._versions: dict[str, str] = {}

        self._thread = threading.Thread(
            target=self._poll_loop,
            name="ChampionWatcher",
            daemon=True,   # dies with the main process, no cleanup needed
        )

    def start(self) -> None:
        # Snapshot current versions before first poll so we don't
        # hot-swap unnecessarily on startup
        for usecase in self._registry.all_usecases():
            self._versions[usecase] = self._fetch_champion_version(usecase)
        self._thread.start()
        logger.info("ChampionWatcher started (interval=%ds)", self._interval)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while True:
            time.sleep(self._interval)
            for usecase in self._registry.all_usecases():
                try:
                    self._check(usecase)
                except Exception:
                    # Never let one bad usecase kill the watcher thread
                    logger.exception("ChampionWatcher error checking '%s'", usecase)

    def _check(self, usecase: str) -> None:
        current = self._fetch_champion_version(usecase)
        if current != self._versions.get(usecase):
            logger.info(
                "Champion changed for '%s': %s → %s. Hot-swapping.",
                usecase,
                self._versions.get(usecase),
                current,
            )
            self._registry.hot_swap(usecase)
            self._versions[usecase] = current

    def _fetch_champion_version(self, usecase: str) -> str | None:
        model_name = f"{usecase}_models"
        try:
            mv = self._client.get_model_version_by_alias(model_name, "champion")
            return mv.version
        except mlflow.exceptions.MlflowException:
            # Model not yet registered — not an error, just not ready
            logger.debug("No champion found for '%s' yet.", usecase)
            return None
# serving/adapter_registry.py
import logging
from serving.usecase_adapter import UsecaseAdapter

logger = logging.getLogger(__name__)


class UsecaseAdapterRegistry:
    def __init__(self):
        self._adapters: dict[str, UsecaseAdapter] = {}
        self._champion_loaded: dict[str, bool] = {}

    def register(self, adapter: UsecaseAdapter) -> None:
        """Register adapter. Champion loading is deferred + optional."""
        self._adapters[adapter.usecase_name] = adapter
        self._champion_loaded[adapter.usecase_name] = False
        try:
            adapter.load_champion()
            self._champion_loaded[adapter.usecase_name] = True
            logger.info(f"Champion loaded for '{adapter.usecase_name}'")
        except Exception as e:
            logger.warning(
                f"No champion found for '{adapter.usecase_name}' — API will accept "
                f"/train requests. Error: {e}"
            )

    def get(self, usecase: str) -> UsecaseAdapter:
        if usecase not in self._adapters:
            raise KeyError(f"No adapter registered for usecase '{usecase}'")
        if not self._champion_loaded[usecase]:
            raise RuntimeError(
                f"Champion not yet loaded for '{usecase}'. "
                f"POST /api/v1/train/{usecase} to train first."
            )
        return self._adapters[usecase]

    def get_unsafe(self, usecase: str) -> UsecaseAdapter:
        """Get adapter even if champion not loaded (for training endpoints)."""
        if usecase not in self._adapters:
            raise KeyError(f"No adapter registered for usecase '{usecase}'")
        return self._adapters[usecase]

    def all_usecases(self) -> list[str]:
        return list(self._adapters.keys())

    def is_champion_loaded(self, usecase: str) -> bool:
        return self._champion_loaded.get(usecase, False)

    def hot_swap(self, usecase: str) -> None:
        """Re-load champion for one usecase without touching others."""
        try:
            self._adapters[usecase].load_champion()
            self._champion_loaded[usecase] = True
            logger.info(f"Champion hot-swapped for '{usecase}'")
        except Exception as e:
            logger.error(f"Failed to hot-swap champion for '{usecase}': {e}")
            self._champion_loaded[usecase] = False
# serving/adapter_registry.py
from serving.usecase_adapter import UsecaseAdapter

class UsecaseAdapterRegistry:
    def __init__(self):
        self._adapters: dict[str, UsecaseAdapter] = {}

    def register(self, adapter: UsecaseAdapter) -> None:
        adapter.load_champion()
        self._adapters[adapter.usecase_name] = adapter

    def get(self, usecase: str) -> UsecaseAdapter:
        if usecase not in self._adapters:
            raise KeyError(f"No adapter registered for usecase '{usecase}'")
        return self._adapters[usecase]

    def all_usecases(self) -> list[str]:
        return list(self._adapters.keys())

    def hot_swap(self, usecase: str) -> None:
        """Re-load champion for one usecase without touching others."""
        self._adapters[usecase].load_champion()
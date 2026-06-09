from serving.adapter_registry import UsecaseAdapterRegistry
from usecases.classification.adapter import TitanicClassificationAdapter


def build_registry() -> UsecaseAdapterRegistry:
    registry = UsecaseAdapterRegistry()
    registry.register(TitanicClassificationAdapter())
    # registry.register(AnomalyDetectionAdapter())  ← only file that ever changes
    return registry
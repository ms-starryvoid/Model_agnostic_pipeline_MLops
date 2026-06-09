# contracts/usecase_group.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

@dataclass
class InputSchema:
    """Strict definition of what one record must look like before preprocessing."""
    fields: dict[str, type]          # {"temperature": float, "pressure": float}
    description: str = ""
    example: dict = field(default_factory=dict)

    def to_json_schema(self) -> dict:
        type_map = {float: "number", int: "integer", str: "string", bool: "boolean"}
        return {
            "type": "object",
            "description": self.description,
            "properties": {k: {"type": type_map[v]} for k, v in self.fields.items()},
            "required": list(self.fields.keys()),
            "example": self.example,
        }

@dataclass
class OutputSchema:
    """Strict definition of what the model returns after postprocessing."""
    fields: dict[str, type]
    description: str = ""

    def to_json_schema(self) -> dict:
        type_map = {float: "number", int: "integer", str: "string", bool: "boolean"}
        return {
            "type": "object",
            "properties": {k: {"type": type_map[v]} for k, v in self.fields.items()},
        }

class UsecaseGroupContract(ABC):
    """
    Every use-case group must declare its name, schemas, and
    the feature processor + model variants that belong to it.
    This is the single source of truth for what a group IS.
    """

    @property
    @abstractmethod
    def usecase_name(self) -> str:
        """Unique slug: 'anomaly_detection', 'classification', etc."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> InputSchema:
        """What raw input this group expects (before feature processing)."""
        ...

    @property
    @abstractmethod
    def output_schema(self) -> OutputSchema:
        """What this group's models return after postprocessing."""
        ...

    @property
    def mlflow_tags(self) -> dict[str, str]:
        """Tags written to every model registered under this group."""
        return {
            "usecase": self.usecase_name,
            "input_schema": str(self.input_schema.to_json_schema()),
            "output_schema": str(self.output_schema.to_json_schema()),
        }
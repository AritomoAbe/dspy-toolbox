from abc import ABC, abstractmethod
from typing import Any

import dspy

INVALID_SCORE = -1


class ScoreExtractor(ABC):
    @abstractmethod
    def extraction_metric(self, example: dspy.Example, prediction: dspy.Prediction, trace: Any = None) -> float:
        ...

    @abstractmethod
    def field_metric(self, field: str, expected_val: Any, predicted_val: Any) -> bool:
        ...

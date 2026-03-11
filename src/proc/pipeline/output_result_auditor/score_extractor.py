from abc import ABC, abstractmethod
from typing import Any

import dspy


class ScoreExtractor(ABC):
    @abstractmethod
    def extraction_metric(self, example: dspy.Example, prediction: dspy.Prediction, trace: Any = None) -> float:
        pass

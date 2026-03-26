from abc import ABC, abstractmethod

import dspy


class BaseDataset(ABC):
    @abstractmethod
    def load(self) -> list[dspy.Example]:
        ...

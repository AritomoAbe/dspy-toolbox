import abc
from typing import Protocol

from returns.result import Result

from proc.base.proc_error import ProcError
from proc.base.proc_score import ProcScore


class ProcNode(Protocol):
    @abc.abstractmethod
    def invoke(self) -> Result[ProcScore, ProcError]:
        ...

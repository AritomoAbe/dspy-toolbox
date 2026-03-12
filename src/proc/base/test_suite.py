import logging

from returns.pipeline import is_successful
from returns.result import Result, Success

from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore


class TestSuite:
    _SCORE: float = 1.0

    def __init__(self, nodes: list[ProcNode]) -> None:
        self._logger = logging.getLogger(__name__)
        self._nodes = nodes

    def run(self) -> Result[ProcScore, ProcError]:
        for node in self._nodes:
            self._logger.info(f'Running {node.__class__.__name__}')
            res = node.invoke()
            if is_successful(res):
                self._logger.info(f'OK {node.__class__.__name__}')
                continue
            self._logger.error(f'FAILED {node.__class__.__name__}: {res}')
            return res
        return Success(ProcScore(value=self._SCORE))

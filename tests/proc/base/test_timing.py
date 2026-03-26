import logging
import time

import pytest

from proc.base.timing import timed

_LOGGER: logging.Logger = logging.getLogger(__name__)
_FAST_HEARTBEAT: float = 0.001
_SHORT_SLEEP: float = 0.02


class TestTimedContextManager:

    def test_yields_without_error(self) -> None:
        with timed("test_op", logger=_LOGGER, heartbeat_interval=0):
            pass

    def test_block_executes_normally(self) -> None:
        result: list[int] = []
        with timed("test_op", logger=_LOGGER, heartbeat_interval=0):
            result.append(1)
        assert result == [1]

    def test_exception_propagates(self) -> None:
        with pytest.raises(ValueError):
            with timed("test_op", logger=_LOGGER, heartbeat_interval=0):
                raise ValueError("expected")

    def test_heartbeat_path_runs(self) -> None:
        with timed("test_op", logger=_LOGGER, heartbeat_interval=_FAST_HEARTBEAT):
            time.sleep(_SHORT_SLEEP)

    def test_heartbeat_disabled_when_zero(self) -> None:
        with timed("test_op", logger=_LOGGER, heartbeat_interval=0):
            time.sleep(0.001)

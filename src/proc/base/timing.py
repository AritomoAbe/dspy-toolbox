"""
Lightweight timing utilities.
"""
import logging
import threading
import time
from contextlib import contextmanager
from typing import Generator

_logger = logging.getLogger(__name__)

_DEFAULT_HEARTBEAT_INTERVAL: float = 30.0


def _heartbeat(
    stop_event: threading.Event,
    t0: float,
    logger: logging.Logger,
    label: str,
    heartbeat_interval: float,
) -> None:
    while not stop_event.wait(heartbeat_interval):
        elapsed = time.perf_counter() - t0
        logger.info("[timing] %s: still running (%.0fs elapsed)", label, elapsed)


def _stop_and_log(
    stop_event: threading.Event,
    thread: threading.Thread | None,
    t0: float,
    logger: logging.Logger,
    label: str,
) -> None:
    stop_event.set()
    if thread is not None:
        thread.join()
    logger.info("[timing] %s: %.3fs", label, time.perf_counter() - t0)


@contextmanager
def timed(
    label: str,
    logger: logging.Logger = _logger,
    heartbeat_interval: float = _DEFAULT_HEARTBEAT_INTERVAL,
) -> Generator[None, None, None]:
    """Context manager that logs wall-clock duration of a block at INFO level.

    For long-running operations it also emits a periodic heartbeat so you can
    see progress without waiting for the block to finish:

        [timing] my_operation: still running (30s elapsed)
        [timing] my_operation: still running (60s elapsed)
        [timing] my_operation: 87.412s

    Parameters
    ----------
    label:
        Human-readable name for the operation, included in every log line.
    logger:
        Logger to write to. Pass ``logger=self._logger`` so messages are
        routed through the caller's logger hierarchy.
    heartbeat_interval:
        Seconds between heartbeat log lines while the block is still running.
        Default: 30s. Set to 0 to disable heartbeats entirely.

    Usage::

        from proc.base.timing import timed

        with timed("my_operation", logger=self._logger):
            ...

        # Faster heartbeat for very long ops:
        with timed("lig.attribute", logger=self._logger, heartbeat_interval=10):
            attributions, delta = lig.attribute(...)
    """
    t0 = time.perf_counter()
    stop_event = threading.Event()
    thread: threading.Thread | None = None
    if heartbeat_interval > 0:
        thread = threading.Thread(
            target=_heartbeat,
            args=(stop_event, t0, logger, label, heartbeat_interval),
            daemon=True,
        )
        thread.start()
    try:
        yield
    finally:
        _stop_and_log(stop_event, thread, t0, logger, label)

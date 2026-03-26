import logging
from abc import ABCMeta, abstractmethod

import dspy
from returns.result import Result

from proc.base.base_llm import BaseLLM, IN, OUT, BaseLLMConfig
from proc.base.proc_error import ProcError


class _DSpyABCMeta(type(dspy.Module), ABCMeta):  # type: ignore[misc]  # noqa: WPS606
    """Combined metaclass that satisfies both dspy.Module and ABCMeta."""


class DSpyLLM(dspy.Module, BaseLLM[IN, OUT], metaclass=_DSpyABCMeta):

    def __init__(self, config: BaseLLMConfig) -> None:
        self._logger = logging.getLogger(__name__)
        dspy.Module.__init__(self)
        BaseLLM.__init__(self, config=config)
        self._dspy_configured = False

    @abstractmethod
    def invoke(self, payload: IN) -> Result[OUT, ProcError]:
        ...

    @abstractmethod
    def get_demos(self, module: dspy.Module) -> Result[list[dspy.Example], ProcError]:  # noqa: WPS463
        ...

    def _ensure_dspy_configured(self) -> None:
        """
        Lazily wire DSPy to the local Ollama instance on first invoke().
        Deferred to avoid the ~10 s startup probe during object construction.
        """
        if self._dspy_configured:
            return
        lm = dspy.LM(
            model=f"ollama/{self._config.name.value}",
            api_base=self._config.base_url,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        dspy.configure(lm=lm)
        self._dspy_configured = True
        self._logger.info("DSPy configured with Ollama model=%s base_url=%s",
                          self._config.name.value, self._config.base_url)

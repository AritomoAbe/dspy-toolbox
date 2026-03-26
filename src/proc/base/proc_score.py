from typing import Optional


class ProcScoreContext:
    def __init__(self) -> None:
        ...


class ProcScore:

    def __init__(self, value: float, context: Optional[ProcScoreContext] = None) -> None:
        self._value = value
        self._context = context

    @property
    def value(self) -> float:
        return self._value

    @property
    def context(self) -> Optional[ProcScoreContext]:
        return self._context

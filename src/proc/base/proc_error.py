from typing import Optional


class ProcError:
    _message: Optional[str]

    def __init__(self, message: Optional[str] = None) -> None:
        self._message = message

    def __str__(self) -> str:
        return f'{self.__class__.__name__}: {self._message}'

    @property
    def message(self) -> Optional[str]:
        return self._message

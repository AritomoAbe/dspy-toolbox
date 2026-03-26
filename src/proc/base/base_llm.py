import abc
from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict
from returns.result import Result

from proc.base.proc_error import ProcError


class MainModelNames(Enum):
    QWEN_2_5_05B_INSTRUCT = "qwen2.5:0.5b-instruct"
    QWEN_2_5_3B_INSTRUCT = "qwen2.5:3b-instruct"
    QWEN_2_5_7B_INSTRUCT = "qwen2.5:7b-instruct"
    QWEN_3_4B_INSTRUCT = "qwen3:4b-instruct"


class BaseLLMConfig(BaseModel):
    name: MainModelNames = MainModelNames.QWEN_3_4B_INSTRUCT
    base_url: str = 'http://localhost:11434'
    temperature: float = 0.0  # noqa: WPS358
    do_warmup: bool = False
    count_tokens: bool = True
    num_ctx: int = 8192
    top_p: float = 0.9
    response_failover: bool = False
    do_json_sanitize: bool = True
    max_tokens: int = 1024


class PromptPayLoad(BaseModel):
    """Base class for structured LLM prompt inputs."""


class PromptResponse(BaseModel):
    proc_time_sec: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    system_prompt_version: Optional[str] = None
    user_prompt_version: Optional[str] = None
    try_count: int = 1
    sanitization_count: Optional[int] = None

    model_config = ConfigDict(validate_assignment=True)


IN = TypeVar("IN", bound=PromptPayLoad)
OUT = TypeVar("OUT", bound=PromptResponse)


class BaseLLM(abc.ABC, Generic[IN, OUT]):

    def __init__(self, config: BaseLLMConfig) -> None:
        self._config = config

    @abc.abstractmethod
    def invoke(self, payload: IN) -> Result[OUT, ProcError]:
        ...

    @property
    def config(self) -> BaseLLMConfig:
        return self._config

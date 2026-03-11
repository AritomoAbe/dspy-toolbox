import json
import logging
from pathlib import Path
from typing import Any

import dspy
from returns.result import Result, Success, Failure

from proc.base.base_llm import BaseLLMConfig, PromptPayLoad
from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError

from proc.demos.meeting_invite.dspy_signatures import EmailToMeetingInfo
from proc.demos.meeting_invite.models import (
    EmailMeetingInfo, PreferredWindow, Urgency, Flexibility, _DEFAULT_DURATION, _UNKNOWN_TZ,
)

LOGGER = logging.getLogger(__name__)

_WINDOWS_KEY: str = "preferred_windows"


class MeetingInvitePayload(PromptPayLoad):
    email_from: str
    email_to: str
    email_body: str
    current_date: str


class MeetingInviteLLM(DSpyLLM[MeetingInvitePayload, EmailMeetingInfo]):  # type: ignore[type-var]

    def __init__(self, config: BaseLLMConfig, optimization: bool = True, tune: bool = False) -> None:
        self._logger = logging.getLogger(__name__)
        super().__init__(config=config)
        self.predict = dspy.ChainOfThought(EmailToMeetingInfo)

        if optimization:
            optimized_path = Path(__file__).parent / "optimized_extractor.json"
            if optimized_path.exists():
                self.load(str(optimized_path))
                self._logger.info("suggest_times: loaded optimised extractor from %s", optimized_path)

        if tune:
            self._ensure_dspy_configured()

    def get_demos(self, module: dspy.Module) -> Result[list[dspy.Example], ProcError]:
        all_predictors = module.named_predictors()
        if len(all_predictors) != 1:
            return Failure(ProcError(f"Incorrect number of predictors: {len(all_predictors)}"))
        name, predictor = all_predictors[0]
        return Success(predictor.demos)

    def forward(
        self,
        email_from: str,
        email_to: str,
        email_body: str,
        current_date: str,
    ) -> dspy.Prediction:
        return self.predict(
            email_from=email_from,
            email_to=email_to,
            email_body=email_body,
            current_date=current_date,
        )

    def invoke(self, payload: MeetingInvitePayload) -> Result[EmailMeetingInfo, ProcError]:
        self._ensure_dspy_configured()
        prediction = self.forward(
            email_from=payload.email_from,
            email_to=payload.email_to,
            email_body=payload.email_body,
            current_date=payload.current_date,
        )
        raw: str = prediction.extracted_json or ""
        return Success(self._parse(raw))

    @staticmethod
    def _parse(raw: str) -> EmailMeetingInfo:
        cleaned = _strip_fences(raw)
        try:
            return _dict_to_email_meeting_info(json.loads(cleaned))
        except (json.JSONDecodeError, Exception) as exc:
            LOGGER.warning("MeetingInfoExtractor: failed to parse LLM output: %s | raw=%r", exc, raw)
            return EmailMeetingInfo(
                sender_iana_timezone=_UNKNOWN_TZ,
                duration_minutes=_DEFAULT_DURATION,
                urgency=Urgency.FLEXIBLE,
                flexibility=Flexibility.FLEXIBLE,
                preferred_windows=[],
            )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return text


def _dict_to_email_meeting_info(data: dict[str, Any]) -> EmailMeetingInfo:
    raw_windows: list[Any] = data.get(_WINDOWS_KEY, [])
    windows: list[PreferredWindow] = []
    for w in raw_windows:
        try:
            windows.append(PreferredWindow.model_validate(w))
        except Exception:
            pass
    return EmailMeetingInfo.model_validate({**data, _WINDOWS_KEY: windows})

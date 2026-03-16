import logging

import dspy
from returns.pipeline import is_successful

from proc.base.base_llm import BaseLLMConfig
from proc.demos.meeting_invite.meeting_invite_extractor_llm import MeetingInviteLLM, MeetingInvitePayload

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

LOGGER = logging.getLogger(__name__)

class MeetingInviteAbeGPT(MeetingInviteLLM):
    def _ensure_dspy_configured(self) -> None:
        if self._dspy_configured:
            return
        lm = dspy.LM("openai/gpt-abe", api_base="http://localhost:8013/v1", api_key="none")
        dspy.configure(lm=lm)
        self._dspy_configured = True
        self._logger.info("DSPy configured with Ollama model=%s base_url=%s",
                          self._config.name.value, self._config.base_url)

config = BaseLLMConfig()
llm = MeetingInviteAbeGPT(config=config)

original_cache = dspy.settings.lm.cache
dspy.settings.lm.cache = False

try:
    res = llm.invoke(
        payload=MeetingInvitePayload(
            email_from="1@test.com",
            email_to="2@test.com",
            email_body="Hi Alex,\n\n"
                       "Let's catch up on the retrospective. I can do Tuesday at lunchtime for 1 hour. (I'm on Paris time.)\n\n"
                       "Cheers,"
                       "Chris",
            current_date="2024-01-08",
        )
    )

    assert is_successful(res)
    meeting_info = res.unwrap()

    LOGGER.info(f"Done. Extracted: {meeting_info}")
finally:
    dspy.settings.lm.cache = original_cache



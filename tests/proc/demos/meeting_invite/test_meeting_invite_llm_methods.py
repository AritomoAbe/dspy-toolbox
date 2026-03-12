from unittest.mock import MagicMock, patch

from returns.pipeline import is_successful

from proc.base.base_llm import BaseLLMConfig
from proc.demos.meeting_invite.meeting_invite_extractor_llm import MeetingInviteLLM, MeetingInvitePayload


class TestMeetingInviteLLMMethods:

    def test_get_demos_one_predictor_ok(self) -> None:
        llm = MeetingInviteLLM(config=BaseLLMConfig())
        module = MagicMock()
        module.named_predictors.return_value = [("predict", MagicMock(demos=[]))]
        assert is_successful(llm.get_demos(module))

    def test_get_demos_many_predictors_fails(self) -> None:
        llm = MeetingInviteLLM(config=BaseLLMConfig())
        module = MagicMock()
        module.named_predictors.return_value = [("p1", MagicMock()), ("p2", MagicMock())]
        assert not is_successful(llm.get_demos(module))

    def test_forward_delegates_to_predict(self) -> None:
        llm = MeetingInviteLLM(config=BaseLLMConfig())
        mock_pred = MagicMock()
        with patch.object(llm, "predict", return_value=mock_pred):
            result = llm.forward(email_from="a", email_to="b", email_body="c", current_date="d")
        assert result is mock_pred

    def test_invoke_returns_success(self) -> None:
        llm = MeetingInviteLLM(config=BaseLLMConfig())
        mock_prediction = MagicMock(extracted_json='{"preferred_windows": []}')
        payload = MeetingInvitePayload(
            email_from="a@b.com",
            email_to="c@d.com",
            email_body="Hello",
            current_date="2026-01-01",
        )
        with patch.object(llm, "_ensure_dspy_configured"):
            with patch.object(llm, "forward", return_value=mock_prediction):
                result = llm.invoke(payload)
        assert is_successful(result)

import json
from pathlib import Path
from typing import Any

import dspy

from proc.demos.meeting_invite.meeting_invite_extractor_llm import _dict_to_email_meeting_info
from proc.pipeline.dataset.base_dataset import BaseDataset

_ENUM_FIELDS: frozenset[str] = frozenset(("urgency", "flexibility"))


def _normalise_expected(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v.lower() if k in _ENUM_FIELDS and isinstance(v, str) else v
        for k, v in raw.items()
    }


class MeetingInviteDataset(BaseDataset):

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[dspy.Example]:
        lines = [ln for ln in self._path.read_text().splitlines() if ln.strip()]
        examples: list[dspy.Example] = []
        for line in lines:
            row: dict[str, Any] = json.loads(line)
            expected = _dict_to_email_meeting_info(_normalise_expected(row["expected"]))
            ex = dspy.Example(
                email_from=row["email_from"],
                email_to=row["email_to"],
                email_body=row["email_body"],
                current_date=row["current_date"],
                expected=expected,
            ).with_inputs("email_from", "email_to", "email_body", "current_date")
            examples.append(ex)
        return examples

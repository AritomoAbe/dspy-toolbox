import json
from pathlib import Path
from typing import Any

import dspy

from proc.pipeline.dataset.base_dataset import BaseDataset


class TrainingSetDataset(BaseDataset):
    _EXPECTED_KEY: str = "expected"

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: list[dspy.Example] | None = None

    def load(self) -> list[dspy.Example]:
        cached = self._cache
        if cached is None:
            cached = self._load_from_disk()
            self._cache = cached
        return cached

    def _load_from_disk(self) -> list[dspy.Example]:
        lines = [ln for ln in self._path.read_text().splitlines() if ln.strip()]
        examples: list[dspy.Example] = []
        for line in lines:
            row: dict[str, Any] = json.loads(line)
            inputs: dict[str, Any] = dict(row)
            expected = inputs.pop(self._EXPECTED_KEY, {})
            ex = dspy.Example(**inputs, expected=expected)
            examples.append(ex.with_inputs(*inputs.keys()))
        return examples

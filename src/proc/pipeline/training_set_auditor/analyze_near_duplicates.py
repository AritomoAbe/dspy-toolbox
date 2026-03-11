import json
from operator import attrgetter
from pathlib import Path
from typing import Any

import numpy as np
from returns.result import Failure, Result, Success
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.pipeline.training_set_auditor.models import FieldDuplicateStats, NearDuplicatePair

_DEFAULT_THRESHOLD: float = 0.85
_MAX_EXAMPLES: int = 5


class AnalyzeNearDuplicates(ProcNode):

    def __init__(self, path: Path, text_fields: list[str]) -> None:
        self._path = path
        self._text_fields = text_fields

    def invoke(self) -> Result[dict[str, FieldDuplicateStats], ProcError]:
        try:
            return Success(self._analyze_near_duplicates(self._read_records()))
        except FileNotFoundError:
            return Failure(ProcError(f"File not found: {self._path}"))
        except json.JSONDecodeError as e:
            return Failure(ProcError(f"Invalid JSONL: {e}"))

    def _read_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with open(self._path) as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    def _find_near_duplicates(
        self, texts: list[str], threshold: float = _DEFAULT_THRESHOLD,
    ) -> list[NearDuplicatePair]:
        if len(texts) < 2:
            return []
        vec = TfidfVectorizer(min_df=1).fit_transform(texts)
        sim = cosine_similarity(vec)
        np.fill_diagonal(sim, 0)
        n = len(texts)
        pairs: list[NearDuplicatePair] = []
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim[i][j])
                if score > threshold:
                    pairs.append(NearDuplicatePair(
                        index_a=i,
                        index_b=j,
                        score=round(score, 3),
                    ))
        return sorted(pairs, key=attrgetter('score'), reverse=True)

    def _analyze_near_duplicates(
        self, records: list[dict[str, Any]],
    ) -> dict[str, FieldDuplicateStats]:
        results: dict[str, FieldDuplicateStats] = {}
        for field in self._text_fields:
            texts = [str(rec.get(field, '')) for rec in records]
            texts = [t for t in texts if t.strip()]
            if len(texts) < 2:
                continue
            dupes = self._find_near_duplicates(texts)
            results[field] = FieldDuplicateStats(
                total=len(texts),
                near_duplicate_pairs=len(dupes),
                near_duplicate_rate_pct=round(len(dupes) / len(texts) * 100, 1),
                examples=dupes[:_MAX_EXAMPLES],
            )
        return results

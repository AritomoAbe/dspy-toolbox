from operator import attrgetter
from typing import Any, Sequence

import numpy as np
from returns.result import Result, Success
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.base.proc_score import ProcScore
from proc.pipeline.dataset.training_dataset import TrainingSetDataset
from proc.pipeline.training_set_auditor.contexts import NearDuplicatesContext
from proc.pipeline.training_set_auditor.models import FieldDuplicateStats, NearDuplicatePair

_DEFAULT_THRESHOLD: float = 0.85
_MAX_EXAMPLES: int = 5


class AnalyzeNearDuplicates(ProcNode):
    _SCORE: float = 1.0

    def __init__(self, dataset: TrainingSetDataset, text_fields: Sequence[str]) -> None:
        self._dataset = dataset
        self._text_fields = text_fields

    def invoke(self) -> Result[ProcScore, ProcError]:
        examples = self._dataset.load()
        records = [{k: v for k, v in ex.items()} for ex in examples]
        result = self._analyze_near_duplicates(records)
        return Success(ProcScore(value=self._score(), context=NearDuplicatesContext(result)))

    def _score(self) -> float:
        return self._SCORE

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

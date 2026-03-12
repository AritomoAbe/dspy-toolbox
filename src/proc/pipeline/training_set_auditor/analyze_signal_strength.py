import warnings
from itertools import combinations
from operator import attrgetter
from typing import Any

import numpy as np
from returns.result import Result, Success
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from proc.base.proc_error import ProcError
from proc.base.proc_node import ProcNode
from proc.pipeline.dataset.training_dataset import TrainingSetDataset
from proc.pipeline.training_set_auditor.enums import (
    LearnabilityLevel,
    ListPresence,
    SeparabilityLevel,
)
from proc.pipeline.training_set_auditor.models import (
    ClassifierError,
    ClassifierStats,
    ClassPairSeparability,
    FieldSignalResult,
    FieldSignalStats,
    KeywordScore,
    SeparabilityStats,
    SkippedField,
)


class _SignalConfig:
    strong_accuracy: float = 0.85
    moderate_accuracy: float = 0.6
    weak_accuracy: float = 0.4
    distinct_distance: float = 0.3
    overlapping_distance: float = 0.1
    min_cv_samples: int = 6
    max_tfidf_features: int = 5000
    top_keywords: int = 5
    free_text_unique_ratio: float = 0.7
    lift_epsilon: float = 1e-6
    lr_max_iter: int = 1000
    lr_regularization: float = 1.0
    cv_folds: int = 5
    cv_random_state: int = 42
    expected_key: str = "expected"
    lr_class_weight: str = "balanced"
    cv_scoring: str = "accuracy"


def _learnability_level(accuracy: float) -> LearnabilityLevel:
    if accuracy >= _SignalConfig.strong_accuracy:
        return LearnabilityLevel.strong
    if accuracy >= _SignalConfig.moderate_accuracy:
        return LearnabilityLevel.moderate
    if accuracy >= _SignalConfig.weak_accuracy:
        return LearnabilityLevel.weak
    return LearnabilityLevel.no_signal


def _separability_level(dist: float) -> SeparabilityLevel:
    if dist > _SignalConfig.distinct_distance:
        return SeparabilityLevel.distinct
    if dist > _SignalConfig.overlapping_distance:
        return SeparabilityLevel.overlapping
    return SeparabilityLevel.indistinct


def _build_text_from_fields(rec: dict[str, Any], text_fields: list[str]) -> str:
    parts: list[str] = []
    for field in text_fields:
        val = rec.get(field, "")
        if val:
            parts.append(str(val))
    return " ".join(parts)


def _value_to_label(value: Any) -> str:
    if not isinstance(value, list):
        return str(value)
    return ListPresence.non_empty if value else ListPresence.empty


def _collect_field_labels(
    records: list[dict[str, Any]],
) -> dict[str, list[str]]:
    field_labels: dict[str, list[str]] = {}
    for rec in records:
        expected = rec.get(_SignalConfig.expected_key, {})
        if not isinstance(expected, dict):
            continue
        for field, value in expected.items():
            field_labels.setdefault(field, []).append(_value_to_label(value))
    return field_labels


class AnalyzeSignalStrength(ProcNode):

    def __init__(self, dataset: TrainingSetDataset, text_fields: list[str]) -> None:
        self._dataset = dataset
        self._text_fields = text_fields

    def invoke(self) -> Result[dict[str, FieldSignalResult], ProcError]:
        examples = self._dataset.load()
        records = [{k: v for k, v in ex.items()} for ex in examples]
        return Success(self._analyze_signal_strength(records))

    def _proxy_classifier_accuracy(
        self, texts: list[str], labels: list[str],
    ) -> ClassifierError | ClassifierStats:
        unique_labels = list(set(labels))
        if len(unique_labels) < 2:
            return ClassifierError(error="only one class present — nothing to classify")
        if len(texts) < _SignalConfig.min_cv_samples:
            return ClassifierError(
                error=f"too few examples ({len(texts)}) for cross-validation",
            )
        le = LabelEncoder()
        y = le.fit_transform(labels)
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=_SignalConfig.max_tfidf_features,
            sublinear_tf=True,
        )
        X = vec.fit_transform(texts)
        n_folds = min(_SignalConfig.cv_folds, len(texts) // 2)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=_SignalConfig.cv_random_state)
        clf = LogisticRegression(
            max_iter=_SignalConfig.lr_max_iter,
            class_weight=_SignalConfig.lr_class_weight,
            C=_SignalConfig.lr_regularization,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(clf, X, y, cv=cv, scoring=_SignalConfig.cv_scoring)
        except ValueError as err:
            return ClassifierError(error=str(err))
        chance = round(1.0 / len(unique_labels), 3)
        mean_acc = round(float(scores.mean()), 3)
        std_acc = round(float(scores.std()), 3)
        denominator = max(1 - chance, _SignalConfig.lift_epsilon)
        lift = round((mean_acc - chance) / denominator, 3)
        return ClassifierStats(
            accuracy=mean_acc,
            std=std_acc,
            chance_baseline=chance,
            lift_over_chance=lift,
            n_folds=n_folds,
            n_classes=len(unique_labels),
            learnability=_learnability_level(mean_acc),
        )

    def _inter_class_separability(
        self, texts: list[str], labels: list[str],
    ) -> SeparabilityStats:
        unique_labels = sorted(set(labels))
        if len(unique_labels) < 2:
            return SeparabilityStats(pairwise=[], cohesion_per_class={}, mean_inter_class_distance=0)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        X = vec.fit_transform(texts)
        centroids: dict[str, Any] = {}
        cohesions: dict[str, float] = {}
        for label in unique_labels:
            idx: list[int] = [i for i, lbl in enumerate(labels) if lbl == label]
            class_vecs = X[idx]
            centroid = np.asarray(class_vecs.mean(axis=0))
            centroids[label] = centroid
            sims = cosine_similarity(class_vecs, centroid)
            cohesions[label] = round(float(sims.mean()), 3)
        pairs: list[ClassPairSeparability] = []
        for label_a, label_b in combinations(unique_labels, 2):
            raw_sim = cosine_similarity(centroids[label_a], centroids[label_b])
            dist = round(1.0 - float(raw_sim[0][0]), 3)
            pairs.append(ClassPairSeparability(
                pair=(label_a, label_b),
                centroid_distance=dist,
                separability=_separability_level(dist),
            ))
        sorted_pairs = sorted(pairs, key=attrgetter('centroid_distance'))
        mean_dist: float = 0
        if pairs:
            total_dist = sum(p.centroid_distance for p in pairs)
            mean_dist = round(total_dist / len(pairs), 3)
        return SeparabilityStats(
            pairwise=sorted_pairs,
            cohesion_per_class=cohesions,
            mean_inter_class_distance=mean_dist,
        )

    def _keyword_signal_analysis(
        self, texts: list[str], labels: list[str],
    ) -> dict[str, list[KeywordScore]]:
        unique_labels = sorted(set(labels))
        if len(unique_labels) < 2:
            return {}
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        X = vec.fit_transform(texts).toarray()
        feature_names = np.array(vec.get_feature_names_out())
        label_arr = np.array(labels)
        result: dict[str, list[KeywordScore]] = {}
        for label in unique_labels:
            mask_in = label_arr == label
            mask_out = ~mask_in
            mean_in = X[mask_in].mean(axis=0)
            if mask_out.any():
                mean_out = X[mask_out].mean(axis=0)
            else:
                mean_out = np.zeros_like(mean_in)
            diff = mean_in - mean_out
            top_idx = np.flip(diff.argsort())[:_SignalConfig.top_keywords]
            keywords: list[KeywordScore] = []
            for i in top_idx:
                kw_score = float(diff[i])
                if kw_score > 0:
                    kw = str(feature_names[i])
                    keywords.append(KeywordScore(keyword=kw, score=round(kw_score, 4)))
            result[label] = keywords
        return result

    def _analyze_signal_strength(
        self, records: list[dict[str, Any]],
    ) -> dict[str, FieldSignalResult]:
        expected_key = _SignalConfig.expected_key
        valid = [rec for rec in records if isinstance(rec.get(expected_key, {}), dict)]
        texts = [_build_text_from_fields(rec, self._text_fields) for rec in valid]
        field_labels = _collect_field_labels(valid)
        results: dict[str, FieldSignalResult] = {}
        for field, labels in field_labels.items():
            unique = set(labels)
            if len(unique) < 2:
                results[field] = SkippedField(skipped="only one class — nothing to separate")
                continue
            if len(unique) / len(labels) > _SignalConfig.free_text_unique_ratio:
                results[field] = SkippedField(
                    skipped=f"too many unique labels ({len(unique)}/{len(labels)}) — treat as free text",
                )
                continue
            results[field] = FieldSignalStats(
                proxy_classifier=self._proxy_classifier_accuracy(texts, labels),
                separability=self._inter_class_separability(texts, labels),
                discriminating_keywords=self._keyword_signal_analysis(texts, labels),
            )
        return results

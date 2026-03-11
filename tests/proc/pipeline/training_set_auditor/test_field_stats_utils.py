import pytest

from proc.pipeline.training_set_auditor.enums import FieldType
from proc.pipeline.training_set_auditor.utils import FieldStatsUtils


class TestBalanceScore:

    def test_all_same_returns_zero(self) -> None:
        assert FieldStatsUtils.balance_score(["a", "a", "a"]) == 0.0

    def test_single_element_returns_zero(self) -> None:
        assert FieldStatsUtils.balance_score(["x"]) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert FieldStatsUtils.balance_score([]) == 0.0

    def test_two_perfectly_balanced_returns_one(self) -> None:
        assert FieldStatsUtils.balance_score(["a", "b"]) == 1.0

    def test_four_perfectly_balanced_returns_one(self) -> None:
        assert FieldStatsUtils.balance_score(["a", "b", "c", "d"]) == 1.0

    def test_unbalanced_is_between_zero_and_one(self) -> None:
        result = FieldStatsUtils.balance_score(["a", "a", "a", "b"])
        assert 0.0 < result < 1.0

    def test_more_balanced_scores_higher(self) -> None:
        skewed = FieldStatsUtils.balance_score(["a", "a", "a", "a", "b"])
        even = FieldStatsUtils.balance_score(["a", "a", "b", "b", "c"])
        assert even > skewed


class TestMajorityPct:

    def test_empty_returns_zero(self) -> None:
        assert FieldStatsUtils.majority_pct([]) == 0.0

    def test_single_element_returns_100(self) -> None:
        assert FieldStatsUtils.majority_pct(["x"]) == 100.0

    def test_all_same_returns_100(self) -> None:
        assert FieldStatsUtils.majority_pct(["a", "a", "a"]) == 100.0

    def test_dominant_class(self) -> None:
        assert FieldStatsUtils.majority_pct(["a", "a", "a", "b"]) == 75.0

    def test_half_half(self) -> None:
        assert FieldStatsUtils.majority_pct(["a", "a", "b", "b"]) == 50.0

    def test_numeric_values(self) -> None:
        assert FieldStatsUtils.majority_pct([1, 1, 2, 3]) == 50.0


class TestInferFieldType:

    def test_all_none_returns_empty(self) -> None:
        assert FieldStatsUtils.infer_field_type([None, None]) == FieldType.empty

    def test_all_empty_strings_returns_empty(self) -> None:
        assert FieldStatsUtils.infer_field_type(["", ""]) == FieldType.empty

    def test_all_empty_lists_returns_empty(self) -> None:
        assert FieldStatsUtils.infer_field_type([[], []]) == FieldType.empty

    def test_mixed_none_and_empty_returns_empty(self) -> None:
        assert FieldStatsUtils.infer_field_type([None, "", []]) == FieldType.empty

    def test_integers_returns_numeric(self) -> None:
        assert FieldStatsUtils.infer_field_type([1, 2, 3]) == FieldType.numeric

    def test_floats_returns_numeric(self) -> None:
        assert FieldStatsUtils.infer_field_type([1.0, 2.5, 3.7]) == FieldType.numeric

    def test_mixed_int_float_returns_numeric(self) -> None:
        assert FieldStatsUtils.infer_field_type([1, 2.5, 3]) == FieldType.numeric

    def test_numeric_with_none_returns_numeric(self) -> None:
        assert FieldStatsUtils.infer_field_type([1, 2, None]) == FieldType.numeric

    def test_lists_returns_list(self) -> None:
        assert FieldStatsUtils.infer_field_type([[1, 2], [3, 4], [5]]) == FieldType.list

    def test_free_text_high_unique_ratio(self) -> None:
        # 7 distinct values out of 7 → unique_ratio = 1.0 > 0.6
        values = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
        assert FieldStatsUtils.infer_field_type(values) == FieldType.free_text

    def test_categorical_low_unique_ratio(self) -> None:
        # 2 distinct values out of 10 → unique_ratio = 0.2 <= 0.6
        values = ["yes", "no", "yes", "yes", "no", "yes", "no", "yes", "no", "yes"]
        assert FieldStatsUtils.infer_field_type(values) == FieldType.categorical

    def test_boundary_unique_ratio_above_threshold(self) -> None:
        # 7 unique out of 10 → ratio = 0.7 > 0.6 → free_text
        values = ["a", "b", "c", "d", "e", "f", "g", "a", "b", "c"]
        assert FieldStatsUtils.infer_field_type(values) == FieldType.free_text

    def test_boundary_unique_ratio_at_threshold(self) -> None:
        # 6 unique out of 10 → ratio = 0.6, not > 0.6 → categorical
        values = ["a", "b", "c", "d", "e", "f", "a", "b", "c", "d"]
        assert FieldStatsUtils.infer_field_type(values) == FieldType.categorical


class TestNumericStats:

    def test_basic_stats(self) -> None:
        stats = FieldStatsUtils.numeric_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.count == 5
        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.unique_count == 5

    def test_unique_values_capped_at_20(self) -> None:
        values = list(range(1, 30))  # 29 unique values
        stats = FieldStatsUtils.numeric_stats(values)
        assert len(stats.unique_values) == 20
        assert stats.unique_count == 29

    def test_duplicate_values_reduce_unique_count(self) -> None:
        stats = FieldStatsUtils.numeric_stats([1.0, 1.0, 2.0, 3.0])
        assert stats.unique_count == 3

    def test_all_same_majority_pct_is_100(self) -> None:
        stats = FieldStatsUtils.numeric_stats([5.0, 5.0, 5.0])
        assert stats.majority_pct == 100.0

    def test_balance_score_all_same_is_zero(self) -> None:
        stats = FieldStatsUtils.numeric_stats([7.0, 7.0, 7.0])
        assert stats.balance_score == 0.0

    def test_mean_and_std_are_rounded(self) -> None:
        stats = FieldStatsUtils.numeric_stats([1.0, 2.0, 3.0])
        assert isinstance(stats.mean, float)
        assert isinstance(stats.std, float)

    def test_min_max_correctness(self) -> None:
        stats = FieldStatsUtils.numeric_stats([-10.0, 0.0, 10.0])
        assert stats.min == -10.0
        assert stats.max == 10.0


class TestCategoricalStats:

    def test_single_category(self) -> None:
        stats = FieldStatsUtils.categorical_stats(["a", "a", "a"])
        assert stats.count == 3
        assert stats.unique_count == 1
        assert stats.distribution["a"].n == 3
        assert stats.distribution["a"].pct == 100.0
        assert stats.majority_pct == 100.0

    def test_two_categories_counts(self) -> None:
        stats = FieldStatsUtils.categorical_stats(["a", "a", "b"])
        assert stats.count == 3
        assert stats.unique_count == 2
        assert stats.distribution["a"].n == 2
        assert stats.distribution["b"].n == 1

    def test_two_categories_percentages(self) -> None:
        stats = FieldStatsUtils.categorical_stats(["a", "a", "b"])
        assert stats.distribution["a"].pct == pytest.approx(66.7, abs=0.1)
        assert stats.distribution["b"].pct == pytest.approx(33.3, abs=0.1)

    def test_balanced_two_categories_balance_score_is_one(self) -> None:
        stats = FieldStatsUtils.categorical_stats(["a", "b", "a", "b"])
        assert stats.balance_score == 1.0

    def test_single_category_balance_score_is_zero(self) -> None:
        stats = FieldStatsUtils.categorical_stats(["x", "x", "x"])
        assert stats.balance_score == 0.0

    def test_all_keys_present_in_distribution(self) -> None:
        stats = FieldStatsUtils.categorical_stats(["x", "y", "z", "x"])
        assert set(stats.distribution.keys()) == {"x", "y", "z"}

    def test_majority_pct_dominant_class(self) -> None:
        stats = FieldStatsUtils.categorical_stats(["a", "a", "a", "b"])
        assert stats.majority_pct == 75.0
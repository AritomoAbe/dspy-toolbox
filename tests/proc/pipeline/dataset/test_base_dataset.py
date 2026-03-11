from typing import Any

import pytest

from proc.pipeline.dataset.base_dataset import BaseDataset


class _ConcreteDataset(BaseDataset):
    def load(self) -> list[Any]:
        return ["item1", "item2"]


class TestBaseDataset:

    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseDataset()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self) -> None:
        ds = _ConcreteDataset()
        assert ds is not None

    def test_load_returns_list(self) -> None:
        ds = _ConcreteDataset()
        assert isinstance(ds.load(), list)

    def test_load_returns_expected_items(self) -> None:
        ds = _ConcreteDataset()
        assert ds.load() == ["item1", "item2"]

    def test_subclass_without_load_raises(self) -> None:
        class _IncompleteDataset(BaseDataset):
            pass

        with pytest.raises(TypeError):
            _IncompleteDataset()  # type: ignore[abstract]

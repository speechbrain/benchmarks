from abc import ABC, abstractmethod
from itertools import chain, combinations, groupby
from operator import itemgetter
from typing import Generic, Hashable, Sequence, TypedDict, TypeVar

from torch.utils.data import Dataset

from speechbrain.dataio.dataset import (
    DynamicItemDataset,
    FilteredSortedDynamicItemDataset,
)

TargetT = TypeVar("TargetT", bound=Hashable)
DatasetT = TypeVar("DatasetT")


class DatasetSplit(TypedDict, Generic[DatasetT]):
    train: Dataset[DatasetT]
    test: Dataset[DatasetT]


class DatasetSplitter(Generic[TargetT, DatasetT], ABC):
    """Abstract class which defines how to split a dataset for cross-validation."""

    def __init__(self, dataset: Dataset[DatasetT]):
        self.dataset = dataset

    @property
    @abstractmethod
    def targets(self) -> Sequence[TargetT]:
        ...

    def __len__(self) -> int:
        return len(self.targets)

    @abstractmethod
    def __getitem__(self, target: TargetT) -> DatasetSplit[DatasetT]:
        ...

    def __iter__(self):
        for target in self.targets:
            yield self[target]


class MetadataSplitter(DatasetSplitter[TargetT, DatasetT]):
    """Splits a dataset"""

    dataset: DynamicItemDataset

    def __init__(self, dataset: DynamicItemDataset, key: str) -> None:
        if not isinstance(dataset, DynamicItemDataset):
            raise ValueError(
                f"{self.__class__.__name__} requires dataset to be instance of `speechbrain.dataio.dataset.DynamicItemDataset`"
            )

        super().__init__(dataset.filtered_sorted(sort_key=key))
        self.unique_ids = set(self.dataset.data_ids)

        self._split_folds(key)

    def _split_folds(self, key):
        self.folds = {}
        with self.dataset.output_keys_as([key, "id"]):
            for target, group in groupby(self.dataset, itemgetter(key)):  # type: ignore
                self.folds[target] = tuple(map(itemgetter("id"), group))

    @property
    def targets(self) -> Sequence[TargetT]:
        return tuple(self.folds)

    def __getitem__(self, target: TargetT) -> DatasetSplit[DatasetT]:
        test_data_ids = self._get_test_data_ids(target)
        train_data_ids = self.unique_ids - set(test_data_ids)

        return DatasetSplit(
            train=FilteredSortedDynamicItemDataset(
                self.dataset, train_data_ids
            ),
            test=FilteredSortedDynamicItemDataset(self.dataset, test_data_ids),
        )

    def _get_test_data_ids(self, target):
        test_data_ids = self.folds[target]
        return test_data_ids


class LeaveKOutSplitter(MetadataSplitter[TargetT, DatasetT]):
    def __init__(self, dataset: DynamicItemDataset, key: str, leave_k_out=1):
        super().__init__(dataset, key)
        self.leave_k_out = leave_k_out

    @property
    def targets(self) -> Sequence[tuple[TargetT]]:
        return tuple(combinations(super().targets, self.leave_k_out))

    def _get_test_data_ids(self, target):
        test_data_ids = tuple(
            chain.from_iterable(self.folds[t] for t in target)
        )

        return test_data_ids


class CrossSubjectSplitter(LeaveKOutSplitter[str, DatasetT]):
    def __init__(self, dataset: DynamicItemDataset, leave_k_out=1):
        super().__init__(dataset, "subject", leave_k_out)


class CrossSessionSplitter(LeaveKOutSplitter[str, DatasetT]):
    def __init__(self, dataset: DynamicItemDataset, leave_k_out=1):
        super().__init__(dataset, "session", leave_k_out)


class CrossDatasetSplitter(LeaveKOutSplitter[str, DatasetT]):
    def __init__(self, dataset: DynamicItemDataset, leave_k_out=1):
        super().__init__(dataset, "dataset", leave_k_out)

from abc import ABC, abstractmethod
from itertools import groupby
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
    """Abstract class which defines"""

    def __init__(self, dataset: Dataset[DatasetT]):
        self.dataset = dataset

    @property
    @abstractmethod
    def targets(self) -> Sequence[TargetT]: ...

    def __len__(self) -> int:
        return len(self.targets)

    @abstractmethod
    def __getitem__(self, target: TargetT) -> DatasetSplit[DatasetT]: ...

    def __iter__(self):
        for target in self.targets:
            yield self[target]


class MetadataSplitter(DatasetSplitter[str, DatasetT]):
    dataset: DynamicItemDataset

    def __init__(self, dataset: DynamicItemDataset, key: str):
        if not isinstance(dataset, DynamicItemDataset):
            raise ValueError(
                f"{self.__class__.__name__} requires dataset to be instance of `speechbrain.dataio.dataset.DynamicItemDataset`"
            )

        super().__init__(dataset.filtered_sorted(sort_key=key))
        self.folds = {}
        with self.dataset.output_keys_as([key, "id"]):
            for target, group in groupby(self.dataset, itemgetter(key)):  # type: ignore
                self.folds[target] = tuple(map(itemgetter("id"), group))

    @property
    def targets(self) -> Sequence[str]:
        return tuple(self.folds)

    def __getitem__(self, target: str) -> DatasetSplit[DatasetT]:
        test_data_ids = self.folds[target]
        train_data_ids = set(self.dataset.data) - set(test_data_ids)

        return DatasetSplit(
            train=FilteredSortedDynamicItemDataset(
                self.dataset, train_data_ids
            ),
            test=FilteredSortedDynamicItemDataset(self.dataset, test_data_ids),
        )

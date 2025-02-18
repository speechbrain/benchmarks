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
    """
    Typed dictionary representing a split of a dataset.

    Attributes
    ----------
    train : Dataset[DatasetT]
        The training subset of the dataset.
    test : Dataset[DatasetT]
        The testing subset of the dataset.
    """


class DatasetSplitter(Generic[TargetT, DatasetT], ABC):
    """
    Abstract base class defining how to split a dataset for cross-validation.

    This class provides the basic interface for splitting a dataset into training
    and testing portions based on targets.

    Parameters
    ----------
    dataset : Dataset[DatasetT]
        The dataset to be split.
    """

    def __init__(self, dataset: Dataset[DatasetT]):
        self.dataset = dataset

    @property
    @abstractmethod
    def targets(self) -> Sequence[TargetT]:
        """
        Sequence[TargetT]: A sequence of targets on which the dataset will be split.
        """

    def __len__(self) -> int:
        """
        Return the number of targets.

        Returns
        -------
        int
            The length of the targets sequence.
        """
        return len(self.targets)

    @abstractmethod
    def __getitem__(self, target: TargetT) -> DatasetSplit[DatasetT]:
        """
        Retrieve the dataset split for a given target.

        Parameters
        ----------
        target : TargetT
            The target value for which the dataset should be split.

        Returns
        -------
        DatasetSplit[DatasetT]
            A dictionary-like object containing 'train' and 'test' splits.
        """

    def __iter__(self):
        """
        Iterate over all dataset splits.

        Yields
        ------
        DatasetSplit[DatasetT]
            Each split corresponding to a target in the dataset.
        """
        for target in self.targets:
            yield self[target]


class MetadataSplitter(DatasetSplitter[TargetT, DatasetT]):
    """
    Splits a dataset based on a metadata key.

    The dataset is first sorted and filtered based on the provided key.
    Then, the entries are grouped by the key, forming folds for cross-validation.

    Attributes
    ----------
    dataset : DynamicItemDataset
        The filtered and sorted dataset.
    folds : dict
        A mapping from each unique target value to the corresponding tuple of data identifiers.
    unique_ids : set
        A set of unique data identifiers present in the dataset.
    """

    dataset: DynamicItemDataset

    def __init__(self, dataset: DynamicItemDataset, key: str) -> None:
        """
        Initialize the MetadataSplitter.

        Parameters
        ----------
        dataset : DynamicItemDataset
            The dataset to be split. Must be an instance of `DynamicItemDataset`.
        key : str
            The metadata key used for sorting and splitting the dataset.

        Raises
        ------
        ValueError
            If the provided dataset is not an instance of `DynamicItemDataset`.
        """
        if not isinstance(dataset, DynamicItemDataset):
            raise ValueError(
                f"{self.__class__.__name__} requires dataset to be instance of `speechbrain.dataio.dataset.DynamicItemDataset`"
            )

        super().__init__(dataset.filtered_sorted(sort_key=key))
        self.unique_ids = set(self.dataset.data_ids)
        self._split_folds(key)

    def _split_folds(self, key):
        """
        Create folds for the dataset by grouping entries based on the metadata key.

        Parameters
        ----------
        key : str
            The metadata key to group the dataset entries.
        """
        self.folds = {}
        with self.dataset.output_keys_as([key, "id"]):
            for target, group in groupby(self.dataset, itemgetter(key)):  # type: ignore
                self.folds[target] = tuple(map(itemgetter("id"), group))

    @property
    def targets(self) -> Sequence[TargetT]:
        """
        Get the targets based on the metadata key.

        Returns
        -------
        Sequence[TargetT]
            A tuple of unique target values representing each fold.
        """
        return tuple(self.folds)

    def __getitem__(self, target: TargetT) -> DatasetSplit[DatasetT]:
        """
        Retrieve the training and testing splits for a given target.

        Parameters
        ----------
        target : TargetT
            The target value for which to create the dataset split.

        Returns
        -------
        DatasetSplit[DatasetT]
            A dictionary with 'train' and 'test' keys containing the respective dataset splits.
        """
        test_data_ids = self._get_test_data_ids(target)
        train_data_ids = self.unique_ids - set(test_data_ids)

        return DatasetSplit(
            train=FilteredSortedDynamicItemDataset(
                self.dataset, train_data_ids
            ),
            test=FilteredSortedDynamicItemDataset(self.dataset, test_data_ids),
        )

    def _get_test_data_ids(self, target):
        """
        Retrieve the test data identifiers for a given target.

        Parameters
        ----------
        target : TargetT
            The target value for which the test identifiers are required.

        Returns
        -------
        tuple
            A tuple of data identifiers corresponding to the test set for the target.
        """
        test_data_ids = self.folds[target]
        return test_data_ids


class LeaveKOutSplitter(MetadataSplitter[TargetT, DatasetT]):
    """
    Splits a dataset using a leave-k-out strategy.

    This splitter creates combinations of targets by leaving out 'k' targets as the test set
    and using the remaining targets as the training set.
    """

    def __init__(self, dataset: DynamicItemDataset, key: str, leave_k_out=1):
        """
        Initialize the LeaveKOutSplitter.

        Parameters
        ----------
        dataset : DynamicItemDataset
            The dataset to be split.
        key : str
            The metadata key used for splitting.
        leave_k_out : int, optional
            The number of targets to leave out for the test split (default is 1).
        """
        super().__init__(dataset, key)
        self.leave_k_out = leave_k_out

    @property
    def targets(self) -> Sequence[tuple[TargetT]]:
        """
        Get combinations of targets for leave-k-out splitting.

        Returns
        -------
        Sequence[tuple[TargetT]]
            A tuple of combinations, where each combination represents a set of targets to be left out.
        """
        return tuple(combinations(super().targets, self.leave_k_out))

    def _get_test_data_ids(self, target):
        """
        Retrieve test data identifiers for a given combination of targets.

        Parameters
        ----------
        target : tuple[TargetT]
            A tuple of target values for which the test set is created.

        Returns
        -------
        tuple
            A tuple containing aggregated data identifiers from all specified targets.
        """
        test_data_ids = tuple(
            chain.from_iterable(self.folds[t] for t in target)
        )
        return test_data_ids


class CrossSubjectSplitter(LeaveKOutSplitter[str, DatasetT]):
    """
    Splits the dataset for cross-subject validation.

    This splitter uses the 'subject' metadata key to perform a leave-k-out split
    based on subject identifiers.
    """

    def __init__(self, dataset: DynamicItemDataset, leave_k_out=1):
        """
        Initialize the CrossSubjectSplitter.

        Parameters
        ----------
        dataset : DynamicItemDataset
            The dataset to be split.
        leave_k_out : int, optional
            The number of subjects to leave out for the test split (default is 1).
        """
        super().__init__(dataset, "subject", leave_k_out)


class CrossSessionSplitter(LeaveKOutSplitter[str, DatasetT]):
    """
    Splits the dataset for cross-session validation.

    This splitter uses the 'session' metadata key to perform a leave-k-out split
    based on session identifiers.
    """

    def __init__(self, dataset: DynamicItemDataset, leave_k_out=1):
        """
        Initialize the CrossSessionSplitter.

        Parameters
        ----------
        dataset : DynamicItemDataset
            The dataset to be split.
        leave_k_out : int, optional
            The number of sessions to leave out for the test split (default is 1).
        """
        super().__init__(dataset, "session", leave_k_out)


class CrossDatasetSplitter(LeaveKOutSplitter[str, DatasetT]):
    """
    Splits the dataset for cross-dataset validation.

    This splitter uses the 'dataset' metadata key to perform a leave-k-out split,
    which is useful when combining multiple datasets.
    """

    def __init__(self, dataset: DynamicItemDataset, leave_k_out=1):
        """
        Initialize the CrossDatasetSplitter.

        Parameters
        ----------
        dataset : DynamicItemDataset
            The dataset to be split.
        leave_k_out : int, optional
            The number of datasets to leave out for the test split (default is 1).
        """
        super().__init__(dataset, "dataset", leave_k_out)

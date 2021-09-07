from __future__ import annotations

from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from functools import partial
from typing import Sequence, Union, Any, List, Iterable, Optional, TypeVar, Dict
from imagenet.definitions_and_structures import DatasetThatSupportsTargets, INCProtocolIterator, INCProtocol, DatasetPart

T_idx = TypeVar('T_idx')


def tensor_as_list(sequence):
    if isinstance(sequence, Tensor):
        return sequence.tolist()
    # Numpy already returns the correct format
    # Example list(np.array([1, 2, 3])) returns [1, 2, 3]
    # whereas list(torch.tensor([1, 2, 3])) returns [tensor(1), tensor(2), tensor(3)], which is "bad"
    return list(sequence)


def tensor_as_set(sequence):
    if isinstance(sequence, Tensor):
        return set(sequence.tolist())
    # Numpy already returns the correct format
    # Example list(np.array([1, 2, 3])) returns [1, 2, 3]
    # whereas list(torch.tensor([1, 2, 3])) returns [tensor(1), tensor(2), tensor(3)], which is "bad"
    return set(sequence)


def __get_indexes_with_patterns_ordered_by_classes(sequence: Sequence[T_idx], search_elements: Sequence[T_idx],
                                                   sort_indexes: bool = True, sort_classes: bool = True,
                                                   class_mapping: Optional[Tensor] = None) -> Tensor:
    # list() handles the situation in which search_elements is a torch.Tensor
    # without it: result_per_class[element].append(idx) -> error
    # as result_per_class[0] won't exist while result_per_class[tensor(0)] will

    result_per_class: Dict[T_idx, List[int]] = OrderedDict()
    result: List[int] = []

    search_elements = tensor_as_list(search_elements)
    sequence = tensor_as_list(sequence)

    if class_mapping is not None:
        class_mapping = tensor_as_list(class_mapping)
    else:
        class_mapping = list(range(max(search_elements) + 1))

    if sort_classes:
        search_elements = sorted(search_elements)

    for search_element in search_elements:
        result_per_class[search_element] = []

    set_search_elements = set(search_elements)

    for idx, element in enumerate(sequence):
        if class_mapping[element] in set_search_elements:
            result_per_class[class_mapping[element]].append(idx)

    for search_element in search_elements:
        if sort_indexes:
            result_per_class[search_element].sort()
        result.extend(result_per_class[search_element])

    return torch.tensor(result, dtype=torch.int)


def __get_indexes_without_class_bucketing(sequence: Sequence[T_idx], search_elements: Sequence[T_idx],
                                          sort_indexes: bool = False, class_mapping: Optional[Tensor] = None) -> Tensor:
    sequence = tensor_as_list(sequence)
    result: List[T_idx] = []

    if class_mapping is not None:
        class_mapping = tensor_as_list(class_mapping)
    else:
        class_mapping = list(range(max(search_elements) + 1))

    search_elements = tensor_as_set(search_elements)

    for idx, element in enumerate(sequence):
        if class_mapping[element] in search_elements:
            result.append(idx)

    if sort_indexes:
        result.sort()
    return torch.tensor(result, dtype=torch.int)


def get_indexes_from_set(sequence: Sequence[T_idx], search_elements: Sequence[T_idx], bucket_classes: bool = True,
                         sort_classes: bool = False, sort_indexes: bool = False,
                         class_mapping: Optional[Tensor] = None) -> Tensor:
    if bucket_classes:
        return __get_indexes_with_patterns_ordered_by_classes(sequence, search_elements, sort_indexes=sort_indexes,
                                                              sort_classes=sort_classes, class_mapping=class_mapping)
    else:
        return __get_indexes_without_class_bucketing(sequence, search_elements, sort_indexes=sort_indexes,
                                                     class_mapping=class_mapping)


class ListsDataset(Dataset):
    """
    A Dataset that applies transformations before returning patterns/targets
    Also, this Dataset supports slicing
    """
    def __init__(self, patterns, targets, transform=None, target_transform=None):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.patterns = patterns
        self.targets = targets

    def __getitem__(self, idx):
        patterns: List[Any] = []
        labels: List[Tensor] = []
        indexes_iterator: Iterable[int]

        treat_as_tensors: bool = True

        # Makes dataset sliceable
        if isinstance(idx, slice):
            indexes_iterator = range(*idx.indices(len(self.patterns)))
        elif isinstance(idx, int):
            indexes_iterator = [idx]
        else:  # Should handle other types (ndarray, Tensor, Sequence, ...)
            if hasattr(idx, 'shape') and len(getattr(idx, 'shape')) == 0:  # Manages 0-d ndarray / Tensor
                indexes_iterator = [int(idx)]
            else:
                indexes_iterator = idx

        for single_idx in indexes_iterator:
            pattern, label = self.__get_single_item(single_idx)
            if not isinstance(pattern, Tensor):
                treat_as_tensors = False

            #pattern = pattern.unsqueeze(0)

            label = torch.as_tensor(label)
            #label = label.unsqueeze(0)

            patterns.append(pattern)
            labels.append(label)

        if len(patterns) == 1:
            if treat_as_tensors:
                patterns[0] = patterns[0].squeeze(0)
            labels[0] = labels[0].squeeze(0)

            return patterns[0], labels[0]
        else:
            labels_cat = torch.cat([single_label.unsqueeze(0) for single_label in labels])

            if treat_as_tensors:
                patterns_cat = torch.cat([single_pattern.unsqueeze(0) for single_pattern in patterns])
                return patterns_cat, labels_cat
            else:
                return patterns, labels_cat

    def __len__(self):
        return len(self.patterns)

    def __get_single_item(self, idx: int):
        pattern, label = self.patterns[idx], self.targets[idx]
        if self.transform is not None:
            pattern = self.transform(pattern)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return pattern, label

class TransformationDataset(Dataset, DatasetThatSupportsTargets):
    """
    A Dataset that applies transformations before returning patterns/targets
    Also, this Dataset supports slicing
    """
    def __init__(self, dataset: DatasetThatSupportsTargets, transform=None, target_transform=None):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        self.preloaded_data = None
        self.targets = dataset.targets

    def __getitem__(self, idx):
        patterns: List[Any] = []
        labels: List[Tensor] = []
        indexes_iterator: Iterable[int]

        treat_as_tensors: bool = True

        # Makes dataset sliceable
        if isinstance(idx, slice):
            indexes_iterator = range(*idx.indices(len(self.dataset)))
        elif isinstance(idx, int):
            indexes_iterator = [idx]
        else:  # Should handle other types (ndarray, Tensor, Sequence, ...)
            if hasattr(idx, 'shape') and len(getattr(idx, 'shape')) == 0:  # Manages 0-d ndarray / Tensor
                indexes_iterator = [int(idx)]
            else:
                indexes_iterator = idx

        for single_idx in indexes_iterator:
            pattern, label = self.__get_single_item(single_idx)
            if not isinstance(pattern, Tensor):
                treat_as_tensors = False

            #pattern = pattern.unsqueeze(0)

            label = torch.as_tensor(label)
            #label = label.unsqueeze(0)

            patterns.append(pattern)
            labels.append(label)

        if len(patterns) == 1:
            if treat_as_tensors:
                patterns[0] = patterns[0].squeeze(0)
            labels[0] = labels[0].squeeze(0)

            return patterns[0], labels[0]
        else:
            labels_cat = torch.cat([single_label.unsqueeze(0) for single_label in labels])

            if treat_as_tensors:
                patterns_cat = torch.cat([single_pattern.unsqueeze(0) for single_pattern in patterns])
                return patterns_cat, labels_cat
            else:
                return patterns, labels_cat

    def __len__(self):
        return len(self.dataset)

    def preload_data(self, num_workers=4, batch_size=50):
        if self.preloaded_data is not None:
            return self

        self.preloaded_data = []

        patterns_loader = DataLoader(self.dataset, num_workers=num_workers,
                                     shuffle=False, drop_last=False, batch_size=batch_size)

        for patterns in patterns_loader:  # patterns is a tuple patterns_x, patterns_y, ... = patterns
            for pattern_idx in range(len(patterns[0])):  # patterns[0] is patterns_x, this means that len(patterns[0]) == patterns_x.shape[0]
                pattern_data = tuple()
                for pattern_element_idx in range(len(patterns)):
                    pattern_data += (patterns[pattern_element_idx][pattern_idx],)
                self.preloaded_data.append(pattern_data)
        return self

    def __get_single_item(self, idx: int):
        if self.preloaded_data is not None:
            return self.preloaded_data[idx]
        # print(type(self.dataset))
        pattern, label = self.dataset[idx]
        # print(type(pattern))
        # print(self.transform)
        if self.transform is not None:
            # print("asddddd")
            # print(self.transform)
            # print(pattern)
            pattern = self.transform(pattern)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return pattern, label


class LazyClassMapping(Sequence[int]):
    """
    Defines a lazy targets class_list_per_batch.

    This class is used when in need of lazy populating a targets field whose
    elements need to be filtered out (when subsetting, see
    :class:`torch.utils.data.Subset`) and/or transformed (based on some
    class_list_per_batch). This will allow for a more efficient memory usage as
    the class_list_per_batch is done on the fly instead of actually allocating a
    new list.
    """
    def __init__(self, targets: Sequence[int],
                 indices: Union[Sequence[int], None],
                 mapping: Optional[Sequence[int]] = None):
        self._targets = targets
        self._mapping = mapping
        self._indices = indices

    def __len__(self):
        if self._indices is None:
            return len(self._targets)
        return len(self._indices)

    def __getitem__(self, item_idx) -> int:
        if self._indices is not None:
            subset_idx = self._indices[item_idx]
        else:
            subset_idx = item_idx

        if self._mapping is not None:
            return self._mapping[self._targets[subset_idx]]

        return self._targets[subset_idx]

    def __str__(self):
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'

class TransformationSubset(Dataset):
    def __init__(self, dataset: DatasetThatSupportsTargets, indices: Sequence[int],
                 transform=None, target_transform=None, class_mapping: Optional[Tensor] = None):
        super().__init__()
        self.dataset = TransformationDataset(dataset, transform=transform, target_transform=target_transform)
        self.indices = indices
        self.class_mapping = class_mapping
        self.targets = LazyClassMapping(dataset.targets, indices,
                                        mapping=class_mapping)

    def __getitem__(self, idx) -> (Tensor, Tensor):
        result = self.dataset[self.indices[idx]]
        if self.class_mapping is not None:
            return result[0], self.class_mapping[result[1]]

        return result

    def __len__(self) -> int:
        return len(self.indices)


def make_transformation_subset(dataset: DatasetThatSupportsTargets, transform: Any, target_transform: Any,
                               class_mapping: Tensor, classes: Sequence[int],
                               bucket_classes=False, sort_classes=False, sort_indexes=False):
    return TransformationSubset(dataset, get_indexes_from_set(dataset.targets, classes,
                                                              bucket_classes=bucket_classes,
                                                              sort_classes=sort_classes,
                                                              sort_indexes=sort_indexes,
                                                              class_mapping=class_mapping),
                                transform=transform, target_transform=target_transform, class_mapping=class_mapping)


class NCProtocol:
    def __init__(self, train_dataset: DatasetThatSupportsTargets, test_dataset: DatasetThatSupportsTargets,
                 n_tasks: int, shuffle: bool = True, seed: Optional[int] = None,
                 train_transform=None, train_target_transform=None, test_transform=None, test_target_transform=None,
                 steal_transforms_from_datasets: bool = True,
                 fixed_class_order: Optional[Sequence[int]] = None, per_task_classes: Optional[Dict[int, int]] = None,
                 remap_class_indexes_in_ascending_order: bool = False):
        self.train_dataset: DatasetThatSupportsTargets = train_dataset
        self.test_dataset: DatasetThatSupportsTargets = test_dataset
        self.validation_dataset: DatasetThatSupportsTargets = train_dataset
        self.n_tasks: int = n_tasks
        self.classes_order: Tensor = torch.unique(torch.tensor(train_dataset.targets))
        self.train_transform = train_transform
        self.train_target_transform = train_target_transform
        self.test_transform = test_transform
        self.test_target_transform = test_target_transform
        self.remap_class_indexes_in_ascending_order = remap_class_indexes_in_ascending_order
        self.n_classes = len(self.classes_order)

        if n_tasks < 1:
            raise ValueError('Invalid task number (n_tasks parameter): must be greater than 0')

        if per_task_classes is not None:
            if max(per_task_classes.keys()) >= n_tasks:
                raise ValueError('Invalid task id in per_task_classes parameter: '
                                 + str(max(per_task_classes.keys())) + ': task ids must be in range [0, n_tasks)')

            if sum(per_task_classes.values()) > self.n_classes:
                raise ValueError('Insufficient number of classes: classes mapping defined in '
                                 'per_task_classes parameter can\'t be satisfied')

            if (self.n_classes - sum(per_task_classes.values())) % (n_tasks - len(per_task_classes)) > 0:
                raise ValueError('Invalid number of tasks: classes contained in dataset cannot be divided by n_tasks')

            default_per_task_classes = (self.n_classes - sum(per_task_classes.values())) // \
                                       (n_tasks - len(per_task_classes))

            self.classes_per_task: List[int] = [default_per_task_classes] * n_tasks
            for task_id in per_task_classes:
                self.classes_per_task[task_id] = per_task_classes[task_id]
        else:
            if self.n_classes % n_tasks > 0:
                raise ValueError('Invalid number of tasks: classes contained in dataset cannot be divided by n_tasks')
            self.classes_per_task: List[int] = [self.n_classes // n_tasks] * n_tasks

        if fixed_class_order is not None:
            self.classes_order = torch.tensor(fixed_class_order)
        elif shuffle:
            if seed is not None:
                torch.random.manual_seed(seed)
            self.classes_order = self.classes_order[torch.randperm(len(self.classes_order))]

        if steal_transforms_from_datasets:
            if hasattr(train_dataset, 'transform'):
                self.train_transform = train_dataset.transform
                train_dataset.transform = None
            if hasattr(train_dataset, 'target_transform'):
                self.train_target_transform = train_dataset.target_transform
                train_dataset.target_transform = None

            if hasattr(test_dataset, 'transform'):
                self.test_transform = test_dataset.transform
                test_dataset.transform = None
            if hasattr(test_dataset, 'target_transform'):
                self.test_target_transform = test_dataset.target_transform
                test_dataset.target_transform = None

    def __iter__(self) -> INCProtocolIterator:
        return NCProtocolIterator(self)

    def get_task_classes(self, task_id: int) -> Tensor:
        classes_start_idx = sum(self.classes_per_task[:task_id])
        classes_end_idx = classes_start_idx + self.classes_per_task[task_id]

        if not self.remap_class_indexes_in_ascending_order:
            return self.classes_order[classes_start_idx:classes_end_idx]
        else:
            return torch.arange(classes_start_idx, classes_end_idx, dtype=torch.long)

    def get_task_classes_mapping(self) -> Tensor:
        if not self.remap_class_indexes_in_ascending_order:
            return torch.tensor(list(range(self.n_classes)))

        classes_order_as_list = self.classes_order.tolist()  # no index() method in Tensor :'(
        return torch.tensor([classes_order_as_list.index(class_idx) for class_idx in range(self.n_classes)])


class NCProtocolIterator:
    def __init__(self, protocol: INCProtocol,
                 swap_train_test_transformations: bool = False,
                 are_transformations_disabled: bool = False,
                 initial_current_task: int = -1):
        self.current_task: int = -1
        self.protocol: INCProtocol = protocol
        self.are_train_test_transformations_swapped = swap_train_test_transformations
        self.are_transformations_disabled = are_transformations_disabled

        self.classes_seen_so_far: Tensor = torch.empty(0, dtype=torch.long)
        self.classes_in_this_task: Tensor = torch.empty(0, dtype=torch.long)
        self.prev_classes: Tensor = torch.empty(0, dtype=torch.long)
        self.future_classes: Tensor = torch.empty(0, dtype=torch.long)

        classes_mapping = self.protocol.get_task_classes_mapping()
        if self.are_transformations_disabled:
            self.train_subset_factory = partial(make_transformation_subset, self.protocol.train_dataset,
                                                None, None, classes_mapping)
            self.test_subset_factory = partial(make_transformation_subset, self.protocol.test_dataset,
                                               None, None, classes_mapping)
        else:
            if self.are_train_test_transformations_swapped:
                self.train_subset_factory = partial(make_transformation_subset, self.protocol.train_dataset,
                                                    self.protocol.test_transform, self.protocol.test_target_transform,
                                                    classes_mapping)
                self.test_subset_factory = partial(make_transformation_subset, self.protocol.test_dataset,
                                                   self.protocol.train_transform, self.protocol.train_target_transform,
                                                   classes_mapping)
            else:
                self.train_subset_factory = partial(make_transformation_subset, self.protocol.train_dataset,
                                                    self.protocol.train_transform, self.protocol.train_target_transform,
                                                    classes_mapping)
                self.test_subset_factory = partial(make_transformation_subset, self.protocol.test_dataset,
                                                   self.protocol.test_transform, self.protocol.test_target_transform,
                                                   classes_mapping)

        for _ in range(initial_current_task+1):
            self.__go_to_next_task()

    def __next__(self) -> (DatasetThatSupportsTargets, INCProtocolIterator):
        self.__go_to_next_task()
        return self.get_current_training_set(), self

    # Training set utils
    def get_current_training_set(self, bucket_classes=False, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        return self.train_subset_factory(self.classes_in_this_task, bucket_classes=bucket_classes,
                                         sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_task_training_set(self, task_id: int, bucket_classes=False, sort_classes=False,
                              sort_indexes=False) -> DatasetThatSupportsTargets:
        classes_in_required_task = self.protocol.get_task_classes(task_id)
        return self.train_subset_factory(classes_in_required_task, bucket_classes=bucket_classes,
                                         sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_cumulative_training_set(self, include_current_task: bool = True, bucket_classes=False, sort_classes=False,
                                    sort_indexes=False) -> DatasetThatSupportsTargets:
        if include_current_task:
            return self.train_subset_factory(self.classes_seen_so_far, bucket_classes=bucket_classes,
                                             sort_classes=sort_classes, sort_indexes=sort_indexes)
        else:
            return self.train_subset_factory(self.prev_classes, bucket_classes=bucket_classes,
                                             sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_complete_training_set(self, bucket_classes=False, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        return self.train_subset_factory(self.protocol.classes_order, bucket_classes=bucket_classes,
                                         sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_future_training_set(self, bucket_classes=False, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        return self.train_subset_factory(self.future_classes, bucket_classes=bucket_classes,
                                         sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_training_set_part(self, dataset_part: DatasetPart, bucket_classes=False, sort_classes=False,
                              sort_indexes=False) -> DatasetThatSupportsTargets:
        if dataset_part == DatasetPart.CURRENT_TASK:
            return self.get_current_training_set(bucket_classes=bucket_classes, sort_classes=sort_classes,
                                                 sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.CUMULATIVE:
            return self.get_cumulative_training_set(include_current_task=True, bucket_classes=bucket_classes,
                                                    sort_classes=sort_classes, sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.OLD:
            return self.get_cumulative_training_set(include_current_task=False, bucket_classes=bucket_classes,
                                                    sort_classes=sort_classes, sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.FUTURE:
            return self.get_future_training_set(bucket_classes=bucket_classes, sort_classes=sort_classes,
                                                sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.COMPLETE_SET:
            return self.get_complete_training_set(bucket_classes=bucket_classes, sort_classes=sort_classes,
                                                  sort_indexes=sort_indexes)
        else:
            raise ValueError('Unsupported dataset part')

    # Test set utils
    def get_current_test_set(self, bucket_classes=False, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        return self.test_subset_factory(self.classes_in_this_task, bucket_classes=bucket_classes,
                                        sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_cumulative_test_set(self, include_current_task: bool = True, bucket_classes=False, sort_classes=False,
                                sort_indexes=False) -> DatasetThatSupportsTargets:
        if include_current_task:
            return self.test_subset_factory(self.classes_seen_so_far, bucket_classes=bucket_classes,
                                            sort_classes=sort_classes, sort_indexes=sort_indexes)
        else:
            return self.test_subset_factory(self.prev_classes, bucket_classes=bucket_classes,
                                            sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_task_test_set(self, task_id: int, bucket_classes=False, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        classes_in_required_task = self.protocol.get_task_classes(task_id)
        return self.test_subset_factory(classes_in_required_task, bucket_classes=bucket_classes,
                                        sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_complete_test_set(self, bucket_classes=False, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        return self.test_subset_factory(self.protocol.classes_order, bucket_classes=bucket_classes,
                                        sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_future_test_set(self, bucket_classes=False, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        return self.test_subset_factory(self.future_classes, bucket_classes=bucket_classes,
                                        sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_test_set_part(self, dataset_part: DatasetPart, bucket_classes=False, sort_classes=False,
                          sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        if dataset_part == DatasetPart.CURRENT_TASK:
            return self.get_current_test_set(bucket_classes=bucket_classes, sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.CUMULATIVE:
            return self.get_cumulative_test_set(include_current_task=True, bucket_classes=bucket_classes,
                                                sort_classes=sort_classes, sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.OLD:
            return self.get_cumulative_test_set(include_current_task=False, bucket_classes=bucket_classes,
                                                sort_classes=sort_classes, sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.FUTURE:
            return self.get_future_test_set(bucket_classes=bucket_classes, sort_classes=sort_classes,
                                            sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.COMPLETE_SET:
            return self.get_complete_test_set(bucket_classes=bucket_classes, sort_classes=sort_classes,
                                              sort_indexes=sort_indexes)
        else:
            raise ValueError('Unsupported dataset part')

    # Transformation utility function. Useful if you want to test on the training set (using test transformations)
    def swap_transformations(self) -> INCProtocolIterator:
        return NCProtocolIterator(self.protocol,
                                  swap_train_test_transformations=not self.are_train_test_transformations_swapped,
                                  are_transformations_disabled=self.are_transformations_disabled,
                                  initial_current_task=self.current_task)

    def disable_transformations(self) -> INCProtocolIterator:
        return NCProtocolIterator(self.protocol,
                                  swap_train_test_transformations=self.are_train_test_transformations_swapped,
                                  are_transformations_disabled=True,
                                  initial_current_task=self.current_task)

    def enable_transformations(self) -> INCProtocolIterator:
        return NCProtocolIterator(self.protocol,
                                  swap_train_test_transformations=self.are_train_test_transformations_swapped,
                                  are_transformations_disabled=False,
                                  initial_current_task=self.current_task)

    def __get_tasks_classes(self, task_start: int, task_end: int = -1):
        if task_end < 0:
            task_end = self.protocol.n_tasks

        all_classes = []
        for task_idx in range(task_start, task_end):
            all_classes.append(self.protocol.get_task_classes(task_idx))

        if len(all_classes) == 0:
            return torch.tensor([], dtype=torch.long)
        return torch.cat(all_classes)

    def __go_to_next_task(self):
        if self.current_task == (self.protocol.n_tasks - 1):
            raise StopIteration()

        self.current_task += 1
        classes_start_idx = sum(self.protocol.classes_per_task[:self.current_task])
        classes_end_idx = classes_start_idx + self.protocol.classes_per_task[self.current_task]

        self.classes_in_this_task = self.protocol.get_task_classes(self.current_task)

        self.prev_classes = self.classes_seen_so_far
        self.future_classes = self.__get_tasks_classes(classes_end_idx)
        self.classes_seen_so_far = torch.cat([self.classes_seen_so_far, self.classes_in_this_task])

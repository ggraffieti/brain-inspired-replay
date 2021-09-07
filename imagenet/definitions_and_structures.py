from enum import Enum
from typing import Protocol, Any, SupportsFloat, Dict, Sequence, Optional, List, TypeVar, Generic

from torch import Tensor
from torch.utils.data import Dataset


class CumulativeStatistic:
    def __init__(self):
        self.values_sum = 0.0
        self.overall_count = 0.0
        self.average = 0.0

    def update_using_counts(self, value: SupportsFloat, count: SupportsFloat = 1.0):
        value = float(value)
        count = float(count)
        self.values_sum += value
        self.overall_count += count
        self.average = self.values_sum / self.overall_count

    def update_using_averages(self, value: SupportsFloat, count: SupportsFloat = 1.0):
        value = float(value) * float(count)
        count = float(count)
        self.values_sum += value
        self.overall_count += count
        self.average = self.values_sum / self.overall_count

    def __float__(self):
        return float(self.average)


class GetItemType(Protocol):
    def __getitem__(self, index: Any):
        ...


T_co = TypeVar('T_co', covariant=True)


class IDataset(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co:
        ...

    def __add__(self, other: Any) -> 'IDataset[T_co]':
        ...

    def __len__(self) -> int:
        ...


class DatasetThatSupportsTargets(IDataset, Protocol):
    targets: Any


class DatasetPart(Enum):
    CURRENT_TASK = 1  # Classes in this task only
    CUMULATIVE = 2  # Encountered classes (including classes in this task)
    OLD = 3  # Encountered classes (excluding classes in this task)
    FUTURE = 4  # Future classes
    COMPLETE_SET = 5  # All classes (encountered + not seen yet)


class DatasetType(Enum):
    VALIDATION = 1  # Validation (or test) set
    TRAIN = 2  # Training set


class INCProtocol(Protocol):
    train_dataset: DatasetThatSupportsTargets
    test_dataset: DatasetThatSupportsTargets
    validation_dataset: DatasetThatSupportsTargets
    n_tasks: int
    n_classes: int
    classes_order: Tensor
    train_transform: Any
    train_target_transform: Any
    test_transform: Any
    test_target_transform: Any
    classes_per_task: List[int]

    def __iter__(self) -> 'INCProtocolIterator':
        pass

    def get_task_classes(self, task_id: int) -> Tensor:
        pass

    def get_task_classes_mapping(self) -> Tensor:
        pass


class INCProtocolIterator(Protocol):
    current_task: int
    protocol: INCProtocol
    are_train_test_transformations_swapped: bool
    are_transformations_disabled: bool
    classes_seen_so_far: Tensor
    classes_in_this_task: Tensor
    prev_classes: Tensor
    train_subset_factory: Any
    test_subset_factory: Any

    def __next__(self) -> (Dataset, 'INCProtocolIterator'):
        pass

    # Training set utils
    def get_current_training_set(self, bucket_classes=True, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        pass

    def get_task_training_set(self, task_id: int, bucket_classes=True, sort_classes=False,
                              sort_indexes=False) -> DatasetThatSupportsTargets:
        pass

    def get_cumulative_training_set(self, include_current_task: bool = True, bucket_classes=True, sort_classes=False,
                                    sort_indexes=False) -> DatasetThatSupportsTargets:
        pass

    def get_complete_training_set(self, bucket_classes=True, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        pass

    def get_future_training_set(self, bucket_classes=True, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        pass

    def get_training_set_part(self, dataset_part: DatasetPart, bucket_classes=True, sort_classes=False,
                              sort_indexes=False) -> DatasetThatSupportsTargets:
        pass

    # Test set utils
    def get_current_test_set(self, bucket_classes=True, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        pass

    def get_cumulative_test_set(self, include_current_task: bool = True, bucket_classes=True, sort_classes=False,
                                sort_indexes=False) -> DatasetThatSupportsTargets:
        pass

    def get_task_test_set(self, task_id: int, bucket_classes=True, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        pass

    def get_complete_test_set(self, bucket_classes=True, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        pass

    def get_future_test_set(self, bucket_classes=True, sort_classes=False, sort_indexes=False) \
            -> DatasetThatSupportsTargets:
        pass

    def get_test_set_part(self, dataset_part: DatasetPart, bucket_classes=True, sort_classes=False,
                          sort_indexes=False) -> DatasetThatSupportsTargets:
        pass

    # Transformation utility function. Useful if you want to test on the training set (using test transformations)
    def swap_transformations(self) -> 'INCProtocolIterator':
        pass

    def disable_transformations(self) -> 'INCProtocolIterator':
        pass

    def enable_transformations(self) -> 'INCProtocolIterator':
        pass


# TODO: epoch?
class ValidationResult:
    def __init__(self, task: int, accuracies_top_k: Dict[int, float], accuracy_per_class: Dict[int, float],
                 confusion_matrix: Tensor, loss: float, loss_per_class: Dict[int, CumulativeStatistic],
                 validation_type: DatasetPart, validation_dataset: DatasetType, task_info: INCProtocolIterator):
        self.task: int = task
        self.accuracies_top_k: Dict[int, float] = accuracies_top_k
        self.accuracy_per_class: Dict[int, float] = accuracy_per_class
        self.confusion_matrix: Tensor = confusion_matrix
        self.loss: float = loss
        self.loss_per_class: Dict[int, CumulativeStatistic] = loss_per_class
        self.dataset_part: DatasetPart = validation_type
        self.validation_dataset: DatasetType = validation_dataset
        # Task info contains: classes in current task, previously encountered classes, all classes in protocol, etc.
        self.task_info: INCProtocolIterator = task_info


# TODO: running confusion matrix?
class TrainingStepResult:
    def __init__(self, task: int,
                 epoch: int,
                 training_loss: float,
                 running_accuracy_top_k: Dict[int, float],
                 loss_per_class: Dict[int, CumulativeStatistic],
                 task_info: INCProtocolIterator):
        self.task: int = task
        self.epoch: int = epoch
        self.training_loss: float = training_loss
        self.running_accuracy_top_k: Dict[int, float] = running_accuracy_top_k
        self.loss_per_class: Dict[int, CumulativeStatistic] = loss_per_class
        # Task info contains: classes in current task, previously encountered classes, all classes in protocol, etc.
        self.task_info: INCProtocolIterator = task_info


class TrainingStepResultBuilder:
    def __init__(self, task: int,
                 initial_epoch: int,
                 required_accuracy_top_k: Sequence[int],
                 task_info: INCProtocolIterator,
                 aggregate_on_epoch: bool):  # TODO: more fine grained control over aggregation (crete enum)
        self.task: int = task
        self.epoch: int = initial_epoch
        self.required_accuracy_top_k: Sequence[int] = required_accuracy_top_k
        # Task info contains: classes in current task, previously encountered classes, all classes in protocol, etc.
        self.task_info: INCProtocolIterator = task_info
        self.aggregate_on_epoch = aggregate_on_epoch

        self._training_loss: CumulativeStatistic = CumulativeStatistic()
        self._running_accuracy_top_k: Dict[int, CumulativeStatistic] = {}
        self._loss_per_class: Dict[int, CumulativeStatistic] = {}
        self._last_result: Optional[TrainingStepResult] = None
        self._reset_internal_stats()

    def add_iteration_result(self, n_patterns: int, loss: SupportsFloat,  accuracy_top_k: Dict[int, SupportsFloat],
                             avg_loss_per_class: Dict[int, SupportsFloat]):
        self._training_loss.update_using_averages(loss, n_patterns)
        for top_k in self._running_accuracy_top_k:
            self._running_accuracy_top_k[top_k].update_using_averages(accuracy_top_k[top_k], n_patterns)

        for class_idx in self._loss_per_class:
            self._loss_per_class[class_idx].update_using_averages(avg_loss_per_class[class_idx], n_patterns)

        if self.aggregate_on_epoch:
            self._last_result = None
        else:
            self._prepare_aggregate_result()

    def epoch_ended(self):
        if self.aggregate_on_epoch:
            self._prepare_aggregate_result()

        self.epoch += 1

    def get_aggregate_result(self) -> Optional[TrainingStepResult]:
        return self._last_result

    def _reset_internal_stats(self):
        self._training_loss = CumulativeStatistic()
        self._running_accuracy_top_k = {}
        self._loss_per_class = {}

        for class_idx in self.task_info.protocol.classes_order:
            self._loss_per_class[int(class_idx)] = CumulativeStatistic()

        for top_k in self.required_accuracy_top_k:
            self._running_accuracy_top_k[top_k] = CumulativeStatistic()

    def _prepare_aggregate_result(self) -> TrainingStepResult:
        aggregate_running_accuracy_top_k: Dict[int, float] = {}

        for top_k in self._running_accuracy_top_k:
            aggregate_running_accuracy_top_k[top_k] = self._running_accuracy_top_k[top_k].average

        self._last_result = TrainingStepResult(self.task, self.epoch, self._training_loss.average,
                                               aggregate_running_accuracy_top_k, self._loss_per_class, self.task_info)
        self._reset_internal_stats()
        return self._last_result

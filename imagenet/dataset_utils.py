from typing import SupportsInt

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor


def get_dataset_per_pixel_mean(dataset: Dataset) -> Tensor:
    result = None
    patterns_count = 0

    for img_pattern, _ in dataset:
        if result is None:
            # Only on first iteration
            result = torch.zeros_like(img_pattern, dtype=torch.float)

        result += img_pattern
        patterns_count += 1

    if result is None:
        result = torch.empty(0, dtype=torch.float)
    else:
        result = result / patterns_count

    return result


def make_single_pattern_one_hot(input_label: SupportsInt, n_classes: int, dtype=torch.float) -> Tensor:
    target = torch.zeros(n_classes, dtype=dtype)
    target[int(input_label)] = 1
    return target


def make_batch_one_hot(input_tensor: Tensor, n_classes: int, dtype=torch.float) -> Tensor:
    targets = torch.zeros(input_tensor.shape[0], n_classes, dtype=dtype)
    targets[range(len(input_tensor)), input_tensor.long()] = 1
    return targets


def load_all_dataset(dataset: Dataset, num_workers: int = 0):
    """
    Retrieves the contents of a whole dataset by using a DataLoader
    :param dataset: The dataset
    :param num_workers: The number of workers the DataLoader should use.
        Defaults to 0.
    :return: The content of the whole Dataset
    """
    # DataLoader parallelism is batch-based. By using "len(dataset)/num_workers"
    # as the batch size, num_workers [+1] batches will be loaded thus
    # using the required number of workers.
    if num_workers > 0:
        batch_size = max(1, len(dataset) // num_workers)
    else:
        batch_size = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                        num_workers=num_workers)
    batches_x = []
    batches_y = []
    for batch_x, batch_y in loader:
        batches_x.append(batch_x)
        batches_y.append(batch_y)

    x, y = torch.cat(batches_x), torch.cat(batches_y)
    return x, y


__all__ = ['get_dataset_per_pixel_mean', 'make_single_pattern_one_hot', 'make_batch_one_hot', 'load_all_dataset']

import pickle
from pathlib import Path

import configparser

import torch
from torchvision import transforms

from imagenet.cl_dataset_tools import NCProtocol
from imagenet.FileListDataset import ImageFilelist

from data.manipulate import UnNormalize


def make_imagenet_40_25_benchmark(class_order_pkl='../imagenet_seeds/seed_1993_imagenet_order_run_0.pkl',
                                    image_size=224, classes_in_batch=40):
    with open(class_order_pkl, 'rb') as f:
        fixed_class_order = pickle.load(f).tolist()

    base_imagenet_path = '/ssd1/datasets/imagenet_old_style/2012'
    try:
        if (Path.home() / 'dataset_folders.txt').exists():
            dataset_config = configparser.ConfigParser()
            dataset_config.read(str(Path.home() / 'dataset_folders.txt'))
            base_imagenet_path = str(dataset_config['DEFAULT']['imagenet'])
        else:
            print('No alternative imagenet path found')
    except Exception as exc:
        print(exc)

    print('Imagenet path is', base_imagenet_path)

    base_imagenet_path = Path(base_imagenet_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Define transformations for training, rehearsal and test patterns

    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(image_size + 10),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    traindir = str(base_imagenet_path / 'train')
    valdir = str(base_imagenet_path / 'valid')
    trainset = ImageFilelist(
        traindir,
        str((base_imagenet_path / 'caffe') / 'train_x144_filelist.txt'),
        transform=transform)
    testset = ImageFilelist(
        valdir,
        str((base_imagenet_path / 'caffe') / 'valid_x144_filelist.txt'),
        transform=transform_test)

    print('Imagenet training set contains', len(trainset), 'patterns')
    print('Imagenet test set contains', len(testset), 'patterns')

    n_classes = len(torch.unique(torch.as_tensor(trainset.targets)))
    print("Training set num classes: {}".format(n_classes))

    # Protocol definition
    # 25 tasks with 40 classes each
    protocol = NCProtocol(trainset,
                          testset,
                          n_tasks=25, shuffle=True, seed=None,
                          fixed_class_order=fixed_class_order,
                          remap_class_indexes_in_ascending_order=True)

    train_datasets = []
    test_datasets = []

    for (train_exp, exp_info) in protocol:
        train_datasets.append(train_exp)
        test_datasets.append(exp_info.get_cumulative_test_set())

    denormalize = UnNormalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    config = {'size': image_size, 'channels': 3, 'classes': n_classes, 'normalize': True, 'denormalize': denormalize}
    return (train_datasets, test_datasets), config, classes_in_batch


__all__ = ['make_imagenet_40_25_benchmark']

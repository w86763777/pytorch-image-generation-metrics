import glob
import logging
import os
from packaging import version

import pytest
import torch
from torchvision.datasets import CIFAR10

from pytorch_image_generation_metrics.utils import ImageDataset


TORCH_VERSION = version.parse(torch.__version__).base_version
TEST_ROOT = os.environ.get('TEST_ROOT', './tests')
TEST_NAME = TORCH_VERSION
PATH_CIFAR10 = "/tmp/cifar10"
PATH_CIFAR10_TRAIN = f"{PATH_CIFAR10}/train"
PATH_CIFAR10_TEST = f"{PATH_CIFAR10}/test"
PATH_CIFAR10_TRAIN_FID_REF_NP = f'{TEST_ROOT}/{TEST_NAME}/cifar10.train.npz'
PATH_CIFAR10_TEST_FID_REF_NP = f'{TEST_ROOT}/{TEST_NAME}/cifar10.test.npz'
PATH_CIFAR10_TRAIN_FID_REF_PT = f'{TEST_ROOT}/{TEST_NAME}/cifar10.train.pt.npz'
PATH_CIFAR10_TEST_FID_REF_PT = f'{TEST_ROOT}/{TEST_NAME}/cifar10.test.pt.npz'
NUM_WORKERS = os.environ.get('NUM_WORKERS', min(torch.get_num_threads(), 4))


def save_dataset(dataset, root):
    os.makedirs(root, exist_ok=True)
    for i, (x, _) in enumerate(dataset):
        x.save(os.path.join(root, f'{i + 1}.png'))


@pytest.fixture
def cifar10_test():
    dataset = CIFAR10(PATH_CIFAR10, train=False, download=True)
    if len(glob.glob(os.path.join(PATH_CIFAR10_TEST, '*.png'))) != len(dataset):
        save_dataset(dataset, root=PATH_CIFAR10_TEST)
    return ImageDataset(PATH_CIFAR10_TEST)


@pytest.fixture
def cifar10_train():
    dataset = CIFAR10(PATH_CIFAR10, train=True, download=True)
    if len(glob.glob(os.path.join(PATH_CIFAR10_TRAIN, '*.png'))) != len(dataset):
        save_dataset(dataset, root=PATH_CIFAR10_TRAIN)
    return ImageDataset(PATH_CIFAR10_TRAIN)


@pytest.fixture(autouse=True)
def set_caplog(caplog):
    caplog.set_level(logging.INFO)
    return caplog


def format_relative_error(name, value, expected):
    return f'{name}: {value:.9f}, expected: {expected:.9f}, relative error: {abs(value - expected) / expected: .5f}'

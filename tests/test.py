import unittest
import os

import torch
import torch.multiprocessing
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from pytorch_gan_metrics.utils import (
    ImageDataset,
    get_inception_score,
    get_inception_score_from_directory,
    get_fid,
    get_fid_from_directory,
    get_inception_score_and_fid,
    get_inception_score_and_fid_from_directory)

from pytorch_gan_metrics.calc_fid_stats import calc_and_save_stats

# Fix RuntimeError: Too many open files.
torch.multiprocessing.set_sharing_strategy('file_system')
num_workers = os.cpu_count()
if num_workers is not None:
    num_workers = min(num_workers - 1, 3)
else:
    num_workers = 3


class AllTestCase(unittest.TestCase):
    pass


def download_cifar10(root):
    dataset_train = CIFAR10(root, train=True, download=True)
    dataset_test = CIFAR10(root, train=False, download=True)
    os.makedirs(os.path.join(root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(root, 'test'), exist_ok=True)
    for i, (x, _) in enumerate(dataset_train):
        x.save(os.path.join(root, f'train/{i + 1}.png'))
    for i, (x, _) in enumerate(dataset_test):
        x.save(os.path.join(root, f'test/{i + 1}.png'))


def test_calc_and_save_stats(input_path, output_path, use_torch):
    calc_and_save_stats(
        input_path, output_path, use_torch=use_torch,
        num_workers=num_workers, verbose=False)
    return None


def test_inception_score_dataloader(path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=num_workers)
    IS, IS_std = get_inception_score(loader, use_torch=use_torch)
    return IS, IS_std


def test_inception_score_tensor(path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=num_workers)
    images = torch.cat([batch_images for batch_images in loader], dim=0)
    IS, IS_std = get_inception_score(images, use_torch=use_torch)
    return IS, IS_std


def test_inception_score_from_directory(path, use_torch):
    IS, IS_std = get_inception_score_from_directory(
        path, use_torch=use_torch)
    return IS, IS_std


def test_fid_dataloader(path, fid_stats_path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=num_workers)
    FID = get_fid(loader, fid_stats_path, use_torch=use_torch)
    return FID,


def test_fid_tensor(path, fid_stats_path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=num_workers)
    images = torch.cat([batch_images for batch_images in loader], dim=0)
    FID = get_fid(images, fid_stats_path, use_torch=use_torch)
    return FID,


def test_fid_from_directory(path, fid_stats_path, use_torch):
    FID = get_fid_from_directory(
        path, fid_stats_path, use_torch=use_torch)
    return FID,


def test_inception_score_and_fid_dataloader(path, fid_stats_path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=num_workers)
    (IS, IS_std), FID = get_inception_score_and_fid(
        loader, fid_stats_path, use_torch=use_torch)
    return IS, IS_std, FID


def test_inception_score_and_fid_tensor(path, fid_stats_path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=num_workers)
    images = torch.cat([batch_images for batch_images in loader], dim=0)
    (IS, IS_std), FID = get_inception_score_and_fid(
        images, fid_stats_path, use_torch=use_torch)
    return IS, IS_std, FID


def test_inception_score_and_fid_from_directory(
        path, fid_stats_path, use_torch):
    (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
        path, fid_stats_path, use_torch=use_torch)
    return IS, IS_std, FID


def create_test(test_fn, inputs, expected_outputs):
    def do_test_expected(self: AllTestCase):
        outputs = test_fn(**inputs)
        if outputs is None:
            return
        for output, expected_output in zip(outputs, expected_outputs):
            relative_err = abs((output - expected_output) / expected_output)
            self.assertLess(
                relative_err, 1e-4, msg=f'{output} != {expected_output}')
    return do_test_expected


def main(name, results_root):
    # Py39, torch1.12.1, CUDA 10.2, CIFAR10
    NP_IS = 11.263923779476338
    PT_IS = 11.263922691345215
    NP_IS_STD = 0.1446554877884545
    PT_IS_STD = 0.15248057246208190
    NP_FID = 3.151765033148024
    PT_FID = 3.150909423828125

    PATH_CIFAR10 = f"{results_root}/cifar10"
    PATH_CIFAR10_TRAIN = f"{results_root}/cifar10/train"
    PATH_CIFAR10_TEST = f"{results_root}/cifar10/test"
    PATH_CIFAR10_STATS_NP_TRAIN = f'{results_root}/{name}/cifar10.train.npz'
    PATH_CIFAR10_STATS_NP_TEST = f'{results_root}/{name}/cifar10.test.npz'
    PATH_CIFAR10_STATS_PT_TRAIN = f'{results_root}/{name}/cifar10.train.pt.npz'
    PATH_CIFAR10_STATS_PT_TEST = f'{results_root}/{name}/cifar10.test.pt.npz'

    configs_calc_stats = [
        {
            'inputs': {
                'input_path': PATH_CIFAR10_TRAIN,
                'output_path': PATH_CIFAR10_STATS_NP_TRAIN,
                'use_torch': False,
            },
            'expected_outputs': None,
        },
        {
            'inputs': {
                'input_path': PATH_CIFAR10_TRAIN,
                'output_path': PATH_CIFAR10_STATS_PT_TRAIN,
                'use_torch': True,
            },
            'expected_outputs': None,
        },
        {
            'inputs': {
                'input_path': PATH_CIFAR10_TEST,
                'output_path': PATH_CIFAR10_STATS_NP_TEST,
                'use_torch': False,
            },
            'expected_outputs': None,
        },
        {
            'inputs': {
                'input_path': PATH_CIFAR10_TEST,
                'output_path': PATH_CIFAR10_STATS_PT_TEST,
                'use_torch': True,
            },
            'expected_outputs': None,
        },
    ]
    configs_calc_metrics = [
        {
            'test_fns': [
                test_inception_score_dataloader,
                test_inception_score_tensor,
                test_inception_score_from_directory
            ],
            'args': [
                {
                    'inputs': {
                        'path': PATH_CIFAR10_TRAIN,
                        'use_torch': False
                    },
                    'expected_outputs': [NP_IS, NP_IS_STD]
                },
                {
                    'inputs': {
                        'path': PATH_CIFAR10_TRAIN,
                        'use_torch': True
                    },
                    'expected_outputs': [PT_IS, PT_IS_STD]
                }
            ],
        },
        {
            'test_fns': [
                test_fid_dataloader,
                test_fid_tensor,
                test_fid_from_directory
            ],
            'args': [
                {
                    'inputs': {
                        'path': PATH_CIFAR10_TRAIN,
                        'use_torch': False,
                        'fid_stats_path': PATH_CIFAR10_STATS_NP_TEST,
                    },
                    'expected_outputs': [NP_FID]
                },
                {
                    'inputs': {
                        'path': PATH_CIFAR10_TRAIN,
                        'use_torch': True,
                        'fid_stats_path': PATH_CIFAR10_STATS_PT_TEST,
                    },
                    'expected_outputs': [PT_FID]
                }
            ],
        },
        {
            'test_fns': [
                test_inception_score_and_fid_dataloader,
                test_inception_score_and_fid_tensor,
                test_inception_score_and_fid_from_directory
            ],
            'args': [
                {
                    'inputs': {
                        'path': PATH_CIFAR10_TRAIN,
                        'use_torch': False,
                        'fid_stats_path': PATH_CIFAR10_STATS_NP_TEST,
                    },
                    'expected_outputs': [
                        NP_IS,
                        NP_IS_STD,
                        NP_FID]
                },
                {
                    'inputs': {
                        'path': PATH_CIFAR10_TRAIN,
                        'use_torch': True,
                        'fid_stats_path': PATH_CIFAR10_STATS_PT_TEST,
                    },
                    'expected_outputs': [
                        PT_IS,
                        PT_IS_STD,
                        PT_FID]
                }
            ],
        },
    ]

    for k, args in enumerate(configs_calc_stats):
        test_method = create_test(
            test_fn=test_calc_and_save_stats,
            inputs=args['inputs'],
            expected_outputs=args['expected_outputs'],
        )
        test_method.__name__ = f'{test_calc_and_save_stats.__name__}_{k}'
        setattr(AllTestCase, test_method.__name__, test_method)

    for config_group in configs_calc_metrics:
        for test_fn in config_group['test_fns']:
            for k, args in enumerate(config_group['args']):
                test_method = create_test(
                    test_fn=test_fn,
                    inputs=args['inputs'],
                    expected_outputs=args['expected_outputs'],
                )
                test_method.__name__ = f'{test_fn.__name__}_{k}'
                setattr(AllTestCase, test_method.__name__, test_method)

    download_cifar10(root=PATH_CIFAR10)
    unittest.main(verbosity=2, failfast=True)


if __name__ == '__main__':
    NAME = os.environ.get('ENV_NAME')
    RESULTS_ROOT = os.environ.get('RESULTS_ROOT')
    main(NAME, RESULTS_ROOT)

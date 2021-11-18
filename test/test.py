import unittest
import os

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing

from pytorch_gan_metrics.utils import (
    ImageDataset,
    get_inception_score,
    get_inception_score_from_directory,
    get_fid,
    get_fid_from_directory,
    get_inception_score_and_fid,
    get_inception_score_and_fid_from_directory)

# Fix RuntimeError: Too many open files.
torch.multiprocessing.set_sharing_strategy('file_system')
num_workers = os.cpu_count()
if num_workers is not None:
    num_workers = min(num_workers - 1, 3)
else:
    num_workers = 3


class AllTestCase(unittest.TestCase):
    pass


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
    def do_test_expected(self):
        outputs = test_fn(**inputs)
        for output, expected_output in zip(outputs, expected_outputs):
            relative_err = abs((output - expected_output) / expected_output)
            self.assertLess(relative_err, 1e-4)
    return do_test_expected


if __name__ == '__main__':
    # CUDA 10.2
    NP_IS = 11.265363978062746
    NP_IS_STD = 0.08081806306810498
    NP_FID = 3.151765556578084
    PT_IS = 11.265363693237305
    PT_IS_STD = 0.08519038558006287
    PT_FID = 3.145416259765625
    configs = [
        {
            'test_fns': [
                test_inception_score_dataloader,
                test_inception_score_tensor,
                test_inception_score_from_directory
            ],
            'args': [
                {
                    'inputs': {
                        'path': './cifar10/train',
                        'use_torch': False
                    },
                    'outputs': [NP_IS, NP_IS_STD]
                },
                {
                    'inputs': {
                        'path': './cifar10/train',
                        'use_torch': True
                    },
                    'outputs': [PT_IS, PT_IS_STD]
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
                        'path': './cifar10/train',
                        'use_torch': False,
                        'fid_stats_path': 'cifar10.test.npz',
                    },
                    'outputs': [NP_FID]
                },
                {
                    'inputs': {
                        'path': './cifar10/train',
                        'use_torch': True,
                        'fid_stats_path': 'cifar10.test.npz',
                    },
                    'outputs': [PT_FID]
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
                        'path': './cifar10/train',
                        'use_torch': False,
                        'fid_stats_path': 'cifar10.test.npz',
                    },
                    'outputs': [
                        NP_IS,
                        NP_IS_STD,
                        NP_FID]
                },
                {
                    'inputs': {
                        'path': './cifar10/train',
                        'use_torch': True,
                        'fid_stats_path': 'cifar10.test.npz',
                    },
                    'outputs': [
                        PT_IS,
                        PT_IS_STD,
                        PT_FID]
                }
            ],
        },
    ]
    for config in configs:
        for test_fn in config['test_fns']:
            for k, arg in enumerate(config['args']):
                test_method = create_test(
                    test_fn=test_fn,
                    inputs=arg['inputs'],
                    expected_outputs=arg['outputs'],
                )
                test_method.__name__ = f'{test_fn.__name__}_{k}'
                setattr(AllTestCase, test_method.__name__, test_method)
    unittest.main(verbosity=2)

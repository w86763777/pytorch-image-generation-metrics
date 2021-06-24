import unittest

import torch
from torch.utils.data import DataLoader

from pytorch_gan_metrics.utils import (
    ImageDataset,
    get_inception_score,
    get_inception_score_from_directory,
    get_fid,
    get_fid_from_directory,
    get_inception_score_and_fid,
    get_inception_score_and_fid_from_directory)


class AllTestCase(unittest.TestCase):
    pass


def test_inception_score_dataloader(path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=4)
    IS, IS_std = get_inception_score(loader, use_torch=use_torch)
    return IS, IS_std


def test_inception_score_tensor(path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=4)
    images = torch.cat([batch_images for batch_images in loader], dim=0)
    IS, IS_std = get_inception_score(images, use_torch=use_torch)
    return IS, IS_std


def test_inception_score_from_directory(path, use_torch):
    IS, IS_std = get_inception_score_from_directory(
        path, use_torch=use_torch)
    return IS, IS_std


def test_fid_dataloader(path, fid_stats_path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=4)
    FID = get_fid(loader, fid_stats_path, use_torch=use_torch)
    return FID,


def test_fid_tensor(path, fid_stats_path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=4)
    images = torch.cat([batch_images for batch_images in loader], dim=0)
    FID = get_fid(images, fid_stats_path, use_torch=use_torch)
    return FID,


def test_fid_from_directory(path, fid_stats_path, use_torch):
    FID = get_fid_from_directory(
        path, fid_stats_path, use_torch=use_torch)
    return FID,


def test_inception_score_and_fid_dataloader(path, fid_stats_path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=4)
    (IS, IS_std), FID = get_inception_score_and_fid(
        loader, fid_stats_path, use_torch=use_torch)
    return IS, IS_std, FID


def test_inception_score_and_fid_tensor(path, fid_stats_path, use_torch):
    loader = DataLoader(
        ImageDataset(path), batch_size=50, num_workers=4)
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
            self.assertEqual(output, expected_output)
    return do_test_expected


if __name__ == '__main__':
    # numpy 11.267066921084192 0.2032413984924413 3.1517650331493883
    # torch 11.26706600189209 0.2142351269721985 3.137664794921875
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
                    'outputs': [11.267066921084192, 0.2032413984924413]
                },
                {
                    'inputs': {
                        'path': './cifar10/train',
                        'use_torch': True
                    },
                    'outputs': [11.26706600189209, 0.2142351269721985]
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
                    'outputs': [3.1517650331493883]
                },
                {
                    'inputs': {
                        'path': './cifar10/train',
                        'use_torch': True,
                        'fid_stats_path': 'cifar10.test.npz',
                    },
                    'outputs': [3.137664794921875]
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
                        11.267066921084192,
                        0.2032413984924413,
                        3.1517650331493883]
                },
                {
                    'inputs': {
                        'path': './cifar10/train',
                        'use_torch': True,
                        'fid_stats_path': 'cifar10.test.npz',
                    },
                    'outputs': [
                        11.26706600189209,
                        0.2142351269721985,
                        3.137664794921875]
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

import os

import pytest

from pytorch_image_generation_metrics.fid_ref import calc_fid_ref
from .conftest import (
    PATH_CIFAR10_TRAIN,
    PATH_CIFAR10_TEST,
    PATH_CIFAR10_TRAIN_FID_REF_NP,
    PATH_CIFAR10_TEST_FID_REF_NP,
    PATH_CIFAR10_TRAIN_FID_REF_PT,
    PATH_CIFAR10_TEST_FID_REF_PT,
    NUM_WORKERS,
)


@pytest.mark.fid
@pytest.mark.order(0)
class TestFidRef:
    @pytest.mark.parametrize("output_path,use_torch", [
        (PATH_CIFAR10_TEST_FID_REF_NP, False),
        (PATH_CIFAR10_TEST_FID_REF_PT, True),
    ])
    def test_cifar10_test_fid_ref(self, output_path, use_torch):
        if not os.path.exists(output_path):
            calc_fid_ref(
                PATH_CIFAR10_TEST, output_path, use_torch=use_torch,
                num_workers=NUM_WORKERS,
                verbose=False)

    @pytest.mark.parametrize("output_path,use_torch", [
        (PATH_CIFAR10_TRAIN_FID_REF_NP, False),
        (PATH_CIFAR10_TRAIN_FID_REF_PT, True),
    ])
    def test_cifar10_train_fid_ref(self, output_path, use_torch):
        if not os.path.exists(output_path):
            calc_fid_ref(
                PATH_CIFAR10_TRAIN, output_path, use_torch=use_torch,
                num_workers=NUM_WORKERS,
                verbose=False)

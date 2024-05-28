import logging

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_image_generation_metrics.utils import (
    get_inception_score_and_fid,
    get_inception_score_and_fid_from_directory,
)
from .conftest import (
    PATH_CIFAR10_TEST,
    PATH_CIFAR10_TRAIN_FID_REF_NP,
    PATH_CIFAR10_TRAIN_FID_REF_PT,
    NUM_WORKERS,
    format_relative_error,
)
from .test_fid import NP_FID, PT_FID
from .test_inception_score import NP_IS, NP_IS_STD, PT_IS, PT_IS_STD


@pytest.mark.fid
@pytest.mark.inception_score
@pytest.mark.order(1)
class TestAllMetrics:
    @pytest.mark.parametrize("batch_size, fid_ref, use_torch, expected_is, expected_std, expected_fid", [
        (50, PATH_CIFAR10_TRAIN_FID_REF_NP, False, NP_IS, NP_IS_STD, NP_FID),
        (50, PATH_CIFAR10_TRAIN_FID_REF_PT, True, PT_IS, PT_IS_STD, PT_FID),
    ])
    def test_inception_score_and_fid_dataloader(
        self,
        cifar10_test,
        batch_size,
        fid_ref,
        use_torch,
        expected_is,
        expected_std,
        expected_fid
    ):
        loader = DataLoader(
            cifar10_test, batch_size=batch_size, num_workers=NUM_WORKERS)
        (IS, IS_std), FID = get_inception_score_and_fid(
            loader, fid_ref, use_torch=use_torch)
        logging.info(format_relative_error("IS", IS, expected_is))
        logging.info(format_relative_error("IS_STD", IS_std, expected_std))
        logging.info(format_relative_error("FID", FID, expected_fid))
        assert torch.allclose(torch.tensor(IS), torch.tensor(expected_is), rtol=1e-3)
        assert torch.allclose(torch.tensor(IS_std), torch.tensor(expected_std), rtol=1e-3)
        assert torch.allclose(torch.tensor(FID), torch.tensor(expected_fid), rtol=1e-3)

    @pytest.mark.parametrize("batch_size, fid_ref, use_torch, expected_is, expected_std, expected_fid", [
        (50, PATH_CIFAR10_TRAIN_FID_REF_NP, False, NP_IS, NP_IS_STD, NP_FID),
        (50, PATH_CIFAR10_TRAIN_FID_REF_PT, True, PT_IS, PT_IS_STD, PT_FID),
    ])
    def test_inception_score_and_fid_tensor(
        self,
        cifar10_test,
        batch_size,
        fid_ref,
        use_torch,
        expected_is,
        expected_std,
        expected_fid
    ):
        loader = DataLoader(
            cifar10_test, batch_size=batch_size, num_workers=NUM_WORKERS)
        images = torch.cat([batch_images for batch_images in loader], dim=0)
        (IS, IS_std), FID = get_inception_score_and_fid(
            images, fid_ref, use_torch=use_torch)
        logging.info(format_relative_error("IS", IS, expected_is))
        logging.info(format_relative_error("IS_STD", IS_std, expected_std))
        logging.info(format_relative_error("FID", FID, expected_fid))
        assert torch.allclose(torch.tensor(IS), torch.tensor(expected_is), rtol=1e-3)
        assert torch.allclose(torch.tensor(IS_std), torch.tensor(expected_std), rtol=1e-3)
        assert torch.allclose(torch.tensor(FID), torch.tensor(expected_fid), rtol=1e-3)

    @pytest.mark.parametrize("batch_size, fid_ref, use_torch, expected_is, expected_std, expected_fid", [
        (50, PATH_CIFAR10_TRAIN_FID_REF_NP, False, NP_IS, NP_IS_STD, NP_FID),
        (50, PATH_CIFAR10_TRAIN_FID_REF_PT, True, PT_IS, PT_IS_STD, PT_FID),
    ])
    def test_inception_score_and_fid_from_directory(
        self,
        batch_size,
        fid_ref,
        use_torch,
        expected_is,
        expected_std,
        expected_fid
    ):
        (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
            PATH_CIFAR10_TEST, fid_ref, batch_size=batch_size,
            use_torch=use_torch)
        logging.info(format_relative_error("IS", IS, expected_is))
        logging.info(format_relative_error("IS_STD", IS_std, expected_std))
        logging.info(format_relative_error("FID", FID, expected_fid))
        assert torch.allclose(torch.tensor(IS), torch.tensor(expected_is), rtol=1e-3)
        assert torch.allclose(torch.tensor(IS_std), torch.tensor(expected_std), rtol=1e-3)
        assert torch.allclose(torch.tensor(FID), torch.tensor(expected_fid), rtol=1e-3)

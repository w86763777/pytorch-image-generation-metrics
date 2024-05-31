# from packaging import version
import logging

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_image_generation_metrics.utils import (
    get_inception_score,
    get_inception_score_from_directory,
)
from .conftest import (
    PATH_CIFAR10_TEST,
    NUM_WORKERS,
    format_relative_error,
)

NP_IS = 10.968601098
NP_IS_STD = 0.193806868
PT_IS = 10.968603134        # torch==2.3.0
PT_IS_STD = 0.204290837     # torch==2.3.0


@pytest.mark.inception_score
@pytest.mark.order(1)
class TestInceptionScore:
    @pytest.mark.parametrize("batch_size, use_torch, expected_is, expected_std", [
        (50, False, NP_IS, NP_IS_STD),
        (50, True, PT_IS, PT_IS_STD),
    ])
    def test_inception_score_dataloader(
        self,
        cifar10_test,
        batch_size,
        use_torch,
        expected_is,
        expected_std
    ):
        loader = DataLoader(
            cifar10_test, batch_size=batch_size, num_workers=NUM_WORKERS)
        IS, IS_std = get_inception_score(loader, use_torch=use_torch)
        logging.info(format_relative_error("IS", IS, expected_is))
        logging.info(format_relative_error("IS_STD", IS_std, expected_std))
        assert torch.allclose(torch.tensor(IS), torch.tensor(expected_is), rtol=1e-2)
        assert torch.allclose(torch.tensor(IS_std), torch.tensor(expected_std), rtol=1e-2)

    @pytest.mark.parametrize("batch_size, use_torch, expected_is, expected_std", [
        (50, False, NP_IS, NP_IS_STD),
        (50, True, PT_IS, PT_IS_STD),
    ])
    def test_inception_score_tensor(
        self,
        cifar10_test,
        batch_size,
        use_torch,
        expected_is,
        expected_std
    ):
        loader = DataLoader(
            cifar10_test, batch_size=batch_size, num_workers=NUM_WORKERS)
        images = torch.cat([batch_images for batch_images in loader], dim=0)
        IS, IS_std = get_inception_score(images, use_torch=use_torch)
        logging.info(format_relative_error("IS", IS, expected_is))
        logging.info(format_relative_error("IS_STD", IS_std, expected_std))
        assert torch.allclose(torch.tensor(IS), torch.tensor(expected_is), rtol=1e-2)
        assert torch.allclose(torch.tensor(IS_std), torch.tensor(expected_std), rtol=1e-2)

    @pytest.mark.parametrize("batch_size, use_torch, expected_is, expected_std", [
        (50, False, NP_IS, NP_IS_STD),
        (50, True, PT_IS, PT_IS_STD),
    ])
    def test_inception_score_from_directory(
        self,
        batch_size,
        use_torch,
        expected_is,
        expected_std
    ):
        IS, IS_std = get_inception_score_from_directory(
            PATH_CIFAR10_TEST, batch_size=batch_size, use_torch=use_torch)
        logging.info(format_relative_error("IS", IS, expected_is))
        logging.info(format_relative_error("IS_STD", IS_std, expected_std))
        assert torch.allclose(torch.tensor(IS), torch.tensor(expected_is), rtol=1e-2)
        assert torch.allclose(torch.tensor(IS_std), torch.tensor(expected_std), rtol=1e-2)

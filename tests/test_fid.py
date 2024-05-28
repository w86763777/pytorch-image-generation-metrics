import logging
from packaging import version

import pytest
import torch
from torch.utils.data import DataLoader


from pytorch_image_generation_metrics.utils import (
    get_fid,
    get_fid_from_directory
)
from .conftest import (
    PATH_CIFAR10_TEST,
    PATH_CIFAR10_TRAIN_FID_REF_NP,
    PATH_CIFAR10_TRAIN_FID_REF_PT,
    NUM_WORKERS,
    format_relative_error,
)


NP_FID = 3.1525318697637204
if version.parse(torch.__version__).base_version == '2.3.0':
    PT_FID = 3.145660400390625
else:
    assert False, f'Unknown torch version: {torch.__version__}'


@pytest.mark.fid
@pytest.mark.order(1)
class TestFID:
    @pytest.mark.parametrize("batch_size, fid_ref, use_torch, expected_fid", [
        (50, PATH_CIFAR10_TRAIN_FID_REF_NP, False, NP_FID),
        (50, PATH_CIFAR10_TRAIN_FID_REF_PT, True, PT_FID),
    ])
    def test_fid_dataloader(
        self,
        cifar10_test,
        batch_size,
        fid_ref,
        use_torch,
        expected_fid
    ):
        loader = DataLoader(
            cifar10_test, batch_size=batch_size, num_workers=NUM_WORKERS)
        FID = get_fid(loader, fid_ref, use_torch=use_torch)
        logging.info(format_relative_error("FID", FID, expected_fid))
        assert torch.allclose(torch.tensor(FID), torch.tensor(expected_fid), rtol=1e-3)

    @pytest.mark.parametrize("batch_size, fid_ref, use_torch, expected_fid", [
        (50, PATH_CIFAR10_TRAIN_FID_REF_NP, False, NP_FID),
        (50, PATH_CIFAR10_TRAIN_FID_REF_PT, True, PT_FID),
    ])
    def test_fid_tensor(
        self,
        cifar10_test,
        batch_size,
        fid_ref,
        use_torch,
        expected_fid
    ):
        loader = DataLoader(
            cifar10_test, batch_size=batch_size, num_workers=NUM_WORKERS)
        images = torch.cat([batch_images for batch_images in loader], dim=0)
        FID = get_fid(images, fid_ref, use_torch=use_torch)
        logging.info(format_relative_error("FID", FID, expected_fid))
        assert torch.allclose(torch.tensor(FID), torch.tensor(expected_fid), rtol=1e-3)

    @pytest.mark.parametrize("batch_size, fid_ref, use_torch, expected_fid", [
        (50, PATH_CIFAR10_TRAIN_FID_REF_NP, False, NP_FID),
        (50, PATH_CIFAR10_TRAIN_FID_REF_PT, True, PT_FID),
    ])
    def test_fid_from_directory(
        self,
        batch_size,
        fid_ref,
        use_torch,
        expected_fid
    ):
        FID = get_fid_from_directory(
            PATH_CIFAR10_TEST, fid_ref, batch_size=batch_size,
            use_torch=use_torch)
        logging.info(format_relative_error("FID", FID, expected_fid))
        assert torch.allclose(torch.tensor(FID), torch.tensor(expected_fid), rtol=1e-3)

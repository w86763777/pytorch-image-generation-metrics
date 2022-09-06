import os
from typing import List, Union, Tuple, Optional
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import to_tensor

from .core import (
    get_inception_feature,
    calculate_inception_score,
    calculate_frechet_inception_distance,
    torch_cov)


class ImageDataset(Dataset):
    def __init__(self, root, exts=['png', 'jpg', 'JPEG'], transform=None,
                 num_images=None):
        self.paths = []
        self.transform = transform
        for ext in exts:
            self.paths.extend(
                list(glob(
                    os.path.join(root, '**/*.%s' % ext), recursive=True)))
        self.paths = self.paths[:num_images]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        image = image.convert('RGB')        # fix ImageNet grayscale images
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = to_tensor(image)
        return image


def get_inception_score_and_fid(
    images: Union[torch.FloatTensor, DataLoader],
    fid_stats_path: str,
    splits: int = 10,
    use_torch: bool = False,
    **kwargs,
) -> Tuple[Tuple[float, float], float]:
    """Calculate Inception Score and FID.
    For each image, only a forward propagation is required to
    calculating features for FID and Inception Score.

    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
                must be float tensor of range [0, 1].
        fid_stats_path: str, Path to pre-calculated statistic
        splits: The number of bins of Inception Score. Default is 10.
        use_torch: bool. The default value is False and the backend is same as
                   official implementation, i.e., numpy. If use_torch is
                   enableb, the backend linalg is implemented by torch, the
                   results are not guaranteed to be consistent with numpy, but
                   the speed can be accelerated by GPU.
        **kwargs: Please refer to `core.get_inception_feature` for other
                  arguments.
    Returns:
        inception_score: float tuple, (mean, std)
        fid: float
    """
    acts, probs = get_inception_feature(
        images, dims=[2048, 1008], use_torch=use_torch, **kwargs)

    # Inception Score
    inception_score, std = calculate_inception_score(probs, splits, use_torch)

    # Frechet Inception Distance
    f = np.load(fid_stats_path)
    mu, sigma = f['mu'][:], f['sigma'][:]
    f.close()
    fid = calculate_frechet_inception_distance(acts, mu, sigma, use_torch)

    return (inception_score, std), fid


def get_inception_score_and_fid_from_directory(
    path: str,
    fid_stats_path: str,
    exts: List[str] = ['png', 'jpg'],
    batch_size: int = 50,
    splits: int = 10,
    use_torch: bool = False,
    **kwargs
) -> Tuple[Tuple[float, float], float]:
    """Calculate Inception Score and FID of images in a directory
    Please refer to `get_inception_score_and_fid` for the arguments
    descriptions.

    Args:
        path: path to a image directory. It does not recursively inspect
        the subfolders.
        exts: target file extentions

    Returns:
        Inception Score: float tuple, mean and std
        FID: float
    """
    return get_inception_score_and_fid(
        images=DataLoader(ImageDataset(path, exts), batch_size=batch_size),
        fid_stats_path=fid_stats_path,
        splits=splits,
        use_torch=use_torch, **kwargs)


def get_fid(
    images: Union[torch.FloatTensor, DataLoader],
    fid_stats_path: str,
    use_torch: bool = False,
    **kwargs,
) -> Tuple[Tuple[float, float], float]:
    """Calculate Frechet Inception Distance.
    Please refer to `get_inception_score_and_fid` for the arguments
    descriptions.

    Returns:
        FID: float
    """
    acts, = get_inception_feature(
        images, dims=[2048], use_torch=use_torch, **kwargs)

    # Frechet Inception Distance
    f = np.load(fid_stats_path)
    mu, sigma = f['mu'][:], f['sigma'][:]
    f.close()
    fid = calculate_frechet_inception_distance(acts, mu, sigma, use_torch)

    return fid


def get_fid_from_directory(
    path: str,
    fid_stats_path: str,
    exts: List[str] = ['png', 'jpg'],
    batch_size: int = 50,
    use_torch: bool = False,
    **kwargs
) -> Tuple[Tuple[float, float], float]:
    """Calculate Frechet Inception Distance of images in a directory
    Please refer to `get_inception_score_and_fid` for the arguments
    descriptions.

    Args:
        path: path to a image directory. It does not recursively inspect
        the subfolders.
        exts: target file extentions

    Returns:
        FID: float
    """
    return get_fid(
        images=DataLoader(ImageDataset(path, exts), batch_size=batch_size),
        fid_stats_path=fid_stats_path,
        use_torch=use_torch,
        **kwargs)


def get_inception_score(
    images: Union[torch.FloatTensor, DataLoader],
    splits: int = 10,
    use_torch: bool = False,
    **kwargs,
) -> Tuple[Tuple[float, float], float]:
    """Calculate Inception Score.
    Please refer to `get_inception_score_and_fid` for the arguments
    descriptions.
    Returns:
        Inception Score: float tuple
    """
    probs, = get_inception_feature(
        images, dims=[1008], use_torch=use_torch, **kwargs)
    inception_score, std = calculate_inception_score(probs, splits, use_torch)
    return (inception_score, std)


def get_inception_score_from_directory(
    path: str,
    splits: int = 10,
    exts: List[str] = ['png', 'jpg'],
    batch_size: int = 50,
    use_torch: bool = False,
    **kwargs
) -> Tuple[Tuple[float, float], float]:
    """Calculate Frechet Inception Distance of images in a directory
    Please refer to `get_inception_score_and_fid` for the arguments
    descriptions.

    Args:
        path: path to a image directory. It does not recursively inspect
        the subfolders.
        exts: target file extentions

    Returns:
        FID: float
    """
    return get_inception_score(
        images=DataLoader(ImageDataset(path, exts), batch_size=batch_size),
        splits=splits,
        use_torch=use_torch,
        **kwargs)


def calc_and_save_stats(
    input_path: str,
    output_path: str,
    batch_size: int = 50,
    img_size: Optional[int] = None,
    use_torch: bool = False,
    num_workers: int = os.cpu_count(),
    verbose: bool = True,
) -> None:
    """Calculate the FID statistics and save them to a file.

    Args:
        input_path (str): Path to the image directory.
        output_path (str): Path to the output file.
        batch_size (int): Batch size. Defaults to 50.
        img_size (int): Image size. If None, use the original image size.
        num_workers (int): Number of dataloader workers. Default:
                           os.cpu_count().
    """
    if img_size is not None:
        transform = Compose([Resize([img_size, img_size]), ToTensor()])
    else:
        transform = ToTensor()

    dataset = ImageDataset(root=input_path, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)
    acts, = get_inception_feature(
        loader, dims=[2048], use_torch=use_torch, verbose=verbose)

    if use_torch:
        mu = torch.mean(acts, dim=0).cpu().numpy()
        sigma = torch_cov(acts, rowvar=False).cpu().numpy()
    else:
        mu = np.mean(acts, axis=0)
        sigma = np.cov(acts, rowvar=False)

    if os.path.dirname(output_path) != "":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)

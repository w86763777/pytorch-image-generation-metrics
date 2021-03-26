import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from PIL import Image

from .inception import InceptionV3
from .fid import calculate_frechet_distance, torch_cov


device = torch.device('cuda:0')


class PathDataset(Dataset):
    def __init__(self, dir_path, exts=['png', 'jpg']):
        self.paths = []
        for ext in exts:
            self.paths.extend(
                list(glob(os.path.join(dir_path, '*.%s' % ext))))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        image = to_tensor(image)
        return image


def get_inception_score_and_fid_from_directory(
        dir_path,
        fid_stats_path,
        num_images=None,
        splits=10,
        batch_size=50,
        use_torch=False,
        verbose=False):
    return get_inception_score_and_fid(
        images=DataLoader(PathDataset(dir_path), batch_size=batch_size),
        fid_stats_path=fid_stats_path,
        splits=splits,
        is_dataloader=True,
        use_torch=use_torch,
        verbose=verbose,
    )


def get_inception_score_and_fid(
        images,
        fid_stats_path,
        splits=10,
        batch_size=50,
        is_dataloader=False,
        use_torch=False,
        verbose=False):
    """Calculate Inception Score and FID.
    For each image, only a forward propagation is required to
    calculating features for FID and Inception Score.

    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
                must be float tensor of range [0, 1].
        fid_stats_path: str, Path to pre-calculated statistic
        splits: The number of bins of Inception Score. Default is 10.
        batch_size: int, The batch size for calculating activations. If
                    `images` is torch.utils.data.Dataloader, this arguments
                    does not work.
        use_torch: bool. The default value is False and the backend is same as
                   official implementation, i.e., numpy. If use_torch is
                   enableb, the backend linalg is implemented by torch, the
                   results are not guaranteed to be consistent with numpy, but
                   the speed can be accelerated by GPU.
        verbose: int. Set verbose to 0 for disabling progress bar. Otherwise,
                 the progress bar is showing when calculating activations.
    Returns:
        inception_score: float tuple, (mean, std)
        fid: float
    """
    if is_dataloader:
        assert isinstance(images, DataLoader)
        num_images = min(len(images.dataset), images.batch_size * len(images))
        batch_size = images.batch_size
    else:
        num_images = len(images)

    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx1, block_idx2]).to(device)
    model.eval()

    if use_torch:
        fid_acts = torch.empty((num_images, 2048)).to(device)
        is_probs = torch.empty((num_images, 1008)).to(device)
    else:
        fid_acts = np.empty((num_images, 2048))
        is_probs = np.empty((num_images, 1008))

    pbar = tqdm(
        total=num_images, dynamic_ncols=True, leave=False,
        disable=not verbose, desc="get_inception_score_and_fid")
    looper = iter(images)
    start = 0
    while start < num_images:
        # get a batch of images from iterator
        if is_dataloader:
            batch_images = next(looper)
        else:
            batch_images = images[start: start + batch_size]
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
            if use_torch:
                fid_acts[start: end] = pred[0].view(-1, 2048)
                is_probs[start: end] = pred[1]
            else:
                fid_acts[start: end] = pred[0].view(-1, 2048).cpu().numpy()
                is_probs[start: end] = pred[1].cpu().numpy()
        start = end
        pbar.update(len(batch_images))
    pbar.close()

    # Inception Score
    scores = []
    for i in range(splits):
        part = is_probs[
            (i * is_probs.shape[0] // splits):
            ((i + 1) * is_probs.shape[0] // splits), :]
        if use_torch:
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        is_score = (torch.mean(scores).cpu().item(),
                    torch.std(scores).cpu().item())
    else:
        is_score = (np.mean(scores), np.std(scores))

    # FID Score
    f = np.load(fid_stats_path)
    m2, s2 = f['mu'][:], f['sigma'][:]
    f.close()
    if use_torch:
        m1 = torch.mean(fid_acts, axis=0)
        s1 = torch_cov(fid_acts, rowvar=False)
        m2 = torch.tensor(m2).to(m1.dtype).to(device)
        s2 = torch.tensor(s2).to(s1.dtype).to(device)
    else:
        m1 = np.mean(fid_acts, axis=0)
        s1 = np.cov(fid_acts, rowvar=False)
    fid_score = calculate_frechet_distance(m1, s1, m2, s2, use_torch=use_torch)

    del fid_acts, is_probs, scores, model
    return is_score, fid_score

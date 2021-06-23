import argparse
import os
import glob

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from score.utils import (
    ImageDataset,
    get_inception_score_and_fid_from_directory,
    get_inception_score_and_fid,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Inception Score and FID")
    parser.add_argument('--path', type=str, required=True,
                        help='path to image directory')
    parser.add_argument('--stats', type=str, required=True,
                        help='precalculated reference statistics')
    args = parser.parse_args()

    # 1. Create custom pytorch DataLoader
    dataset = ImageDataset(args.path, exts=['png', 'jpg'])
    loader = DataLoader(dataset, batch_size=50, num_workers=4)
    (IS, IS_std), FID = get_inception_score_and_fid(
        loader, args.stats, verbose=True)
    print(IS, IS_std, FID)

    # 2. Calculate scores for images on disk which can avoid out of memory.
    (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
        args.path, args.stats, verbose=True)
    print(IS, IS_std, FID)

    # 3. Read or generate images
    files = (
        list(glob.glob(os.path.join(args.path, '*.png'))) +
        list(glob.glob(os.path.join(args.path, '*.jpg')))
    )
    images = torch.stack([to_tensor(Image.open(path)) for path in files])
    IS, FID = get_inception_score_and_fid(
        images, args.stats, verbose=True)
    print(IS, FID)

import argparse
import os
import glob

import torch
import numpy as np
from torchvision.transforms.functional import to_tensor
from PIL import Image

from score.score import get_inception_and_fid_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Calculate FID(CIFAR10) and inception score")
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--stats_cache', type=str,
                        default='./stats/cifar10_stats.npz')
    args = parser.parse_args()

    files = (
        list(glob.glob(os.path.join(args.dir, '*.png'))) +
        list(glob.glob(os.path.join(args.dir, '*.jpg')))
    )

    imgs = []
    for file_path in files:
        img = Image.open(file_path)
        img = to_tensor(img)
        imgs.append(img.numpy())
    imgs = np.array(imgs)

    is_score, fid_score = get_inception_and_fid_score(
        imgs, torch.device('cuda:0'), args.stats_cache, verbose=True)
    print(is_score, fid_score)

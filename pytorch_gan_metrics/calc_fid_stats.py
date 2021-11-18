import argparse
import os

import numpy as np
from torch.utils.data import DataLoader

from . import ImageDataset
from .core import get_inception_feature


def calc_and_save_stats(path, output, batch_size):
    dataset = ImageDataset(path, exts=['png', 'jpg'])
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    acts, = get_inception_feature(loader, dims=[2048], verbose=True)

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)

    if os.path.dirname(output) != "":
        os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez_compressed(output, mu=mu, sigma=sigma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Pre-calculate statistics of images")
    parser.add_argument("--path", type=str, required=True,
                        help='path to image directory')
    parser.add_argument("--output", type=str, required=True,
                        help="output path")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="batch size (default=50)")
    args = parser.parse_args()

    calc_and_save_stats(args.path, args.output, args.batch_size)

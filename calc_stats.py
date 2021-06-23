import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from score.core import get_inception_feature
from score.utils import ImageDataset


DIM = 2048
device = torch.device('cuda:0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Pre-calculate statistics of images")
    parser.add_argument("--path", type=str, required=True,
                        help='path to image directory')
    parser.add_argument("--output", type=str, required=True,
                        help="output path")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="batch size (default=50)")
    parser.add_argument("--inception_dir", type=str, default='/tmp',
                        help='path to inception model dir')
    args = parser.parse_args()

    dataset = ImageDataset(args.path, exts=['png', 'jpg'])
    loader = DataLoader(dataset, batch_size=50)
    acts, = get_inception_feature(loader, dims=[2048], verbose=True)

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)

    if os.path.dirname(args.output) != "":
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, mu=mu, sigma=sigma)

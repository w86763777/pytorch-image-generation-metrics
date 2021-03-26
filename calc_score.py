import argparse
import os
import glob

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image

from score.both import (
    get_inception_score_and_fid_from_directory,
    get_inception_score_and_fid
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Inception Score and FID")
    parser.add_argument('--path', type=str, required=True, help='image dir')
    parser.add_argument('--stats', type=str,
                        default='./stats/cifar10.test.npz',
                        help='reference stats for FID')
    parser.add_argument('--use_torch', action='store_true',
                        help='make torch be backend, or the numpy is used')
    args = parser.parse_args()

    # 1. from file
    # print("Fome file (save memory)")
    # IS, FID = get_inception_score_and_fid_from_directory(
    #     args.path, args.stats, use_torch=args.use_torch, verbose=True)
    # print(IS, FID)

    # 2. preload
    print("Preload images (efficient IO)")
    files = (
        list(glob.glob(os.path.join(args.path, '*.png'))) +
        list(glob.glob(os.path.join(args.path, '*.jpg')))
    )
    images = torch.stack([to_tensor(Image.open(path)) for path in files])
    IS, FID = get_inception_score_and_fid(
        images, args.stats, use_torch=args.use_torch, verbose=True)
    print(IS, FID)

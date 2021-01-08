import argparse
import os
import glob

import numpy as np
from torchvision.transforms.functional import to_tensor
from PIL import Image

from score.both import get_inception_and_fid_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Calculate FID(CIFAR10) and Inception Score")
    parser.add_argument('--dir', type=str, required=True, help='image dir')
    parser.add_argument('--stats', type=str,
                        default='./stats/cifar10_test.npz',
                        help='reference stats for FID')
    parser.add_argument('--use_torch', action='store_true',
                        help='make torch be backend, or the numpy is used')
    args = parser.parse_args()

    files = (
        list(glob.glob(os.path.join(args.dir, '*.png'))) +
        list(glob.glob(os.path.join(args.dir, '*.jpg')))
    )

    # Support: Load every images before calculating IS and FID
    imgs = []
    for file_path in files:
        img = Image.open(file_path)
        img = to_tensor(img)
        imgs.append(img.numpy())
    imgs = np.array(imgs)

    IS, FID = get_inception_and_fid_score(
        imgs, args.stats, use_torch=args.use_torch, verbose=True)
    print(IS, FID)

    # Support: Load images on demand
    def images_generator(files):
        for file_path in files:
            img = Image.open(file_path)
            img = to_tensor(img).numpy()
            yield img

    IS, FID = get_inception_and_fid_score(
        images_generator(files), args.stats, num_images=len(files),
        use_torch=args.use_torch, verbose=True, parallel=False)
    print(IS, FID)

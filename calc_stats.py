import argparse
import os

import numpy as np
import torch
from torchvision import datasets, transforms

from score.inception import InceptionV3
from score.fid import get_statistics


DIM = 2048
device = torch.device('cuda:0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate states of CIFAR10/STL10")
    parser.add_argument("--output", type=str, default='./cifar10_test.npz',
                        help="stats output path (default=./cifar10_test.npz)")
    parser.add_argument("--inception_dir", type=str, default='./data',
                        help='path to inception model dir (default=./data)')
    parser.add_argument("--dataset", type=str, default='cifar10',
                        choices=['cifar10', 'stl10'],
                        help='dataset (default=cifar10)')
    parser.add_argument("--batch_size", type=int, default=50,
                        help="batch size (default=50)")
    parser.add_argument('--use_torch', action='store_true',
                        help='make torch be backend, or the numpy is used '
                             '(default=False)')
    args = parser.parse_args()

    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(
            './data', train=False, download=True,
            transform=transforms.ToTensor())
    elif args.dataset == "stl10":
        dataset = datasets.STL10(
            './data', split='unlabeled', download=True,
            transform=transforms.Compose([
                transforms.Resize((48, 48)), transforms.ToTensor()]))
    else:
        # implement your dataset here
        raise NotImplementedError("Dataset %s is not supported" % args.dataset)

    def image_generator(dataset):
        for x, _ in dataset:
            yield x.numpy()

    m, s = get_statistics(
        image_generator(dataset), num_images=len(dataset), batch_size=50,
        use_torch=args.use_torch, verbose=True, parallel=False)

    if args.use_torch:
        m = m.cpu().numpy()
        s = s.cpu().numpy()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, mu=m, sigma=s)

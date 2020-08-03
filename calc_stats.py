import argparse
import os

import numpy as np
import torch
from torchvision import datasets, transforms

from score.inception import InceptionV3
from score.fid_score import get_statistics


DIM = 2048


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate states of CIFAR10/STL10")
    parser.add_argument("--inception_dir", type=str, default='./data',
                        help='path to inception model dir')
    parser.add_argument("--dataset", type=str, default='cifar10',
                        choices=['cifar10', 'stl10'],
                        help='dataset (default=cifar10)')
    parser.add_argument("--stats_path", type=str,
                        help="stats output path")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="batch size (default=50)")
    args = parser.parse_args()

    device = torch.device('cuda:0')

    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.ToTensor())
    else:
        dataset = datasets.STL10(
            './data', split='unlabeled', download=True,
            transform=transforms.Compose([
                transforms.Resize((48, 48)), transforms.ToTensor()]))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[DIM]
    model = InceptionV3([block_idx]).to(device)

    images = []
    for batch_images, _ in dataloader:
        images.append(batch_images.cpu().numpy())
    images = np.concatenate(images, axis=0)

    m, s = get_statistics(
        images, model, device, args.batch_size, verbose=True)
    os.makedirs(os.path.dirname(args.stats_path), exist_ok=True)
    np.savez_compressed(args.stats_path, mu=m, sigma=s)

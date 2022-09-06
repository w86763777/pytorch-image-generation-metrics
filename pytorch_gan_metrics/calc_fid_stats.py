import argparse
import os

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose, Resize, ToTensor

from . import ImageDataset
from .core import get_inception_feature


def calc_and_save_stats(
    input_path,
    output_path,
    batch_size=50,
    img_size=None,
    num_workers=os.cpu_count(),
    verbose=True,
):
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
    acts, = get_inception_feature(loader, dims=[2048], verbose=verbose)

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)

    if os.path.dirname(output_path) != "":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "A handy cli tool for function calc_and_save_stats.")
    parser.add_argument("--path", type=str, required=True,
                        help='path to image directory')
    parser.add_argument("--output", type=str, required=True,
                        help="output path")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="batch size (default=50)")
    parser.add_argument("--img_size", type=int, default=None,
                        help="resize image")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="dataloader workers")

    args = parser.parse_args()

    calc_and_save_stats(args.path,
                        args.output,
                        args.batch_size,
                        args.img_size,
                        args.num_workers)

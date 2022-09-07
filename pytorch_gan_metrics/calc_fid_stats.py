"""Calculate statistics for FID and save to a file."""

import argparse
import os

from .utils import calc_and_save_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "A handy cli tool to use function "
        "pytorch_gan_metrics.calc_fid_stats.calc_and_save_stats.")
    parser.add_argument("--path", type=str, required=True,
                        help='path to image directory')
    parser.add_argument("--output", type=str, required=True,
                        help="output path")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="batch size (default=50)")
    parser.add_argument("--img_size", type=int, default=None,
                        help="resize image")
    parser.add_argument('--use_torch', action='store_true',
                        help='using pytorch as the matrix operations backend')
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="dataloader workers")

    args = parser.parse_args()

    calc_and_save_stats(
        args.path,
        args.output,
        args.batch_size,
        args.img_size,
        args.use_torch,
        args.num_workers)

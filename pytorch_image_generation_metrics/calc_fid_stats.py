"""Calculate statistics for FID and save them to a file."""

import argparse
import os
import tempfile

import torch

from .districuted import init
from .utils import calc_and_save_stats


def calc(args):
    calc_and_save_stats(
        args.path,
        args.output,
        args.batch_size,
        args.img_size,
        args.use_torch,
        args.num_workers)


def calc_init(init_method, world_size, rank, args):
    init(init_method, world_size, rank)
    calc(args)


def main():
    parser = argparse.ArgumentParser(
        description="A command-line tool to compute Frechet Inception Distance (FID) statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", type=str, required=True,
                        help='Path to the directory containing images (including subfolders).')
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path for saving the computed statistics.")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for processing images.")
    parser.add_argument("--img_size", type=int, default=None,
                        help="Resize images to this specified size (if provided).")
    parser.add_argument('--use_torch', action='store_true',
                        help='Use PyTorch for matrix operations.')
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Number of worker processes for data loading.")
    args = parser.parse_args()

    world_size = len(os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(','))
    if world_size == 1:
        calc(args)
    else:
        with tempfile.TemporaryDirectory() as temp:
            init_method = f'file://{os.path.abspath(os.path.join(temp, ".ddp"))}'
            processes = []
            for rank in range(world_size):
                p = torch.multiprocessing.Process(
                    target=calc_init,
                    args=(init_method, world_size, rank, args))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()


if __name__ == '__main__':
    main()

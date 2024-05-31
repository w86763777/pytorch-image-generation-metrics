"""Calculate the FID and Inception Score of images in a directory."""

import argparse
import os
import tempfile

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .districuted import init, world_size, print0
from .utils import ImageDataset, get_inception_score_and_fid


def calc(args):
    """Calculate the FID and Inception Score of images in a directory."""
    dataset = ImageDataset(root=args.path, num_images=args.num_images)
    if world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers)
    (IS, IS_std), FID = get_inception_score_and_fid(
        loader,
        args.fid_ref,
        use_torch=args.use_torch,
        verbose=True)
    print0(IS, IS_std, FID)


def calc_init(init_method, world_size, rank, args):
    """Initialize the distributed environment and calculate the FID and Inception Score of images in a directory."""
    init(init_method, world_size, rank)
    calc(args)


def main():
    """Parse command-line arguments and calculate the FID and Inception Score of images in a directory."""
    parser = argparse.ArgumentParser(
        description="A command-line tool to calculate Frechet Inception Distance (FID) between generated and reference images.",
        epilog="Example: CUDA_VISIBLE_DEVICES=0,1 python -m pytorch_image_generation_metrics.calc_metrics --path cifar10/train --fid_ref cifar10.test.npz --batch 64",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the directory containing generated images.')
    parser.add_argument('--fid_ref', type=str, required=True,
                        help='Path to precalculated reference statistics file.')
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for processing images.")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Number of images to use for calculating FID. If not specified, all images in the directory will be used.")
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


if __name__ == "__main__":
    main()

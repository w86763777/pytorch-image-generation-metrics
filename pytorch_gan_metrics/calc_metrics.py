import argparse

from torch.utils.data import DataLoader

from . import ImageDataset, get_inception_score_and_fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Inception Score and FID")
    parser.add_argument('--path', type=str, required=True,
                        help='path to image directory')
    parser.add_argument('--stats', type=str, required=True,
                        help='precalculated reference statistics')
    parser.add_argument('--use_torch', action='store_true',
                        help='using pytorch as the matrix operations backend')
    args = parser.parse_args()

    dataset = ImageDataset(args.path, exts=['png', 'jpg'])
    loader = DataLoader(dataset, batch_size=50, num_workers=4)
    (IS, IS_std), FID = get_inception_score_and_fid(
        loader, args.stats, use_torch=args.use_torch, verbose=True)
    print(IS, IS_std, FID)

import version

from pytorch_gan_metrics.utils import (
    ImageDataset,
    get_inception_score,
    get_inception_score_from_directory,
    get_fid,
    get_fid_from_directory,
    get_inception_score_and_fid,
    get_inception_score_and_fid_from_directory)

__version__ = open('version.py').read().split('=')[1].strip().strip("'")

__all__ = [
    ImageDataset,
    get_inception_score,
    get_inception_score_from_directory,
    get_fid,
    get_fid_from_directory,
    get_inception_score_and_fid,
    get_inception_score_and_fid_from_directory,
]

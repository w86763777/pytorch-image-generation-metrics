# Pytorch Implementation of Common Image Generation Metrics

![PyPI](https://img.shields.io/pypi/v/pytorch_image_generation_metrics)

## Installation
```
pip install pytorch-image-generation-metrics
```

## Quick Start
```python
from pytorch_image_generation_metrics import get_inception_score, get_fid

images = ... # [N, 3, H, W] normalized to [0, 1]
IS, IS_std = get_inception_score(images)        # Inception Score
FID = get_fid(images, 'path/to/fid_ref.npz') # Frechet Inception Distance
```
The file `path/to/fid_ref.npz` is compatiable with the [official FID implementation](https://github.com/bioinf-jku/TTUR).

## Notes
The FID implementation is inspired by [pytorch-fid](https://github.com/mseitzer/pytorch-fid).

This repository is developed for personal research. If you find this package useful, please feel free to open issues.

## Features
- Currently, this package supports the following metrics:
  - [Inception Score](https://github.com/openai/improved-gan) (IS)
  - [Fréchet Inception Distance](https://github.com/bioinf-jku/TTUR) (FID)
- The computation procedures for IS and FID are integrated to avoid multiple forward passes.
- Supports reading images on the fly to prevent out-of-memory issues, especially for large-scale images.
- Supports computation on GPU to speed up some CPU operations, such as `np.cov` and `scipy.linalg.sqrtm`.

## Reproducing Results of Official Implementations on CIFAR-10

|                     |Train IS  |Test IS   |Train(50k) vs Test(10k)<br>FID|
|---------------------|:--------:|:--------:|:----------------------------:|
|Official             |11.24±0.20|10.98±0.22|3.1508                        |
|ours                 |11.26±0.13|10.97±0.19|3.1525                        |
|ours `use_torch=True`|11.26±0.15|10.97±0.20|3.1457                        |

The results differ slightly from the official implementations due to the framework differences between PyTorch and TensorFlow.

## Documentation

### Prepare Statistical Reference for FID
- [Download](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing) the pre-calculated reference, or
- Calculate the statistical reference for your custom dataset using the command-line tool:
    ```bash
    python -m pytorch_image_generation_metrics.fid_ref \
        --path path/to/images \
        --output path/to/fid_ref.npz
    ```
    See [fid_ref.py](./pytorch_image_generation_metrics/fid_ref.py) for details.

### Inception Features
- When getting IS or FID, the `InceptionV3` model will be loaded into `torch.device('cuda:0')` by default.
- Change the `device` argument in the `get_*` functions to set the torch device.

### Using `torch.Tensor` as images

- Prepare images as `torch.float32` tensors with shape `[N, 3, H, W]`, normalized to `[0,1]`.
    ```python
    from pytorch_image_generation_metrics import (
        get_inception_score,
        get_fid,
        get_inception_score_and_fid
    )

    images = ... # [N, 3, H, W]
    assert 0 <= images.min() and images.max() <= 1

    # Inception Score
    IS, IS_std = get_inception_score(
        images)

    # Frechet Inception Distance
    FID = get_fid(
        images, 'path/to/fid_ref.npz')

    # Inception Score & Frechet Inception Distance
    (IS, IS_std), FID = get_inception_score_and_fid(
        images, 'path/to/fid_ref.npz')

    ```

### Using PyTorch DataLoader to Provide Images

1. Use `pytorch_image_generation_metrics.ImageDataset` to collect images from your storage or use your custom `torch.utils.data.Dataset`.
    ```python
    from pytorch_image_generation_metrics import ImageDataset
    from torch.utils.data import DataLoader

    dataset = ImageDataset(path_to_dir, exts=['png', 'jpg'])
    loader = DataLoader(dataset, batch_size=50, num_workers=4)
    ```

    You can wrap a generative model in a dataset to support generating images on the fly.
    ```python
    class GeneratorDataset(Dataset):
        def __init__(self, G, noise_dim):
            self.G = G
            self.noise_dim = noise_dim

        def __len__(self):
            return 50000

        def __getitem__(self, index):
            return self.G(torch.randn(1, self.noise_dim))

    dataset = GeneratorDataset(G, noise_dim=128)
    loader = DataLoader(dataset, batch_size=50, num_workers=0)
    ```

2. Calculate metrics
    ```python
    from pytorch_image_generation_metrics import (
        get_inception_score,
        get_fid,
        get_inception_score_and_fid
    )

    # Inception Score
    IS, IS_std = get_inception_score(
        loader)

    # Frechet Inception Distance
    FID = get_fid(
        loader, 'path/to/fid_ref.npz')

    # Inception Score & Frechet Inception Distance
    (IS, IS_std), FID = get_inception_score_and_fid(
        loader, 'path/to/fid_ref.npz')
    ```

### Load Images from a Directory

- Calculate metrics for images in a directory and its subfolders.
    ```python
    from pytorch_image_generation_metrics import (
        get_inception_score_from_directory,
        get_fid_from_directory,
        get_inception_score_and_fid_from_directory)

    IS, IS_std = get_inception_score_from_directory(
        'path/to/images')
    FID = get_fid_from_directory(
        'path/to/images', 'path/to/fid_ref.npz')
    (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
        'path/to/images', 'path/to/fid_ref.npz')
    ```

### Accelerating Matrix Computation with PyTorch

- Set `use_torch=True` when calling functions like `get_inception_score`, `get_fid`, etc.

- **WARNING**: when `use_torch=True` is used, the FID might be `nan` due to the unstable implementation of matrix sqrt root.

## Tested Versions
- `python 3.9 + torch 1.13.1 + CUDA 11.7`
- `python 3.9 + torch 2.3.0 + CUDA 12.1`

## License

This implementation is licensed under the Apache License 2.0.

This implementation is derived from [pytorch-fid](https://github.com/mseitzer/pytorch-fid), licensed under the Apache License 2.0.

FID was introduced by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler and Sepp Hochreiter in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", see [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

The original implementation of FID is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0.
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).

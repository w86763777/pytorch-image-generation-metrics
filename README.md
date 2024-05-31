
### *This project has been renamed to pytorch-image-generation-metrics. Please visit the new repository [here](https://github.com/w86763777/pytorch-image-generation-metrics).*

# Pytorch Implementation of Common GAN metrics

![PyPI](https://img.shields.io/pypi/v/pytorch-gan-metrics)

## Install
```
pip install pytorch-gan-metrics
```
- `torch>=1.8.2`
- `torchvision>=0.9.2`

## Quick Start
```python
from pytorch_gan_metrics import get_inception_score, get_fid

images = ... # [N, 3, H, W] normalized to [0, 1]
IS, IS_std = get_inception_score(images)        # Inception Score
FID = get_fid(images, 'path/to/statistics.npz') # Frechet Inception Distance
```
`path/to/statistics.npz` is compatiable with [official](https://github.com/bioinf-jku/TTUR) FID implementation.

## Notes
The FID implementation is inspired from [pytorch-fid](https://github.com/mseitzer/pytorch-fid).

This repository is developed for personal research. If you think this package can also benefit your life, please feel free to open issues.

## Features
- Currently, this package supports following metrics:
  - [Inception Score](https://github.com/openai/improved-gan) (IS)
  - [Fréchet Inception Distance](https://github.com/bioinf-jku/TTUR) (FID)
- The computation procedure of IS and FID are integrated to avoid multiple forward propagations.
- Support reading images on the fly to avoid out of memory especially for large scale images.
- Support computation on GPU to speed up some cpu operations such as `np.cov` and `scipy.linalg.sqrtm`.

## Reproducing Results of Official Implementations on CIFAR-10

|                   |Train IS  |Test IS   |Train(50k) vs Test(10k)<br>FID|
|-------------------|:--------:|:--------:|:----------------------------:|
|Official           |11.24±0.20|10.98±0.22|3.1508                        |
|pytorch-gan-metrics|11.26±0.13|10.96±0.35|3.1518                        |
|pytorch-gan-metrics<br>`use_torch=True`<br>`torch==2.0.1`|11.26±0.15|10.95±0.16|3.1309                        |

The results are slightly different from official implementations due to the framework difference between PyTorch and TensorFlow.

## Documentation

### Prepare Statistics (for FID)
- [Download](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing) precalculated statistics or
- Calculate statistics for your custom dataset using command line tool
    ```bash
    python -m pytorch_gan_metrics.calc_fid_stats \
        --path path/to/images \
        --stats path/to/statistics.npz
    ```
    See [calc_fid_stats.py](./pytorch_gan_metrics/calc_fid_stats.py) for details.

### Inception Features
- When getting IS or FID, the `InceptionV3` will be loaded into `torch.device('cuda:0')` if GPU is availabel; Otherwise, `torch.device('cpu')` will be used.
- Change `device` argument in `get_*` functions to set torch device.

### Using `torch.Tensor` as images

- Prepare images in type `torch.float32` with shape `[N, 3, H, W]` and normalized to `[0,1]`.
    ```python
    from pytorch_gan_metrics import (get_inception_score,
                                     get_fid,
                                     get_inception_score_and_fid)
    images = ... # [N, 3, H, W]
    assert 0 <= images.min() and images.max() <= 1
    # Inception Score
    IS, IS_std = get_inception_score(
        images)
    # Frechet Inception Distance
    FID = get_fid(
        images, 'path/to/statistics.npz')
    # Inception Score & Frechet Inception Distance
    (IS, IS_std), FID = get_inception_score_and_fid(
        images, 'path/to/statistics.npz')

    ```

### Using PyTorch DataLoader to Provide Images

- Use `pytorch_gan_metrics.ImageDataset` to collect images on your storage or use your custom `torch.utils.data.Dataset`.
    ```python
    from pytorch_gan_metrics import ImageDataset

    dataset = ImageDataset(path_to_dir, exts=['png', 'jpg'])
    loader = DataLoader(dataset, batch_size=50, num_workers=4)
    ```

- It is possible to wrap a generative model in a dataset to support generating images on the fly. Remember to set `num_workers=0` to avoid copying models across multiprocess.
    ```python
    class GeneratorDataset(Dataset):
        def __init__(self, G, z_dim):
            self.G = G
            self.z_dim = z_dim

        def __len__(self):
            return 50000

        def __getitem__(self, index):
            return self.G(torch.randn(1, self.z_dim).cuda())[0]

    dataset = GeneratorDataset(G, z=128)
    loader = DataLoader(dataset, batch_size=50, num_workers=0)
    ```

- Calculate metrics
    ```python
    from pytorch_gan_metrics import (get_inception_score,
                                     get_fid,
                                     get_inception_score_and_fid)
    # Inception Score
    IS, IS_std = get_inception_score(
        loader)
    # Frechet Inception Distance
    FID = get_fid(
        loader, 'path/to/statistics.npz')
    # Inception Score & Frechet Inception Distance
    (IS, IS_std), FID = get_inception_score_and_fid(
        loader, 'path/to/statistics.npz')
    ```

### Load Images from a Directory

- Calculate metrics for images in a directory and its subfolders.
    ```python
    from pytorch_gan_metrics import (
        get_inception_score_from_directory,
        get_fid_from_directory,
        get_inception_score_and_fid_from_directory)

    IS, IS_std = get_inception_score_from_directory(
        'path/to/images')
    FID = get_fid_from_directory(
        'path/to/images', 'path/to/statistics.npz')
    (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
        'path/to/images', 'path/to/statistics.npz')
    ```

### Accelerating Matrix Computation by PyTorch

- Set `use_torch=True` when calling functions `get_*` such as `get_inception_score`, `get_fid`, etc.

- **WARNING** when `use_torch=True` is used, the FID might be `nan` due to the unstable implementation of matrix sqrt.

- This option is recommended to be used when evaluating generative models on a server which is equipped with high efficiency GPUs while the cpu frequency is low.

## Tested Versions
- `python 3.9 + torch 1.8.2 + CUDA 10.2`
- `python 3.9 + torch 1.11.0 + CUDA 10.2`
- `python 3.9 + torch 1.12.1 + CUDA 10.2`

## License

This implementation is licensed under the Apache License 2.0.

This implementation is derived from [pytorch-fid](https://github.com/mseitzer/pytorch-fid), licensed under the Apache License 2.0.

FID was introduced by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler and Sepp Hochreiter in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", see [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

The original implementation of FID is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0.
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).

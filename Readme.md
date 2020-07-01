# Pytorch Implementation of Inception Score and FID Score

|                            |Original Implementation|This implementation|
|----------------------------|-----------------------|-------------------|
|Inception Score (SNGAN)     |8.1700 (0.1143)        |8.1858 (0.1140)    |
|Inception Score (CIFAR10)   |11.2671 (0.2032)       |11.2391 (0.2005)   |
|FID Score (SNGAN on CIFAR10)|14.5675                |14.4361            |

- Original implementation: [Inception Score](https://github.com/openai/improved-gan), [FID Score](https://github.com/bioinf-jku/TTUR)

- The original implementation of Inception Score ignores the bias of fully connect layer in inception v3, that is the most important detail to reimplement it

## Requirements
- torch 1.4.0
- torchvision 0.5.0
- tqdm
- scipy 1.5.0
- Install requirements
    ```
    pip install -r requirements.txt
    ```

## Example
- Prepare Statistics for calculating FID Score
    - [Download](https://github.com/bioinf-jku/TTUR#precalculated-statistics-for-fid-calculation) Precalculated Statistics for your dataset or
    - Calculate statistics for your dataset. See [example](./calc_stats.py)
        ```
        # Calculate statistics of CIFAR10
        # Save stats to ./stats/cifar10_stats.npz
        python calc_stats.py --stats_path ./stats/cifar10_stats.npz
        ```

- Calculate Inception Score and FID Score at a time. Both score share same
Inception v3 outputs so only one forward propagation is needed for each image
    ```
    python calc_score.py \
        --stats_cache ./stats/cifar10_stats.npz \
        --dir path/to/images
    ```

## Integrate into training scripts
```python
import torch
from score.fid_score import get_fid_score
from score.inception_score import get_inception_score
from score.score import get_inception_and_fid_score

imgs = np.array(...)    # Channel first, Normalized to [0, 1]
                        # e.g. shape = [N, 3, 32, 32]
                        # the image size will be resize to [299, 299] to match
                        # Inception V3 input size
device = torch.device('cuda:0')
stats_cache = "./stats/cifar10_stats.npz"

# Calculate Inception Score only
is_score = get_inception_score(
    imgs, device, verbose=True)

# Calculate FID Score only
fid_score = get_fid_score(
    imgs, stats_cache, device, verbose=False)

# Calculate Inception Score and FID Score at a time
is_score, fid_score = get_inception_and_fid_score(
    imgs, device, stats_cache, verbose=True)
```

## TODO

- [ ] Dynamic loading images
- [ ] Multi-GPU computing
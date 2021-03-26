# Pytorch Implementation of Inception Score and FID Score

The key feature of this repo is speeding up Inception Score and FID computation by integrating them in a forward propogation.

## Reproduce Results of Official Implementations
- Official implementation: [Inception Score](https://github.com/openai/improved-gan), [Fréchet Inception Distance](https://github.com/bioinf-jku/TTUR)

- Incpetion Score on CIFAR10
    |      |Official  |This implementation|
    |------|:--------:|:-----------------:|
    |Train |11.24±0.20|11.26±0.20         |
    |Test  |10.98±0.22|10.97±0.33         |

- FID on CIFAR10 Test Set (10k)
    |                       |Official FID|This implementation|
    |-----------------------|:----------:|:-----------------:|
    |CIFAR10 Train Set (50k)|3.15        |3.15               |

- FID on CIFAR10 Train Set (50k)
    |                       |Official FID|This implementation|
    |-----------------------|:----------:|:-----------------:|
    |CIFAR10 Test Set (10k) |3.15        |3.15               |

## Notes

- Our FID implementation is inspired from [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

- The official implementation of Inception Score ignores the bias term of last layer in inception v3, this is the most important detail when reimplement Inception Score.

- Due to the difference of framework implementations, both scores are slightly different from official implementations.

## Requirements
- torch 1.7.1
- torchvision 0.8.2
- tqdm 4.55.1
- scipy 1.5.4
- Install requirements
    ```
    pip install -r requirements.txt
    ```

## Example
- Prepare Statistics for calculating FID Score
    - [Download](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing) Precalculated Statistics for your dataset or
    - Calculate statistics for your dataset. See [example](./calc_stats.py)

- Calculate Inception Score and FID Score at a time. Both score share same Inception v3 outputs so only one forward propagation is needed for each image. See [example](./calc_stats.py) for more details.
    ```
    python calc_score.py \
        --path ./cifar10/train \
        --stats ./cifar10.test.npz
    ```

## Integrate into training scripts
```python
from score.both import get_inception_score_and_fid

images = ...    # torch float tensor of shape [num_images, 3, height, width]
                # the value range is [0, 1]
IS, FID = get_inception_score_and_fid(images, ./cifar10.test.npz, verbose=True)
print(IS, FID)
```

## TODO

- [x] Dynamic loading images
- [ ] Multi-GPU computing
"""Install the package."""

import os
import setuptools


def read(rel_path):
    """Read a file."""
    base_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_path, rel_path), 'r') as f:
        return f.read()


if __name__ == '__main__':
    # get __version__
    with open('./pytorch_image_generation_metrics/version.py') as f:
        exec(f.read())

    setuptools.setup(
        name='pytorch_image_generation_metrics',
        version=__version__,    # noqa: F821
        author='Yi-Lun Wu',
        author_email='w86763777@gmail.com',
        description=('Package for calculating image generation metrics using Pytorch'),
        long_description=read('README.md'),
        long_description_content_type='text/markdown',
        url='https://github.com/w86763777/pytorch_image_generation_metrics',
        packages=setuptools.find_packages(include=['pytorch_image_generation_metrics']),
        keywords=[
            'PyTorch',
            'Inception Score',
            'IS',
            'Frechet Inception Distance',
            'FID'],
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.6',
        install_requires=[
            "packaging",
            "tqdm",
            "scipy",
            "torch>=1.8.2",
            "torchvision>=0.9.2",
        ],
    )

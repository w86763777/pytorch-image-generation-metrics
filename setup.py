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
    with open('./pytorch_gan_metrics/version.py') as f:
        exec(f.read())

    setuptools.setup(
        name='pytorch_gan_metrics',
        version=__version__,    # noqa: F821
        author='Yi-Lun Wu',
        author_email='w86763777@gmail.com',
        description=('Package for calculating GAN metrics using Pytorch'),
        long_description=read('README.md'),
        long_description_content_type='text/markdown',
        url='https://github.com/w86763777/pytorch-image-generation-metrics',
        keywords=[
            'PyTorch',
            'GAN',
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
            "pytorch-image-generation-metrics",
        ],
    )

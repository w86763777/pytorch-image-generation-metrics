import os
import setuptools


def read(rel_path):
    base_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_path, rel_path), 'r') as f:
        return f.read()


if __name__ == '__main__':
    setuptools.setup(
        name='pytorch_gan_metrics',
        version='0.4.0',
        author='Yi-Lun Wu',
        author_email='w86763777@gmail.com',
        description=('Package for calculating GAN metrics using Pytorch'),
        long_description=read('README.md'),
        long_description_content_type='text/markdown',
        url='https://github.com/w86763777/pytorch-gan-metrics',
        packages=setuptools.find_packages(include=['pytorch_gan_metrics']),
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
            'tqdm',
            'scipy==1.5.4',
            'torch>=1.8.0',
            'torchvision>=0.9.0',
        ],
    )

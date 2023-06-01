from distutils.core import setup

from setuptools import find_packages

setup(
    name="nr4seg",
    version="0.0.1",
    author="Zhizheng Liu, Francesco Milano",
    author_email="liuzhi@student.ethz.ch, francesco.milano@mavt.ethz.ch",
    packages=find_packages(),
    python_requires=">=3.6",
    description=
    "[CVPR 2023] Unsupervised Continual Semantic Adaptation through Neural Rendering",
)

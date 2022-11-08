# coding=utf-8

from setuptools import setup, find_packages

setup(
    name="tf_geometric",
    python_requires='>3.5.0',
    version="0.1.4",
    author="Jun Hu",
    author_email="hujunxianligong@gmail.com",
    packages=find_packages(
        exclude=[
            'benchmarks',
            'data',
            'demo',
            'dist',
            'doc',
            'docs',
            'logs',
            'models',
            'test'
        ]
    ),
    install_requires=[
        "tf_sparse >= 0.0.17",
        "numpy >= 1.17.4",
        "networkx >= 2.1",
        "scipy >= 1.1.0",
        "scikit-learn >= 0.22",
        "ogb_lite >= 0.0.3",
        "tqdm"
    ],
    extras_require={
        'tf1-cpu': ["tensorflow >= 1.15.0,<2.0.0"],
        'tf1-gpu': ["tensorflow-gpu >= 1.15.0,<2.0.0"],
        'tf2-cpu': ["tensorflow >= 2.4.0"],
        'tf2-gpu': ["tensorflow >= 2.4.0"]
    },
    description="Efficient and Friendly Graph Neural Network Library for TensorFlow 1.x and 2.x.",
    license="GNU General Public License v3.0 (See LICENSE)",
    long_description=open("README.rst", "r", encoding="utf-8").read(),
    url="https://github.com/CrawlScript/tf_geometric"
)
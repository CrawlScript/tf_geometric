from setuptools import setup, find_packages

setup(
    name="tf_geometric",
    python_requires='>3.5.0',
    version="0.0.22",
    author="Jun Hu",
    author_email="hujunxianligong@gmail.com",
    packages=find_packages(
        exclude=[
            'data',
            'demo',
            'doc',
            'docs'
        ]
    ),
    install_requires=[
        "numpy >= 1.17.4",
        "networkx >= 2.1",
        "scipy >= 1.1.0",
        "scikit-learn >= 0.22",
        "tqdm"
    ],
    extras_require={
        'tf1-cpu': ["tensorflow >= 1.14.0,<2.0.0"],
        'tf1-gpu': ["tensorflow-gpu >= 1.14.0,<2.0.0"],
        'tf2-cpu': ["tensorflow >= 2.0.0b1"],
        'tf2-gpu': ["tensorflow-gpu >= 2.0.0b1"]
    },
    description="""
        Efficient and Friendly Graph Neural Network Library for TensorFlow 1.x and 2.x.
    """,
    license="GNU General Public License v3.0 (See LICENSE)",
    long_description=open("README.rst", "r", encoding="utf-8").read(),
    url="https://github.com/CrawlScript/tf_geometric"
)
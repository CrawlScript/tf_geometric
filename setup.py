from setuptools import setup, find_packages

setup(
    name="tf_geometric",
    python_requires='>3.5.0',
    version="0.0.3",
    author="Jun Hu",
    author_email="hujunxianligong@gmail.com",
    packages=find_packages(
        exclude=[
            'data',
        ]
    ),
    install_requires=[
        "tensorflow-gpu >= 1.14.0",
        "numpy >= 1.17.4",
        "networkx >= 2.1"

    ],
    description="""
        Efficient and Friendly Graph Neural Network Library for TensorFlow 1.x and 2.x.
    """,
    license="GNU General Public License v3.0 (See LICENSE)",
    long_description=open("README.rst", "r", encoding="utf-8").read(),
    url="https://github.com/CrawlScript/tf_geometric"
)
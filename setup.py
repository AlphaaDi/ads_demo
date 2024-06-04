import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="unipop",
    version="0.0.1",
    author="Dzmitry Kamarouski & Ivan Shpuntov",
    description=("Unipop Core Alg"),
    keywords="",
    url="",
    packages=find_packages(),
    setup_requires=["numpy>1.15"],
    zip_safe=False,
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
    ],
)
#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="jointmoments",
    version="0.1",
    description="Tensors and statistics for joint central moments",
    author="Jack Peterson",
    author_email="<jack@tinybike.net>",
    maintainer="Jack Peterson",
    maintainer_email="<jack@tinybike.net>",
    license="MIT",
    url="https://github.com/tensorjack/jointmoments",
    download_url = "https://github.com/tensorjack/jointmoments/tarball/0.1",
    packages=["jointmoments"],
    install_requires=["numpy", "pandas", "six"],
    keywords = ["moments", "joint", "mixed", "central", "centered", "cumulants", "tensors"]
)

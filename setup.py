#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="mowgli",
    version="0.0.1",
    description="Mowgli: Multi Omics Wasserstein inteGrative anaLysIs.",
    author="Geert-Jan Huizing",
    author_email="huizing@ens.fr",
    packages=["mowgli"],
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "sklearn",
        "muon",
        "tqdm",
        "scanpy",
        "anndata",
        "matplotlib",
        "plotly",
        "scipy",
        "scikit-network",
    ],
)

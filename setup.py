#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="mowgli",
    version="0.1.0",
    description="Mowgli: Multi Omics Wasserstein inteGrative anaLysIs.",
    author="Geert-Jan Huizing",
    author_email="huizing@ens.fr",
    packages=["mowgli"],
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "muon",
        "tqdm",
        "scanpy",
        "anndata",
        "matplotlib",
        "scipy",
        "gprofiler-official",
        "leidenalg",
        "nbsphinx",
        "furo",
    ],
)

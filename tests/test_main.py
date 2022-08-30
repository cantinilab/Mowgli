from array import ArrayType
from context import models
import muon as mu
import anndata as ad
import numpy as np


def test_init():
    n_cells = 20
    n_genes = 50
    n_peaks = 5
    latent_dim = 5

    # Create a random anndata object for RNA.
    rna = ad.AnnData(np.random.rand(n_cells, n_genes))
    rna.var["highly_variable"] = True

    # Create a random anndata object for ATAC.
    atac = ad.AnnData(np.random.rand(n_cells, n_peaks))
    atac.var["highly_variable"] = True

    # Create a MuData object combining RNA and ATAC.
    mdata = mu.MuData({"rna": rna, "atac": atac})

    # Initialize the Mowgli model.
    model = models.MowgliModel(latent_dim=latent_dim)

    # Train the model.
    model.train(mdata)

    # Check the size of the embedding.
    assert mdata.obsm["W_OT"].shape == (n_cells, latent_dim)

    # Check the size of the dictionaries.
    assert mdata["rna"].uns["H_OT"].shape == (latent_dim, n_genes)
    assert mdata["atac"].uns["H_OT"].shape == (latent_dim, n_peaks)

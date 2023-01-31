from context import models
import muon as mu
import anndata as ad
import torch
import numpy as np

n_cells, n_genes, n_peaks = 20, 50, 5
latent_dim = 5

# Create a random anndata object for RNA.
rna = ad.AnnData(np.random.rand(n_cells, n_genes))
rna.var["highly_variable"] = True

# Create a random anndata object for ATAC.
atac = ad.AnnData(np.random.rand(n_cells, n_peaks))
atac.var["highly_variable"] = True

# Create a MuData object combining RNA and ATAC.
mdata = mu.MuData({"rna": rna, "atac": atac})

mdata.obs["rna:mod_weight"] = 0.5
mdata.obs["atac:mod_weight"] = 0.5


def test_default_params():

    # Initialize the Mowgli model.
    model = models.MowgliModel(
        latent_dim=latent_dim,
        cost_path="cost.npy",
    )

    # Train the model.
    model.train(mdata)

    # Check the size of the embedding.
    assert mdata.obsm["W_OT"].shape == (n_cells, latent_dim)

    # Check the size of the dictionaries.
    assert mdata["rna"].uns["H_OT"].shape == (n_genes, latent_dim)
    assert mdata["atac"].uns["H_OT"].shape == (n_peaks, latent_dim)


def test_custom_params():

    # Initialize the Mowgli model.
    model = models.MowgliModel(
        latent_dim=latent_dim,
        h_regularization={"rna": 0.1, "atac": 0.1},
        use_mod_weight=True,
        pca_cost=True,
        cost_path="cost.npy",
    )
    model.init_parameters(
        mdata,
        force_recompute=True,
        normalize_rows=True,
        dtype=torch.float,
        device="cpu",
    )

    # Train the model.
    model.train(mdata, optim_name="adam")

    # Check the size of the embedding.
    assert mdata.obsm["W_OT"].shape == (n_cells, latent_dim)

    # Check the size of the dictionaries.
    assert mdata["rna"].uns["H_OT"].shape == (n_genes, latent_dim)
    assert mdata["atac"].uns["H_OT"].shape == (n_peaks, latent_dim)

from context import models, pl, tl, score
import muon as mu
import anndata as ad
import torch
import numpy as np

# Define some gene names (useful for enrichment analysis).
gene_names = [
    "ENSG00000125877",
    "ENSG00000184840",
    "ENSG00000164440",
    "ENSG00000177144",
    "ENSG00000186815",
    "ENSG00000079974",
    "ENSG00000136159",
    "ENSG00000177243",
    "ENSG00000163932",
    "ENSG00000112799",
    "ENSG00000075618",
    "ENSG00000092531",
    "ENSG00000171408",
    "ENSG00000150527",
    "ENSG00000202429",
    "ENSG00000140807",
    "ENSG00000154589",
    "ENSG00000166263",
    "ENSG00000205268",
    "ENSG00000115008",
]

n_cells, n_genes, n_peaks = 20, len(gene_names), 5
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
mdata.obs["label"] = np.random.choice(["A", "B", "C"], size=n_cells)


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


def test_plotting():

    # Make a clustermap.
    pl.clustermap(mdata)

    # Make a violin plot.
    pl.factor_violin(mdata, groupby="label", dim=0)

    # Make a heatmap.
    pl.heatmap(mdata, groupby="label")


def test_tools():

    # Compute top genes.
    tl.top_features(mdata, mod="rna", dim=0, threshold=0.2)

    # Compute top peaks.
    tl.top_features(mdata, mod="atac", dim=0, threshold=0.2)

    # Compute enrichment.
    tl.enrich(mdata, n_genes=10, ordered=False)


def test_score():

    # Compute a silhouette score.
    score.embedding_silhouette_score(
        embedding=mdata.obsm["W_OT"],
        labels=mdata.obs["label"],
        metric="euclidean",
    )

    # Compute leiden clustering across resolutions.
    score.embedding_leiden_across_resolutions(
        embedding=mdata.obsm["W_OT"],
        labels=mdata.obs["label"],
        n_neighbors=10,
        resolutions=[0.1, 0.5, 1.0],
    )

    # Compute a knn from the embedding.
    knn = score.embedding_to_knn(embedding=mdata.obsm["W_OT"], k=15, metric="euclidean")

    # Compute the knn purity score.
    score.knn_purity_score(knn=knn, labels=mdata.obs["label"])
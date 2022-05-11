################################### IMPORTS ###################################

# Biology imports.
import scanpy as sc
import muon as mu
import anndata as ad

# Typing imports.
from typing import Dict, Iterable, List

# Matrix operations.
import numpy as np

# Statistics.
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from scipy.stats import pearsonr, spearmanr

# Pretty progress bars.
from tqdm import tqdm

# Distance functions.
from scipy.spatial.distance import cdist

# Networks.
from sknetwork.topology import get_connected_components

from mowgli import score

################################## EMBEDDING ##################################


def umap(
    mdata: mu.MuData,
    obsm: str,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    **kwds
) -> None:
    """Compute UMAP of the given `obsm`.

    Args:
        mdata (mu.MuData): Input data.
        obsm (str): The embedding.
        n_neighbors (int, optional):
            Number of neighbors for UMAP. Defaults to 15.
        metric (str, optional):
            Which metric to compute neighbors. Defaults to 'euclidean'.
    """

    # Create an AnnData from the joint embedding.
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)

    # Compute neighbours on that embedding.
    sc.pp.neighbors(
        joint_embedding, use_rep="X", n_neighbors=n_neighbors, metric=metric
    )

    # Compute UMAP based on these neighbours.
    sc.tl.umap(joint_embedding, **kwds)

    # Copy the UMPA embedding to the input data's obsm field.
    mdata.obsm[obsm + "_umap"] = joint_embedding.obsm["X_umap"]


################################## CLUSTERING #################################


def leiden(
    mdata: mu.MuData,
    n_neighbors: int = 15,
    obsm: str = "W_OT",
    resolution: float = 1,
):
    """Perform Leiden clustering on the joint embedding.

    Args:
        mdata (mu.MuData): The input data.
        n_neighbors (int, optional): Number of neighbours. Defaults to 15.
        obsm (str, optional): Which obsm field to consider. Defaults to 'W_OT'.
        resolution (float, optional): The Leiden resolution. Defaults to 1.
    """

    # Create an AnnData from the joint embedding.
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)

    # Compute neighbors based on that joint embedding.
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)

    # Perform Leiden clustering.
    sc.tl.leiden(joint_embedding, resolution=resolution)

    # Copy the Leiden labels to the input object.
    mdata.obs["leiden"] = joint_embedding.obs["leiden"]


############################### ANALYSE FACTORS ###############################


def top_features(
    mdata: mu.MuData,
    mod: str = "rna",
    uns: str = "H_OT",
    dim: int = 0,
    n_features: int = 5,
) -> Iterable:
    """Returns the top features for a given modality and latent dimension.

    Args:
        mdata (mu.MuData): The input data
        mod (str, optional): The modality. Defaults to 'rna'.
        uns (str, optional): Where to look for H. Defaults to 'H_OT'.
        dim (int, optional): The latent dimension. Defaults to 0.
        n_features (int, optional): The number of top features. Defaults to 5.

    Returns:
        Iterable: A list of features names.
    """
    # TODO: put variable names in uns!

    # Get names for highly variable features.
    idx = mdata[mod].var.highly_variable
    var_names = mdata[mod].var_names[idx]

    # Sort them by contribution.
    var_idx = np.argsort(mdata[mod].uns[uns][:, dim])[::-1]

    # Return the top ones.
    return var_names[var_idx[:n_features]].tolist()


def enrich(
    mdata: mu.MuData,
    mod: str = "rna",
    uns: str = "H_OT",
    n_genes: int = 200,
    sources: Iterable[str] = ["GO:MF", "GO:CC", "GO:BP"],
    ordered: bool = True,
    domain_scope="custom_annotated",
):
    """Return Gene Set Enrichment Analysis results for each dimension.

    Args:
        mdata (mu.MuData): Input data.
        mod (str, optional): Modality that contains genes. Defaults to 'rna'.
        uns (str, optional): Name of H matrix. Defaults to 'H_OT'.
        n_genes (int, optional):
            Number of top genes by dimension. Defaults to 200.
        sources (Iterable[str], optional):
            Enrichment sources. Defaults to ['GO:MF', 'GO:CC', 'GO:BP'].
        ordered (bool, optional):
            Make query with ordered genes. Defaults to True.

    Returns:
        Pandas dataframe with the results of the queries, as well
        as average best p_value across dimensions.
    """

    # Initialize ordered genes dictionary.
    ordered_genes = {}

    # Get background genes.
    background = mdata[mod].var.index.tolist()

    # For each dimension,
    for dim in range(mdata[mod].uns[uns].shape[1]):

        # Sort the gene indices by weight.
        idx_sorted = mdata[mod].uns[uns][:, dim].argsort()[::-1]

        if n_genes == "auto":
            nn = np.sum(np.cumsum(np.sort(mdata[mod].uns[uns][:, dim])) > 0.1)
        else:
            nn = n_genes

        # Select the `n_genes` highest genes.
        gene_list = mdata[mod].var[mdata[mod].var.highly_variable].index
        gene_list = gene_list[idx_sorted].tolist()[:nn]

        # Input them in the dictionary.
        ordered_genes["dimension " + str(dim)] = gene_list

    # Make the queries to gProfiler, specifying if genes are ordered.
    if "custom" in domain_scope:
        enr = sc.queries.enrich(
            ordered_genes,
            gprofiler_kwargs={
                "ordered": ordered,
                "sources": sources,
                "domain_scope": domain_scope,
                "background": background,
                "no_evidences": True,
            },
        )
    else:
        enr = sc.queries.enrich(
            ordered_genes,
            gprofiler_kwargs={
                "ordered": ordered,
                "sources": sources,
                "domain_scope": domain_scope,
                "no_evidences": True,
            },
        )

    # Compute the average of the best p_values for each dimension.
    mean_best_p = enr.groupby("query")["p_value"].min().mean()

    # Return the results of the queries and the average best p_value.
    return enr, mean_best_p

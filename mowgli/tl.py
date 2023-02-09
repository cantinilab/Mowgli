# Biology imports.
import scanpy as sc
import mudata as md

# Typing imports.
from typing import Iterable

# Matrix operations.
import numpy as np


def top_features(
    mdata: md.MuData,
    mod: str = "rna",
    uns: str = "H_OT",
    dim: int = 0,
    threshold: float = 0.2,
) -> Iterable:
    """Returns the top features for a given modality and latent dimension.

    Args:
        mdata (md.MuData): The input data
        mod (str, optional): The modality. Defaults to 'rna'.
        uns (str, optional): Where to look for H. Defaults to 'H_OT'.
        dim (int, optional): The latent dimension. Defaults to 0.
        n_features (int, optional): The number of top features. Defaults to 5.

    Returns:
        Iterable: A list of features names.
    """
    # TODO: put variable names in uns!

    # Compue the number of features needed.
    n_features = np.sum(np.cumsum(np.sort(mdata[mod].uns[uns][:, dim])) > threshold)

    # Get names for highly variable features.
    idx = mdata[mod].var.highly_variable
    var_names = mdata[mod].var_names[idx]

    # Sort them by contribution.
    var_idx = np.argsort(mdata[mod].uns[uns][:, dim])[::-1]

    # Return the top ones.
    return var_names[var_idx[:n_features]].tolist()


def enrich(
    mdata: md.MuData,
    mod: str = "rna",
    uns: str = "H_OT",
    n_genes: int = 200,
    sources: Iterable[str] = ["GO:MF", "GO:CC", "GO:BP"],
    ordered: bool = True,
    domain_scope="custom_annotated",
):
    """Return Gene Set Enrichment Analysis results for each dimension.

    Args:
        mdata (md.MuData): Input data.
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

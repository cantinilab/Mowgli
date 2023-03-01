import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt


def clustermap(mdata: md.MuData, obsm: str = "W_OT", cmap="viridis", **kwds):
    """Wrapper around Scanpy's clustermap.

    Args:
        mdata (md.MuData): The input data
        obsm (str, optional): The obsm field to consider. Defaults to 'W_OT'.
        cmap (str, optional): The colormap. Defaults to 'viridis'.
    """

    # Create an AnnData with the joint embedding.
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)

    # Make the clustermap plot.
    sc.pl.clustermap(joint_embedding, cmap=cmap, **kwds)


def factor_violin(
    mdata: md.MuData,
    groupby: str,
    obsm: str = "W_OT",
    dim: int = 0,
    **kwds,
):
    """Make a violin plot of cells for a given latent dimension.

    Args:
        mdata (md.MuData): The input data
        dim (int, optional): The latent dimension. Defaults to 0.
        obsm (str, optional): The embedding. Defaults to 'W_OT'.
        groupby (str, optional): Observation groups.
    """

    # Create an AnnData with the joint embedding.
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)

    # Add the obs field that we're interested in.
    joint_embedding.obs["Factor " + str(dim)] = joint_embedding.X[:, dim]

    # Make the violin plot.
    sc.pl.violin(joint_embedding, keys="Factor " + str(dim), groupby=groupby, **kwds)


def heatmap(
    mdata: md.MuData,
    groupby: str,
    obsm: str = "W_OT",
    cmap: str = "viridis",
    sort_var: bool = False,
    save: str = None,
    **kwds,
) -> None:
    """Produce a heatmap of an embedding

    Args:
        mdata (md.MuData): Input data
        groupby (str): What to group by
        obsm (str): The embedding. Defaults to 'W_OT'.
        cmap (str, optional): Color map. Defaults to 'viridis'.
        sort_var (bool, optional):
            Sort dimensions by variance. Defaults to False.
    """

    # Create an AnnData with the joint embedding.
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)

    # Try to compute a dendrogram.
    try:
        sc.pp.pca(joint_embedding)
        sc.tl.dendrogram(joint_embedding, groupby=groupby, use_rep="X_pca")
    except:
        pass

    # Get the dimension names to show.
    if sort_var:
        idx = joint_embedding.X.std(0).argsort()[::-1]
        var_names = joint_embedding.var_names[idx]
    else:
        var_names = joint_embedding.var_names

    # PLot the heatmap.
    return sc.pl.heatmap(
        joint_embedding, var_names, groupby=groupby, cmap=cmap, save=save, **kwds
    )


def enrich(enr: pd.DataFrame, query_name: str, n_terms: int = 10):
    """Display a list of enriched terms.

    Args:
        enr (pd.DataFrame): The enrichment object returned by mowgli.tl.enrich()
        query_name (str): The name of the query, e.g. "dimension 0".
    """

    # Subset the enrichment object to the query of interest.
    sub_enr = enr[enr["query"] == query_name].head(n_terms)
    sub_enr["minlogp"] = -np.log10(sub_enr["p_value"])

    fig, ax = plt.subplots()

    # Display the enriched terms.
    ax.hlines(
        y=sub_enr["name"],
        xmin=0,
        xmax=sub_enr["minlogp"],
        color="lightgray",
        zorder=1,
        alpha=0.8,
    )
    sns.scatterplot(
        data=sub_enr,
        x="minlogp",
        y="name",
        hue="source",
        s=100,
        alpha=0.8,
        ax=ax,
        zorder=3,
    )

    ax.set_xlabel("$-log_{10}(p)$")
    ax.set_ylabel("Enriched terms")

    plt.show()


def top_features(
    mdata: md.MuData,
    mod: str = "rna",
    uns: str = "H_OT",
    dim: int = 0,
    n_top: int = 10,
):
    """Display the top features for a given dimension.

    Args:
        mdata (md.MuData): The input data
        mod (str, optional): The modality to consider. Defaults to 'rna'.
        uns (str, optional): The uns field to consider. Defaults to 'H_OT'.
        dim (int, optional): The latent dimension. Defaults to 0.
        n_top (int, optional): The number of top features to display. Defaults to 10.
    """

    # Get the variable names.
    var_names = mdata[mod].var_names[mdata[mod].var.highly_variable]

    # Get the top features.
    idx_top_features = np.argsort(mdata[mod].uns[uns][:, dim])[::-1][:n_top]
    df = pd.DataFrame(
        {
            "features": var_names[idx_top_features],
            "weights": mdata[mod].uns[uns][idx_top_features, dim],
        }
    )

    # Display the top features.
    sns.barplot(data=df, x="weights", y="features", palette="Blues_r")

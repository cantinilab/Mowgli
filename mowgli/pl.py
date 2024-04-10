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
    except Exception:
        print("Dendrogram not computed.")
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
    ax: plt.axes = None, 
    palette: str = 'Blues_r'
):
    """Display the top features for a given dimension.

    Args:
        mdata (md.MuData): The input mdata object
        mod (str, optional): The modality to consider. Defaults to 'rna'.
        uns (str, optional): The uns field to consider. Defaults to 'H_OT'.
        dim (int, optional): The latent dimension. Defaults to 0.
        n_top (int, optional): The number of top features to display. Defaults to 10.
        ax (plt.axes, optional): The axes to use. Defaults to None.
        palette (str, optional): The color palette to use. Defaults to 'Blues_r'.

    Returns:
        plt.axes: The axes used.
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
    if ax is None:
        ax = sns.barplot(data=df, x="weights", y="features", palette=palette)
    else:
        sns.barplot(data=df, x="weights", y="features", palette=palette, ax=ax)
    
    return ax

def umap(mdata: md.MuData, dim: int | list = 0,  rescale: bool = False, obsm: str = "W_OT", neighbours_key = None, **kwds):
    """Wrapper around Scanpy's sc.pl.umap. Computes UMAP for a given latent dimension and plots it.
    Args:
        mdata (md.MuData): The input data
        dim (int | list, optional): The latent dimension. Defaults to 0.
        rescale (bool, optional): If True, Rescale the color palette across all plots to the maximum value in the weight matrix. Defaults to False.
        obsm (str, optional): The embedding. Defaults to 'W_OT'.
        neighbours_key (str, optional): The key for the neighbours in `mdata.uns` to use to compute neighbors. Defaults to None.
    """

    adata_tmp = ad.AnnData(mdata.obsm[obsm], obs=pd.DataFrame(index=mdata.obs.index))
    
    if isinstance(dim, int):
        mowgli_cat = f"mowgli:{dim}"
        
    elif isinstance(dim ,list):
        # clean dim of doubles and sort them
        dim = sorted(list(set(dim)))
        mowgli_cat = [f"mowgli:{x}" for x in dim]

    else:
        raise ValueError("dim must be an integer or a list of integers")

    adata_tmp.obs[mowgli_cat] = adata_tmp.X[:, dim]

    # check if neighbors exists
    if neighbours_key is None:
        print("Computing neighbors with scanpy default parameters")
        neighbours_key = "mowgli_neighbors" # set the default neighbors key 
        # compute neiughborts using all dimension in the mowgli matrix 
        sc.pp.neighbors(adata_tmp, use_rep='X', key_added = neighbours_key)
        
    else:
        if neighbours_key not in mdata.uns.keys():
            raise ValueError(f"neighbours key {neighbours_key} not found in mdata.uns")
        
        adata_tmp.uns[neighbours_key] = mdata.uns[neighbours_key]

    # compute umap 
    print('Computing UMAP')
    sc.tl.umap(adata_tmp, neighbors_key = neighbours_key)

    # plot umap
    if rescale:
        vmax = adata_tmp.X.max()
        sc.pl.umap(adata_tmp, color=mowgli_cat, size=18.5, alpha=0.4, vmax = vmax,  **kwds)
    else:
        sc.pl.umap(adata_tmp, color=mowgli_cat, size=18.5, alpha=0.4, **kwds)
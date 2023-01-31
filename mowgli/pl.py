import scanpy as sc
import muon as mu
import anndata as ad


def clustermap(mdata: mu.MuData, obsm: str = "W_OT", cmap="viridis", **kwds):
    """Wrapper around Scanpy's clustermap.

    Args:
        mdata (mu.MuData): The input data
        obsm (str, optional): The obsm field to consider. Defaults to 'W_OT'.
        cmap (str, optional): The colormap. Defaults to 'viridis'.
    """

    # Create an AnnData with the joint embedding.
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)

    # Make the clustermap plot.
    sc.pl.clustermap(joint_embedding, cmap=cmap, **kwds)


def factor_violin(
    mdata: mu.MuData,
    groupby: str,
    obsm: str = "W_OT",
    dim: int = 0,
    **kwds,
):
    """Make a violin plot of cells for a given latent dimension.

    Args:
        mdata (mu.MuData): The input data
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
    mdata: mu.MuData,
    groupby: str,
    obsm: str = "W_OT",
    cmap: str = "viridis",
    sort_var: bool = False,
    save: str = None,
    **kwds,
) -> None:
    """Produce a heatmap of an embedding

    Args:
        mdata (mu.MuData): Input data
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

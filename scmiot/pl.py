# Biology
import scanpy as sc
import muon as mu
import anndata as ad
import numpy as np


def heatmap(mdata: mu.MuData, obsm: str, groupby: str, cmap: str = 'viridis', sort_var: bool = False) -> None:
    """Produce a heatmap of an embedding

    Args:
        mdata (mu.MuData): Input data
        obsm (str): The embedding
        groupby (str): What to group by
        cmap (str, optional): Color map. Defaults to 'viridis'.
        sort_var (bool, optional): Sort dimensions by variance. Defaults to False.
    """
    
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    if sort_var:
        idx = joint_embedding.var_names[joint_embedding.X.std(0).argsort()[::-1]]
    else:
        idx = joint_embedding.var_names
    sc.pl.heatmap(joint_embedding, idx, groupby=groupby, cmap=cmap)
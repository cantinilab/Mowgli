# Biology
import scanpy as sc
import muon as mu
import anndata as ad
import numpy as np


def heatmap(mdata, obsm, groupby):
    try:
        joint_embedding = ad.AnnData(mdata.obsm[obsm].cpu().numpy(), obs=mdata.obs)
    except:
        joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    idx = joint_embedding.var_names[joint_embedding.X.std(0).argsort()[::-1]]
    sc.pl.heatmap(joint_embedding, idx, groupby=groupby, cmap='viridis')

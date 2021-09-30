# Biology
import scanpy as sc
import muon as mu
import anndata as ad


def heatmap(mdata, obsm, groupby):
    try:
        joint_embedding = ad.AnnData(mdata.obsm[obsm].cpu().numpy(), obs=mdata.obs)
    except:
        joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    idx = joint_embedding.var_names[joint_embedding.X.std(0).argsort()[::-1]]
    sc.pl.heatmap(joint_embedding, idx, groupby=groupby, cmap='viridis')

def umap(mdata, obsm, groupby, size=50):
    try:
        joint_embedding = ad.AnnData(mdata.obsm[obsm].cpu().numpy(), obs=mdata.obs)
    except:
        joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    sc.pp.neighbors(joint_embedding)
    sc.tl.umap(joint_embedding)
    sc.pl.umap(joint_embedding, color=groupby, size=size)

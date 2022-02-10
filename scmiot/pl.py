# Biology
import scanpy as sc
import muon as mu
import anndata as ad
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import torch
from torch.nn.utils import parameters_to_vector as Params2Vec
from torch.nn.utils import vector_to_parameters as Vec2Params

def heatmap(mdata: mu.MuData, obsm: str, groupby: str, cmap: str = 'viridis',
            sort_var: bool = False, save: str = None, **kwds) -> None:
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
    sc.pl.heatmap(joint_embedding, idx, groupby=groupby, cmap=cmap, save=save, **kwds)

def tau_2d(alpha, beta, theta):
  a = torch.rand_like(theta[:,None,None])
  b = torch.rand_like(theta[:,None,None])
  return (1 - alpha/2 - beta/2)*theta[:,None,None] + alpha*a/2 + beta*b/2

def plot_loss(model, loss_fn, resolution=30, span=10) -> None:
    G_list = [model.G[mod] for mod in model.G]
    G_vec = Params2Vec(G_list)

    x = torch.linspace(-1, 1, resolution)*span
    y = torch.linspace(-1, 1, resolution)*span
    alpha, beta = torch.meshgrid(x, y)
    space = tau_2d(alpha, beta, G_vec)

    losses = torch.empty_like(space[0, :, :])
    for a in range(len(x)):
        for b in range(len(y)):
            Vec2Params(space[:, a, b], G_list)
            losses[a][b] = loss_fn().detach().cpu()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(alpha, beta, losses.numpy(), cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5)

    Vec2Params(G_vec, G_list)

    plt.show()

def plot_loss_h(model, **kwds):
    plot_loss(model, model.loss_fn_h, **kwds)

def plot_loss_w(model, **kwds):
    plot_loss(model, model.loss_fn_w, **kwds)
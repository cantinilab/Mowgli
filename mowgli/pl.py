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
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

def rankings(mdata, mod = 'rna', uns = 'H_OT', dim = 0, n_show = 5):
    weights = mdata[mod].uns[uns][:,dim]
    idx = np.argsort(weights)
    var_names = mdata[mod].var_names[mdata[mod].var['highly_variable']]

    yy = weights[idx]
    xx = np.arange(len(weights))
    ss = np.array(var_names.tolist())[idx]

    plt.scatter(xx, yy, s=10)
    for x, y, s in zip(xx[-n_show:], yy[-n_show:], ss[-n_show:]):
        plt.text(x, y, s)
    plt.title('Factor ' + str(dim) + ' (' + mod + ')')
    plt.xlabel('Ranking')
    plt.ylabel('Weight')
    plt.show()

def riverplot(H_list, threshold):
    k_list = [H.shape[1] for H in H_list]
    id_left = 0
    id_right = k_list[0]

    labels_right = ['K' + str(k_list[0]) + '_' + str(i) for i in range(k_list[0])]
    labels = [] + labels_right

    source, target, value = [], [], []

    for p in range(1, len(k_list)):

        labels_left = [] + labels_right
        labels_right = ['K' + str(k_list[p]) + '_' + str(i) for i in range(k_list[p])]
        labels += labels_right

        D = cdist(H_list[p-1].T, H_list[p].T, metric='cosine')

        for i in range(H_list[p-1].shape[1]):
            for j in range(H_list[p].shape[1]):
                source.append(id_left + i)
                target.append(id_right + j)
                value.append(1 - D[i, j])
        
        id_left = id_right
        id_right = id_left + k_list[p]
    
    color = []
    for i in range(len(value)):
        if value[i] < threshold:
            color.append("rgba(0,0,0,.02)")
        else:
            color.append("lightgrey")

    fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 10,
        label = labels,
    ),
    link = dict(
        source = source,
        target = target,
        value = value,
        color = color
    ))])

    fig.update_layout(title_text="Cosine similarity between factors", font_size=10)
    fig.show()

def clustermap(mdata: mu.MuData, obsm: str = 'W_OT', cmap='viridis', **kwds):
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    sc.pl.clustermap(joint_embedding, cmap=cmap, **kwds)

def factor_violin(
    mdata: mu.MuData, groupby: str, obsm: str = 'W_OT', dim: int = 0, **kwds):
    """Make a violin plot of cells for a given latent dimension.

    Args:
        mdata (mu.MuData): The input data
        dim (int, optional): The latent dimension. Defaults to 0.
        obsm (str, optional): The embedding. Defaults to 'W_OT'.
        groupby (str, optional): Observation groups.
    """    
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    joint_embedding.obs['Factor ' + str(dim)] = joint_embedding.X[:,dim]
    sc.pl.violin(joint_embedding, keys='Factor ' + str(dim), groupby=groupby, **kwds)

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
    try:
        sc.pp.pca(joint_embedding)
        sc.tl.dendrogram(joint_embedding, groupby=groupby, use_rep='X_pca')
    except:
        pass
    if sort_var:
        idx = joint_embedding.var_names[joint_embedding.X.std(0).argsort()[::-1]]
    else:
        idx = joint_embedding.var_names
    return sc.pl.heatmap(joint_embedding, idx, groupby=groupby, cmap=cmap, save=save, **kwds)

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
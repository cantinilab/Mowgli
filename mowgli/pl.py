import scanpy as sc
import muon as mu
import anndata as ad

import numpy as np
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go

import torch
from torch.nn.utils import parameters_to_vector as Params2Vec
from torch.nn.utils import vector_to_parameters as Vec2Params


def rankings(
    mdata: mu.MuData, mod: str = "rna", uns: str = "H_OT", dim: int = 0, n_show: int = 5
) -> None:
    """Display the feature ranking in a dimension of the dictionary.

    Args:
        mdata (mu.MuData): The input data.
        mod (str, optional): The modality to look at. Defaults to 'rna'.
        uns (str, optional): The uns key to consider. Defaults to 'H_OT'.
        dim (int, optional): The dimension to consider. Defaults to 0.
        n_show (int, optional):
            The number of features to highlight. Defaults to 5.
    """

    # Recover the weights of features.
    weights = mdata[mod].uns[uns][:, dim]

    # Sort these weights.
    idx = np.argsort(weights)

    # Get the feature names for plotting.
    var_names = mdata[mod].var_names[mdata[mod].var["highly_variable"]]

    # Get the weights and names after sorting.
    xx = np.arange(len(weights))
    yy = weights[idx]
    ss = np.array(var_names.tolist())[idx]

    # Plot the weights.
    plt.scatter(xx, yy, s=10)

    # Plot the feature names.
    for x, y, s in zip(xx[-n_show:], yy[-n_show:], ss[-n_show:]):
        plt.text(x, y, s)

    # Make the legend and show.
    plt.title("Factor " + str(dim) + " (" + mod + ")")
    plt.xlabel("Ranking")
    plt.ylabel("Weight")
    plt.show()


def riverplot(H_list: list, threshold: float):
    """Plot the cosine similarity between factors as a riverplot.

    Args:
        H_list (list): A list of H matrices for various latent dimensions.
        threshold (float): A threshold to display the link.
    """

    # Get the list of latent dimensions.
    k_list = [H.shape[1] for H in H_list]

    # Initialize the left and right ids.
    id_left = 0
    id_right = k_list[0]

    # Initialize the labels.
    labels_right = ["K" + str(k_list[0]) + "_" + str(i) for i in range(k_list[0])]
    labels = [] + labels_right

    # Initalize the source, target and values.
    source, target, value = [], [], []

    # For each entry in the input,
    for p in range(1, len(k_list)):

        # Add the label.
        labels_left = [] + labels_right
        labels_right = ["K" + str(k_list[p]) + "_" + str(i) for i in range(k_list[p])]
        labels += labels_right

        # Compute the cosine similarities.
        D = cdist(H_list[p - 1].T, H_list[p].T, metric="cosine")

        # Add the corresponding links to the plot.
        for i in range(H_list[p - 1].shape[1]):
            for j in range(H_list[p].shape[1]):
                source.append(id_left + i)
                target.append(id_right + j)
                value.append(1 - D[i, j])

        # Update the indices.
        id_left = id_right
        id_right = id_left + k_list[p]

    # Set colors depending on the threshold.
    color = []
    for i in range(len(value)):
        if value[i] < threshold:
            color.append("rgba(0,0,0,.02)")
        else:
            color.append("lightgrey")

    # Make the figure.
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=10,
                    label=labels,
                ),
                link=dict(source=source, target=target, value=value, color=color),
            )
        ]
    )

    # Set title and show.
    title_text = "Cosine similarity between factors"
    fig.update_layout(title_text=title_text, font_size=10)
    fig.show()


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
    mdata: mu.MuData, groupby: str, obsm: str = "W_OT", dim: int = 0, **kwds
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
    obsm: str,
    groupby: str,
    cmap: str = "viridis",
    sort_var: bool = False,
    save: str = None,
    **kwds
) -> None:
    """Produce a heatmap of an embedding

    Args:
        mdata (mu.MuData): Input data
        obsm (str): The embedding
        groupby (str): What to group by
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


def tau_2d(alpha, beta, theta):
    """Random 2d map used to plot the loss landscape.

    Args:
        alpha (_type_): Meshgrid x
        beta (_type_): Meshgrid y
        theta (_type_): Parameter vector.

    Returns: Interpolation.
    """

    # Generate random parameters.
    a = torch.rand_like(theta[:, None, None])
    b = torch.rand_like(theta[:, None, None])

    # Interpolate between the optimal parameter and the random ones.
    return (
        (1 - alpha / 2 - beta / 2) * theta[:, None, None] + alpha * a / 2 + beta * b / 2
    )


def plot_loss(model, loss_fn, resolution=30, span=10) -> None:
    """Plot the loss landscape in 3d.

    Args:
        model (_type_): The model.
        loss_fn (_type_): The loss function.
        resolution (int, optional): The precision of the plot. Defaults to 30.
        span (int, optional): The span of the plot. Defaults to 10.
    """

    # Turn the parameters into a vector.
    G_list = [model.G[mod] for mod in model.G]
    G_vec = Params2Vec(G_list)

    # Generate a grid interpolation between G and random parameters.
    x = torch.linspace(-1, 1, resolution) * span
    y = torch.linspace(-1, 1, resolution) * span
    alpha, beta = torch.meshgrid(x, y)
    space = tau_2d(alpha, beta, G_vec)

    # Compute the lossed for all parameter possibilities on the grid.
    losses = torch.empty_like(space[0, :, :])
    for a in range(len(x)):
        for b in range(len(y)):
            Vec2Params(space[:, a, b], G_list)
            losses[a][b] = loss_fn().detach().cpu()

    # Intialize the figure.
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(
        alpha, beta, losses.numpy(), cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5)

    # Send G back to parameters.
    Vec2Params(G_vec, G_list)

    # Show the plot.
    plt.show()


def plot_loss_h(model, **kwds):
    """Convenience wrapper around `plot_loss`"""
    plot_loss(model, model.loss_fn_h, **kwds)


def plot_loss_w(model, **kwds):
    """Convenience wrapper around `plot_loss`"""
    plot_loss(model, model.loss_fn_w, **kwds)

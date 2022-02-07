# Biology
from typing import Iterable, List, Tuple
import scanpy as sc
import muon as mu
import anndata as ad
from sklearn.metrics import r2_score, explained_variance_score, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from scipy.stats import pearsonr, spearmanr
from sknetwork.topology import get_connected_components

def umap(mdata: mu.MuData, obsm: str, n_neighbors: int = 15, metric: str = 'euclidean', **kwds) -> None:
    """Compute UMAP of the given `obsm`.

    Args:
        mdata (mu.MuData): Input data
        obsm (str): The embedding
        n_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 15.
        metric (str, optional): Which metric to compute neighbors. Defaults to 'euclidean'.
    """
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    sc.pp.neighbors(joint_embedding, n_neighbors=n_neighbors, metric=metric)
    sc.tl.umap(joint_embedding, **kwds)

    mdata.obsm[obsm + '_umap'] = joint_embedding.obsm['X_umap']

def sil_score(mdata: mu.MuData, obsm: str, obs: str) -> float:
    """Compute silhouette score of an embedding

    Args:
        mdata (mu.MuData): Input data
        obsm (str): Embedding
        obs (str): Annotation

    Returns:
        float: Silhouette score
    """
    return silhouette_score(mdata.obsm[obsm], mdata.obs[obs])

def knn_score(mdata: mu.MuData, obs: str, obsm: str = 'W_OT', max_neighbors: int = 15) -> List:
    """Computes the k-NN purity score, for varying numbers of neighbors between 1 and 15.

    Args:
        mdata (mu.MuData): Input data
        obs (str): Annotation
        obsm (str, optional): Embedding. Defaults to 'W_OT'.
        max_neighbors (int, optional): Maximum number of neighbors. Defaults to 15.

    Returns:
        List[int]: The kNN scores for varying k.
    """    
    distances = cdist(mdata.obsm[obsm], mdata.obsm[obsm])
    s = 0
    for i in tqdm(range(mdata.n_obs)):
        idx = distances[i].argsort()[1:max_neighbors]
        s += np.cumsum(np.array(mdata.obs[obs][i] == mdata.obs[obs][idx]))/np.arange(1, max_neighbors)
    return s / mdata.n_obs

def leiden_multi(mdata: mu.MuData, n_neighbors: int = 15,
                obsm: str = 'W_OT', obs: str = 'rna:celltype',
                resolutions: Iterable[float] = np.arange(.1, 2.1, .1)):
    """Compute leiden clustering for multiple resultions, and return ARI and NMI.

    Args:
        mdata (mu.MuData): Input data.
        n_neighbors (int, optional): Number of neighbors. Defaults to 15.
        obsm (str, optional): Obsm to use. Defaults to 'W_OT'.
        obs (str, optional): Annotation. Defaults to 'rna:celltype'.
        resolutions (Iterable[float], optional): Iterable of resultions. Defaults to np.arange(.1, 2.1, .1).

    Returns:
        [type]: [description]
    """    
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    aris, nmis = [], []
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)
    for resolution in tqdm(resolutions):
        sc.tl.leiden(joint_embedding, resolution=resolution)
        aris.append(ARI(joint_embedding.obs['leiden'], mdata.obs[obs]))
        nmis.append(NMI(joint_embedding.obs['leiden'], mdata.obs[obs]))
    return resolutions, aris, nmis

def leiden_multi_silhouette(mdata: mu.MuData, n_neighbors: int = 15,
                obsm: str = 'W_OT', obs: str = 'rna:celltype',
                resolutions: Iterable[float] = np.arange(.1, 2.1, .1)):
    """Compute leiden clustering for multiple resultions, and return ARI and NMI.

    Args:
        mdata (mu.MuData): Input data.
        n_neighbors (int, optional): Number of neighbors. Defaults to 15.
        obsm (str, optional): Obsm to use. Defaults to 'W_OT'.
        obs (str, optional): Annotation. Defaults to 'rna:celltype'.
        resolutions (Iterable[float], optional): Iterable of resultions. Defaults to np.arange(.1, 2.1, .1).

    Returns:
        [type]: [description]
    """    
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    sils = []
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)
    for resolution in tqdm(resolutions):
        sc.tl.leiden(joint_embedding, resolution=resolution)
        if len(np.unique(joint_embedding.obs['leiden'])) > 1:
            sils.append(silhouette_score(joint_embedding.X, joint_embedding.obs['leiden']))
        else:
            sils.append(-1)
    return resolutions, sils

def predict_features_corr(mdata: mu.MuData, mod: str, n_neighbors: int,
                        features_idx: Iterable[int] = [],
                        remove_zeros: bool = True, obsp: str = None,
                        obsm: str = None):
    """Output a the Pearson and Spearman correlation scores between the actual
       values of a modality and the prediction based on a joint embedding.

    Args:
        mdata (mu.MuData): Input data
        mod (str): Modality to predict
        n_neighbors (int): Number of neigbours
        features_idx (Iterable[int]): List of feature indices
        remove_zeros (bool, optional): Whether to remove values from the data.
            Indeed they might be dropouts, in which case the prediction might
            be better than the real values. Defaults to True.
        obsp (str, optional): Obsp key with distances. Defaults to None.
        obsm (str, optional): Obsm key with embedding. Defaults to None.

    Returns:
        Tuple((List, List)): Pearson correlation, Spearman correlation
    """

    if len(features_idx) == 0:
        features_idx = np.arange(mdata[mod].n_vars)

    if obsp: # If nearest neighbour distances are already computed.
        distances = np.array(mdata.obsp[obsp].todense())
        distances[distances == 0] = np.max(distances)
        np.fill_diagonal(distances, 0)
    else: # Else use obsm.
        distances = cdist(mdata.obsm[obsm], mdata.obsm[obsm])
    
    # Initialize prediction
    pred = np.zeros((mdata.n_obs, len(features_idx)))

    # For each cell,
    for i in tqdm(range(mdata.n_obs)):
        # Recover neighbors' indices
        idx = distances[i].argsort()[1:1+n_neighbors]

        # Take the mean of neighbors as prediction
        pred[i] = np.mean(mdata[mod].X[idx][:,features_idx], axis=0)
    
    # The truth to compute correlation against.
    truth = np.array(mdata[mod].X[:, features_idx])

    # Intitialize correlation
    pearson, spearman = [], []

    # For each cell,
    for i in range(len(features_idx)):
        # Truth and prediction for a given cell i
        x, y = truth[:,i], pred[:,i]

        # If `remove_zeros`, select indices where x > 0
        idx = x > 0 if remove_zeros else np.arange(len(x))

        # Append to correlation lists.
        pearson.append(pearsonr(x[idx], y[idx])[0])
        spearman.append(spearmanr(x[idx], y[idx])[0])
    
    return pearson, spearman

def variance_explained(mdata, score_function='explained_variance_score', plot=True):
    """experimental, i have to test this function"""
    if score_function == 'explained_variance_score':
        f_score = explained_variance_score
    elif score_function == 'r2_score':
        f_score = r2_score
    else:
        f_score = explained_variance_score
        print('function not recognized, defaulting to explained_variance_score')
    score = []
    k = mdata.obsm['W_OT'].shape[1]
    for mod in mdata.mod:
        score.append([])
        A = mdata.mod[mod].uns['H_OT'] @ mdata.obsm['W_OT'].T
        A = A.cpu().numpy()

        for i in range(k):
            rec = mdata.mod[mod].uns['H_OT'][:,[i]] @ mdata.obsm['W_OT'][:,[i]].T
            rec = rec.cpu().numpy()
            score[-1].append(f_score(A, rec))

    if plot:
        fig, ax = plt.subplots()
        plt.imshow(score, aspect='auto', interpolation='none')
        ax.set_xticks(range(k))
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Modality')
        ax.set_yticks(range(len(mdata.mod)))
        ax.set_title('Variance explained')
        ax.set_yticklabels(mdata.mod.keys())
        plt.colorbar()
        plt.show()
    return score

def graph_connectivity(mdata: mu.MuData, obs: str, obsm: str,
                      n_neighbors: int = 15) -> float:
    """Compute graph connectivity score, as defined in OpenProblems.

    Args:
        mdata (mu.MuData): Input data
        obs (str): Observation key
        obsm (str): Obsm key
        n_neighbors (int, optional): Number of neighbors. Defaults to 15.

    Returns:
        float: graph connectivity score
    """
    # Represent the joint embedding as an AnnData object.
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)

    # Find the joint embedding's neighbors.
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=15)

    # Define the adjacency matrix
    adjacency = joint_embedding.obsp['distances']

    # Initialize the proportions
    props = []

    # For all unique obsevervation categories,
    for celltype in np.unique(mdata.obs[obs]):

        # Define the indices of cells concerned.
        idx = np.where(mdata.obs[obs] == celltype)[0]

        try:
            # Find the connected components in the category's subgraph.
            conn_comp = get_connected_components(adjacency[idx][:,idx], connection='strong')

            # Count the occurences of the components.
            _, counts = np.unique(conn_comp, return_counts=True)

            # The proportion is the largest component over the number of cells in the cluster.
            props.append(counts.max() / idx.shape[0])
        except:
            props.append(0)
            print('Warning: empty component')

    # Return average of the proportions.
    return np.array(props).mean()

# def leiden_multi_obsp(mdata, n_neighbors=15, neighbors_key='wnn', obs='rna:celltype', resolutions=10.**np.linspace(-2, 1, 20)):
#     aris = []
#     nmis = []
#     for resolution in tqdm(resolutions):
#         sc.tl.leiden(mdata, resolution=resolution, neighbors_key=neighbors_key, key_added='leiden_' + neighbors_key)
#         aris.append(ARI(mdata.obs['leiden_' + neighbors_key], mdata.obs[obs]))
#         nmis.append(NMI(mdata.obs['leiden_' + neighbors_key], mdata.obs[obs]))
#     return resolutions, aris, nmis

def leiden(mdata, n_neighbors=15, obsm='W_OT', resolution=1):
    try:
        joint_embedding = ad.AnnData(mdata.obsm[obsm].cpu().numpy(), obs=mdata.obs)
    except:
        joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)
    sc.tl.leiden(joint_embedding, resolution=resolution)
    mdata.obs['leiden'] = joint_embedding.obs['leiden']

def inflexion_pt(a):
    second_derivative = [-np.inf]
    for i in range(1, len(a) - 1):
        second_derivative.append(a[i+1] + a[i-1] - 2 * a[i])
    return np.argmax(second_derivative)

def select_dimensions(mdata, plot=True):
    latent_dim = mdata.obsm['W_OT'].shape[1]
    s = np.zeros(latent_dim)
    for mod in mdata.mod:
        s += np.array([(mdata[mod].uns['H_OT'][:,[k]] @ mdata.obsm['W_OT'].T[[k]]).std(1).sum() for k in range(latent_dim)])

    i = inflexion_pt(np.sort(s)[::-1])
    i = max(i, 4)
    if plot:
        plt.plot(np.sort(s)[::-1])
        plt.scatter(range(latent_dim), np.sort(s)[::-1])
        plt.scatter(range(i+1), np.sort(s)[::-1][:i+1])
        plt.plot(np.sort(s)[::-1][:i+1])
        plt.show()
    return np.argsort(s)[::-1][:i+1].copy()

def trim_dimensions(mdata, dims):
    mdata.obsm['W_OT'] = mdata.obsm['W_OT'][:,dims]
    for mod in mdata.mod:
        mdata[mod].uns['H_OT'] = mdata[mod].uns['H_OT'][:,dims]

def best_leiden_resolution(mdata, obsm='W_OT', method='elbow', resolution_range=None, n_neighbors=15, plot=True):
    if resolution_range==None:
        resolution_range = 10.**np.linspace(-2, 1, 20)

    if method != 'elbow' and method != 'silhouette':
        print('method not recognized, defaulting to elbow')
        method = 'elbow'

    joint_embedding = ad.AnnData(mdata.obsm[obsm].cpu().numpy(), obs=mdata.obs)
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)

    if method == 'elbow':
        vars = []
        for res in resolution_range:
            sc.tl.leiden(joint_embedding, resolution=res)
            wss = []
            for cat in joint_embedding.obs['leiden'].unique():
                wss.append(joint_embedding.X[joint_embedding.obs['leiden'] == cat].std(0).sum())
            vars.append(np.mean(wss))

        i = inflexion_pt(vars)
        if plot:
            plt.xscale('log')
            plt.scatter(resolution_range, vars)
            plt.scatter(resolution_range[i], vars[i])
            plt.plot(resolution_range, vars)
            plt.ylabel('Average intra-cluster variation')
            plt.xlabel('Resolution')
            plt.show()

        return resolution_range[i]

    elif method == 'silhouette':
        sils = []
        for res in resolution_range:
            sc.tl.leiden(joint_embedding, resolution=res)
            try:
                sils.append(silhouette_score(joint_embedding.X, joint_embedding.obs['leiden']))
            except:
                sils.append(-1)

        maxes = []
        for i in range(1, len(sils)-1):
            if sils[i] > sils[i+1] and sils[i] >= sils[i-1]:
                maxes.append(i)

        if plot:
            plt.xscale('log')
            plt.scatter(resolution_range, sils)
            plt.plot(resolution_range, sils)
            plt.scatter([resolution_range[maxes[0]]], [sils[maxes[0]]])
            plt.ylabel('Silhouette score')
            plt.xlabel('Resolution')
            plt.show()

        return resolution_range[maxes[0]]

def enrich(mdata: mu.MuData, mod: str = 'rna', uns: str = 'H_OT', n_genes: int = 200,
           sources: Iterable[str] = ['GO:MF', 'GO:CC', 'GO:BP'], ordered: bool = True):
    """Return Gene Set Enrichment Analysis results for each dimension.

    Args:
        mdata (mu.MuData): Input data.
        mod (str, optional): Modality that contains genes. Defaults to 'rna'.
        uns (str, optional): Name of H matrix. Defaults to 'H_OT'.
        n_genes (int, optional): Number of top genes by dimension. Defaults to 200.
        sources (Iterable[str], optional): Enrichment sources. Defaults to ['GO:MF', 'GO:CC', 'GO:BP'].
        ordered (bool, optional): Make query with ordered genes. Defaults to True.

    Returns:
        [type]: Pandas dataframe with the results of the queries, as well as average best p_value across dimensions.
    """    
    
    # Initialize ordered genes dictionary.
    ordered_genes = {}

    # For each dimension,
    for dim in range(mdata[mod].uns[uns].shape[1]):
        # Sort the gene indices by weight.
        idx_sorted = mdata[mod].uns[uns][:,dim].argsort()[::-1]

        # Select the `n_genes` highest genes.
        gene_list = mdata[mod].var.index[idx_sorted].tolist()[:n_genes]

        # Input them in the dictionary.
        ordered_genes['dimension ' + str(dim)] = gene_list

    # Make the queries to gProfiler, specifying if genes are ordered.
    enr = sc.queries.enrich(ordered_genes, gprofiler_kwargs={
        'ordered': ordered,
        'sources': sources})
    
    # Compute the average of the best p_values for each dimension.
    mean_best_p = enr.groupby('query')['p_value'].min().mean()

    # Print the 5 top results.
    print(enr.loc[:5, ['name', 'p_value', 'query']])

    # Return the results of the queries and the average best p_value.
    return enr, mean_best_p
################################### IMPORTS ###################################

# Biology imports.
import scanpy as sc
import muon as mu
import anndata as ad

# Typing imports.
from typing import Dict, Iterable, List

# Matrix operations.
import numpy as np

# Statistics.
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from scipy.stats import pearsonr, spearmanr

# Pretty progress bars.
from tqdm import tqdm

# Distance functions.
from scipy.spatial.distance import cdist

# Networks.
from sknetwork.topology import get_connected_components

################################## EMBEDDING ##################################

def umap(
    mdata: mu.MuData, obsm: str, n_neighbors: int = 15,
    metric: str = 'euclidean', **kwds) -> None:
    """Compute UMAP of the given `obsm`.

    Args:
        mdata (mu.MuData): Input data.
        obsm (str): The embedding.
        n_neighbors (int, optional):
            Number of neighbors for UMAP. Defaults to 15.
        metric (str, optional):
            Which metric to compute neighbors. Defaults to 'euclidean'.
    """

    # Create an AnnData from the joint embedding.
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)

    # Compute neighbours on that embedding.
    sc.pp.neighbors(
        joint_embedding, use_rep="X", n_neighbors=n_neighbors, metric=metric)
    
    # Compute UMAP based on these neighbours.
    sc.tl.umap(joint_embedding, **kwds)

    # Copy the UMPA embedding to the input data's obsm field.
    mdata.obsm[obsm + '_umap'] = joint_embedding.obsm['X_umap']

################################## CLUSTERING #################################

def leiden(
    mdata: mu.MuData, n_neighbors: int = 15,
    obsm: str = 'W_OT', resolution: float = 1):
    """Perform Leiden clustering on the joint embedding.

    Args:
        mdata (mu.MuData): The input data.
        n_neighbors (int, optional): Number of neighbours. Defaults to 15.
        obsm (str, optional): Which obsm field to consider. Defaults to 'W_OT'.
        resolution (float, optional): The Leiden resolution. Defaults to 1.
    """

    # Create an AnnData from the joint embedding.
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)

    # Compute neighbors based on that joint embedding.
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)

    # Perform Leiden clustering.
    sc.tl.leiden(joint_embedding, resolution=resolution)

    # Copy the Leiden labels to the input object.
    mdata.obs['leiden'] = joint_embedding.obs['leiden']

############################## EVALUATE EMBEDDING #############################

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

def knn_score(
    mdata: mu.MuData, obs: str,
    obsm: str = 'W_OT', max_neighbors: int = 15
    ) -> List:
    """Computes the k-NN purity score, for varying numbers of neighbors.

    Args:
        mdata (mu.MuData): Input data
        obs (str): Annotation
        obsm (str, optional): Embedding. Defaults to 'W_OT'.
        max_neighbors (int, optional):
            Maximum number of neighbors. Defaults to 15.

    Returns:
        List[int]: The kNN scores for varying k.
    """

    # Compute euclidean distances on the embedding.
    distances = cdist(mdata.obsm[obsm], mdata.obsm[obsm])

    # Intialize the score to zero.
    s = 0

    # For each cell,
    for i in tqdm(range(mdata.n_obs)):

        # Get the `max_neighbors` nearest neighbors.
        idx = distances[i].argsort()[1:max_neighbors]

        # Check if they have the same label as out cell.
        same_label = np.array(mdata.obs[obs][i] == mdata.obs[obs][idx])

        # This computes the success proportion, with growing # of neighbors.
        s += np.cumsum(same_label)/np.arange(1, max_neighbors)
    
    # Return the score.
    return s / mdata.n_obs

def graph_connectivity(
    mdata: mu.MuData, obs: str, obsm: str, n_neighbors: int = 15) -> float:
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
            conn_comp = get_connected_components(
                adjacency[idx][:,idx], connection='strong')

            # Count the occurences of the components.
            _, counts = np.unique(conn_comp, return_counts=True)

            # The proportion is the largest component
            # over the number of cells in the cluster.
            props.append(counts.max() / idx.shape[0])
        except:
            props.append(0)
            print('Warning: empty component')

    # Return average of the proportions.
    return np.array(props).mean()

def predict_features_corr(
    mdata: mu.MuData, mod: str, n_neighbors: int,
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

        # Append to correlation lists, if the vectors are not empty nor constant
        constant_x = np.any(x[idx] != x[idx][0])
        constant_y = np.any(y[idx] != y[idx][0])
        if np.sum(idx) > 2 and constant_x and constant_y:
            pearson.append(pearsonr(x[idx], y[idx])[0])
            spearman.append(spearmanr(x[idx], y[idx])[0])
    
    return pearson, spearman

def leiden_multi(
    mdata: mu.MuData, n_neighbors: int = 15,
    obsm: str = 'W_OT', obs: str = 'rna:celltype',
    resolutions: Iterable[float] = np.arange(.1, 2.1, .1)):
    """Compute leiden clustering for multiple resultions. Return ARI and NMI.

    Args:
        mdata (mu.MuData): Input data.
        n_neighbors (int, optional): Number of neighbors. Defaults to 15.
        obsm (str, optional): Obsm to use. Defaults to 'W_OT'.
        obs (str, optional): Annotation. Defaults to 'rna:celltype'.
        resolutions (Iterable[float], optional):
            Iterable of resultions. Defaults to np.arange(.1, 2.1, .1).

    Returns: Resolutions, ARIs and NMIs.
    """
    
    # Create an AnnData object with the joint embedding.
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)

    # Initialize the results.
    aris, nmis = [], []

    # Compute neighbors on the joint embedding.
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)

    # For all resolutions,
    for resolution in tqdm(resolutions):

        # Perform Leiden clustering.
        sc.tl.leiden(joint_embedding, resolution=resolution)

        # Compute ARI and NMI
        aris.append(ARI(joint_embedding.obs['leiden'], mdata.obs[obs]))
        nmis.append(NMI(joint_embedding.obs['leiden'], mdata.obs[obs]))
    
    # Return ARI and NMI for various resolutions.
    return resolutions, aris, nmis

def leiden_multi_silhouette(
    mdata: mu.MuData, n_neighbors: int = 15,
    obsm: str = 'W_OT', obs: str = 'rna:celltype',
    resolutions: Iterable[float] = np.arange(.1, 2.1, .1)):
    """Compute leiden clustering for multiple resultions. Return ASW.

    Args:
        mdata (mu.MuData): Input data.
        n_neighbors (int, optional): Number of neighbors. Defaults to 15.
        obsm (str, optional): Obsm to use. Defaults to 'W_OT'.
        obs (str, optional): Annotation. Defaults to 'rna:celltype'.
        resolutions (Iterable[float], optional):
            Iterable of resultions. Defaults to np.arange(.1, 2.1, .1).

    Returns: resolutions and silhouette scores.
    """    
    
    # Create an AnnData object with the joint embedding.
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)

    # Initialize the results.
    sils = []

    # Compute neighbors on the joint embedding.
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)

    # For all resolutions,
    for resolution in tqdm(resolutions):

        # Perform Leiden clustering.
        sc.tl.leiden(joint_embedding, resolution=resolution)

        # Compute silhouette score.
        if len(np.unique(joint_embedding.obs['leiden'])) > 1:
            sils.append(silhouette_score(
                joint_embedding.X,
                joint_embedding.obs['leiden']))
        else:
            sils.append(-1)
    
    # Return silhouette score for various resolutions.
    return resolutions, sils

############################### ANALYSE FACTORS ###############################

def top_features(
    mdata: mu.MuData, mod: str = 'rna', uns: str = 'H_OT',
    dim: int = 0, n_features: int = 5) -> Iterable:
    """Returns the top features for a given modality and latent dimension.

    Args:
        mdata (mu.MuData): The input data
        mod (str, optional): The modality. Defaults to 'rna'.
        uns (str, optional): Where to look for H. Defaults to 'H_OT'.
        dim (int, optional): The latent dimension. Defaults to 0.
        n_features (int, optional): The number of top features. Defaults to 5.

    Returns:
        Iterable: A list of features names.
    """    
    # TODO: put variable names in uns!

    # Get names for highly variable features.
    idx = mdata[mod].var.highly_variable
    var_names = mdata[mod].var_names[idx]

    # Sort them by contribution.
    var_idx = np.argsort(mdata[mod].uns[uns][:,dim])[::-1]

    # Return the top ones.
    return var_names[var_idx[:n_features]].tolist()

def enrich(
    mdata: mu.MuData, mod: str = 'rna',
    uns: str = 'H_OT', n_genes: int = 200,
    sources: Iterable[str] = ['GO:MF', 'GO:CC', 'GO:BP'],
    ordered: bool = True, domain_scope='custom_annotated'):
    """Return Gene Set Enrichment Analysis results for each dimension.

    Args:
        mdata (mu.MuData): Input data.
        mod (str, optional): Modality that contains genes. Defaults to 'rna'.
        uns (str, optional): Name of H matrix. Defaults to 'H_OT'.
        n_genes (int, optional):
            Number of top genes by dimension. Defaults to 200.
        sources (Iterable[str], optional):
            Enrichment sources. Defaults to ['GO:MF', 'GO:CC', 'GO:BP'].
        ordered (bool, optional):
            Make query with ordered genes. Defaults to True.

    Returns:
        Pandas dataframe with the results of the queries, as well
        as average best p_value across dimensions.
    """    
    
    # Initialize ordered genes dictionary.
    ordered_genes = {}

    # Get background genes.
    background = mdata[mod].var.index.tolist()

    # For each dimension,
    for dim in range(mdata[mod].uns[uns].shape[1]):

        # Sort the gene indices by weight.
        idx_sorted = mdata[mod].uns[uns][:,dim].argsort()[::-1]

        if n_genes == 'auto':
            nn = np.sum(np.cumsum(np.sort(mdata[mod].uns[uns][:,dim])) > .05)
        else:
            nn = n_genes

        # Select the `n_genes` highest genes.
        gene_list = mdata[mod].var[mdata[mod].var.highly_variable].index
        gene_list = gene_list[idx_sorted].tolist()[:nn]

        # Input them in the dictionary.
        ordered_genes['dimension ' + str(dim)] = gene_list

    # Make the queries to gProfiler, specifying if genes are ordered.
    enr = sc.queries.enrich(ordered_genes, gprofiler_kwargs={
        'ordered': ordered,
        'sources': sources,
        'domain_scope': domain_scope,
        # 'background': background,
        'no_evidences': True
        })
    
    # Compute the average of the best p_values for each dimension.
    mean_best_p = enr.groupby('query')['p_value'].min().mean()

    # Return the results of the queries and the average best p_value.
    return enr, mean_best_p
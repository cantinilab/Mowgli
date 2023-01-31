from typing import Iterable
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
import anndata as ad
import scanpy as sc
from tqdm import tqdm

############################ BASED ON AN EMBEDDING ############################


def embedding_silhouette_score(
    embedding: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Compute the silhouette score for an embedding.

    Args:
        embedding (np.ndarray):
            The embedding, shape (n_obs, n_latent)
        labels (np.ndarray):
            The labels, shape (n_obs,)
        metric (str, optional):
            The metric on the embedding space. Defaults to "euclidean".

    Returns:
        float: The silhouette score.
    """
    # Check the dimensions of the inputs.
    assert embedding.shape[0] == labels.shape[0]

    # Compute and return the silhouette score.
    return silhouette_score(embedding, labels, metric=metric)


def embedding_leiden_across_resolutions(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int,
    resolutions: Iterable[float] = np.arange(0.1, 2.1, 0.1),
):
    # Create an AnnData object with the joint embedding.
    joint_embedding = ad.AnnData(embedding)

    # Initialize the results.
    aris, nmis = [], []

    # Compute neighbors on the joint embedding.
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)

    # For all resolutions,
    for resolution in tqdm(resolutions):

        # Perform Leiden clustering.
        sc.tl.leiden(joint_embedding, resolution=resolution)

        # Compute ARI and NMI
        aris.append(ARI(joint_embedding.obs["leiden"], labels))
        nmis.append(NMI(joint_embedding.obs["leiden"], labels))

    # Return ARI and NMI for various resolutions.
    return resolutions, aris, nmis


################################ BASED ON A KNN ###############################


def knn_purity_score(knn: np.ndarray, labels: np.ndarray) -> float:
    """Compute the kNN purity score, averaged over all observations.
    For one observation, the purity score is the percentage of
    nearest neighbors that share its label.

    Args:
        knn (np.ndarray):
            The knn, shaped (n_obs, k). The i-th row should contain integers
            representing the indices of the k nearest neighbors.
        labels (np.ndarray):
            The labels, shaped (n_obs)

    Returns:
        float: The purity score.
    """
    # Check the dimensions of the input.
    assert knn.shape[0] == labels.shape[0]

    # Initialize a list of purity scores.
    score = 0

    # Iterate over the observations.
    for i, neighbors in enumerate(knn):

        # Do the neighbors have the same label as the observation?
        matches = labels[neighbors] == labels[i]

        # Add the purity rate to the scores.
        score += np.mean(matches)/knn.shape[0]

    # Return the average purity.
    return score


def knn_prediction_score(
    X: np.ndarray,
    knn: np.ndarray,
    correlation_fn: str = "pearson",
    along: str = "var",
) -> float:
    """Compute the prediction score given the input data and a kNN, averaged
    over observations. For one observation, this is the correlation between
    the observation and the average of its neighbors. In other words, this
    measures the capacity of a kNN to impute missing values.

    Args:
        X (np.ndarray):
            The input data. (n_obs, n_features)
        knn (np.ndarray):
            The knn (n_obs, k). The i-th row contains the indices
            to the k nearest neighbors.
        correlation_fn (str, optional):
            The function to use for correlation, either 'pearson'
            or 'spearman'. Defaults to "pearson".
        along (str, optional):
            Compute along either 'obs' or 'var'. Defaults to "var".

    Returns:
        float: The averaged knn prediction score.
    """
    # Check that the correlation function is correct.
    assert correlation_fn == "pearson" or correlation_fn == "spearman"
    assert along == "obs" or along == "var"

    # Define the correlation function.
    corr = pearsonr if correlation_fn == "pearson" else spearmanr

    # Check that the input's dimensions are correct.
    assert X.shape[0] == knn.shape[0]

    # Initialize the prediction scores.
    scores = []

    # Initialize the prediction
    X_predicted = np.zeros_like(X)

    # Iterate over the observations.
    for i, neighbors in enumerate(knn):

        # Predict cell i from its neighbors.
        X_predicted[i] = np.mean(X[neighbors], axis=0)

    if along == 'var':

        # Iterate over the variables.
        for j in range(X.shape[1]):

            # Add the correlation score.
            scores.append(corr(X_predicted[:, j], X[:, j])[0])
    else:
        
        # Iterate over the obs.
        for i in range(X.shape[0]):

            # Add the correlation score.
            scores.append(corr(X_predicted[i], X[i])[0])

    # Return the prediction scores
    return scores

def avg_knn_prediction_score(
    X: np.ndarray,
    knn: np.ndarray,
    correlation_fn: str = "pearson",
    along: str = "var",
) -> float:
    """Compute the **average** prediction score given the input data and a kNN, averaged
    over observations. For one observation, this is the correlation between
    the observation and the average of its neighbors. In other words, this
    measures the capacity of a kNN to impute missing values.

    Args:
        X (np.ndarray):
            The input data. (n_obs, n_features)
        knn (np.ndarray):
            The knn (n_obs, k). The i-th row contains the indices
            to the k nearest neighbors.
        correlation_fn (str, optional):
            The function to use for correlation, either 'pearson'
            or 'spearman'. Defaults to "pearson".
        along (str, optional):
            Compute along either 'obs' or 'var'. Defaults to "var".

    Returns:
        float: The averaged knn prediction score.
    """
    # Check that the correlation function is correct.
    assert correlation_fn == "pearson" or correlation_fn == "spearman"
    assert along == "obs" or along == "var"

    # Define the correlation function.
    corr = pearsonr if correlation_fn == "pearson" else spearmanr

    # Check that the input's dimensions are correct.
    assert X.shape[0] == knn.shape[0]

    # Initialize the prediction scores.
    scores = 0

    if along == 'var':

        # Initialize the prediction
        X_predicted = np.zeros_like(X)

        # Iterate over the observations.
        for i, neighbors in enumerate(knn):

            # Predict cell i from its neighbors.
            X_predicted[i] = np.mean(X[neighbors], axis=0)

        # Iterate over the variables.
        for j in range(X.shape[1]):

            # Add the correlation score.
            scores += corr(X_predicted[:, j], X[:, j])[0]/X.shape[1]
    else:
        
        # Iterate over the obs.
        for i in range(X.shape[0]):

            # Add the correlation score.
            scores += corr(np.mean(X[knn[i]], axis=0), X[i])[0]/X.shape[0]

    # Return the average prediction score
    return scores

#################################### EMBEDDING TO KNN ####################################


def embedding_to_knn(
    embedding: np.ndarray, k: int = 15, metric: str = "euclidean"
) -> np.ndarray:
    """Convert embedding to knn

    Args:
        embedding (np.ndarray): The embedding (n_obs, n_latent)
        k (int, optional): The number of nearest neighbors. Defaults to 15.
        metric (str, optional): The metric to compute neighbors with. Defaults to "euclidean".

    Returns:
        np.ndarray: The knn (n_obs, k)
    """
    # Initialize the knn graph.
    knn = np.zeros((embedding.shape[0], k), dtype=int)

    # Compute pariwise distances between observations.
    distances = cdist(embedding, embedding, metric=metric)

    # Iterate over observations.
    for i in range(distances.shape[0]):

        # Get the `max_neighbors` nearest neighbors.
        knn[i] = distances[i].argsort()[1 : k + 1]

    # Return the knn graph.
    return knn

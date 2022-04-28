import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import silhouette_score

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
    scores = []

    # Iterate over the observations.
    for i, neighbors in enumerate(knn):

        # Do the neighbors have the same label as the observation?
        matches = labels[neighbors] == labels[i]

        # Add the purity rate to the scores.
        scores.append(np.mean(matches))

    # Return the average purity.
    return np.mean(scores)


def knn_prediction_score(
    X: np.ndarray,
    knn: np.ndarray,
    correlation_fn: str = "pearson",
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

    Returns:
        float: The averaged knn prediction score.
    """
    # Check that the correlation function is correct.
    assert correlation_fn == "pearson" or correlation_fn == "spearman"

    # Define the correlation function.
    corr = pearsonr if correlation_fn == "pearson" else spearmanr

    # Check that the input's dimensions are correct.
    assert X.shape[0] == knn.shape[0]

    # Initialize the prediction scores.
    scores = []

    # Iterate over the observations.
    for i, neighbors in enumerate(knn):

        # Predict cell i from its neighbors.
        x_predicted = np.mean(X[neighbors], axis=0)

        # Check that the prediction has the right shape.
        assert x_predicted.shape[0] == X.shape[1]

        # Add the correlation score.
        scores.append(corr(x_predicted, X[i]))

    # Return the average prediction score
    return np.mean(scores)


############################ BASED ON A CLUSTERING ############################

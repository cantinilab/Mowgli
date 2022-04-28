import numpy as np
from sklearn.metrics import silhouette_score

############################ BASED ON AN EMBEDDING ############################


def silhouette(
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


def purity(knn: np.ndarray, labels: np.ndarray) -> float:
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


############################ BASED ON A CLUSTERING ############################

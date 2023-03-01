from typing import Iterable, List
import torch
from scipy.spatial.distance import cdist
import numpy as np


def reference_dataset(
    X, dtype: torch.dtype, device: torch.device, keep_idx: Iterable
) -> torch.Tensor:
    """Select features, transpose dataset and convert to Tensor.

    Args:
        X (array-like): The input data
        dtype (torch.dtype): The dtype to create
        device (torch.device): The device to create on
        keep_idx (Iterable): The variables to keep.

    Returns:
        torch.Tensor: The reference dataset A.
    """

    # Keep only the highly variable features.
    A = X[:, keep_idx].T

    # Check that the dataset is positive.
    assert (A >= 0).all()

    # If the dataset is sparse, make it dense.
    try:
        A = A.todense()
    except:
        pass

    # Send the matrix `A` to PyTorch.
    return torch.from_numpy(A).to(device=device, dtype=dtype).contiguous()


def compute_ground_cost(
    features,
    cost: str,
    eps: float,
    force_recompute: bool,
    cost_path: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute the ground cost (not lazily!)

    Args:
        features (array-like): A array with the features to compute the cost on.
        cost (str): The function to compute the cost. Scipy distances are allowed.
        force_recompute (bool): Recompute even is there is already a cost matrix saved at the provided path.
        cost_path (str): Where to look for or where to save the cost.
        dtype (torch.dtype): The dtype for the output.
        device (torch.device): The device for the ouput.

    Returns:
        torch.Tensor: The ground cost
    """

    # Initialize the `recomputed variable`.
    recomputed = False

    # If we force recomputing, then compute the ground cost.
    if force_recompute:
        K = cdist(features, features, metric=cost)
        recomputed = True

    # If the cost is not yet computed, try to load it or compute it.
    if not recomputed:
        try:
            K = np.load(cost_path)
        except:
            if cost == "ones":
                K = 1 - np.eye(features.shape[0])
            else:
                K = cdist(features, features, metric=cost)
            recomputed = True

    # If we did recompute the cost, save it.
    if recomputed and cost_path:
        np.save(cost_path, K)

    K = torch.from_numpy(K).to(device=device, dtype=dtype)
    K /= eps * K.max()

    # Compute the kernel K.
    K = torch.exp(-K).to(device=device, dtype=dtype)

    return K


def normalize_tensor(X: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor along columns

    Args:
        X (torch.Tensor): The tensor to normalize.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    return X / X.sum(0)


def entropy(
    X: torch.Tensor, min_one: bool = False, rescale: bool = False
) -> torch.Tensor:
    """Entropy function, :math:`E(X) = \langle X, \log X - 1 \rangle`.

    Args:
        X (torch.Tensor):
            The parameter to compute the entropy of.
        min_one (bool, optional):
            Whether to inclue the :math:`-1` in the formula. Defaults to False.
        rescale (bool, optional):
            Rescale so that the value is between 0 and 1 (when min_one=False).

    Returns:
        torch.Tensor: The entropy of X.
    """
    offset = 1 if min_one else 0
    scale = X.shape[1] * np.log(X.shape[0]) if rescale else 1
    return -torch.sum(X * (torch.nan_to_num(X.log()) - offset)) / scale


def entropy_dual_loss(Y: torch.Tensor) -> torch.Tensor:
    """Compute the Legendre dual of the entropy.

    Args:
        Y (torch.Tensor): The input parameter.

    Returns:
        torch.Tensor: The loss.
    """
    return -torch.logsumexp(Y, dim=0).sum()


def ot_dual_loss(
    A: dict, G: dict, K: dict, eps: float, mod_weights: torch.Tensor, dim=(0, 1)
) -> torch.Tensor:
    """Compute the Legendre dual of the entropic OT loss.

    Args:
        A (dict): The input data.
        G (dict): The dual variable.
        K (dict): The kernel.
        eps (float): The entropic regularization.
        mod_weights (torch.Tensor): The weights per cell and modality.
        dim (tuple, optional): How to sum the loss. Defaults to (0, 1).

    Returns:
        torch.Tensor: The loss
    """

    log_fG = G / eps

    # Compute the non stabilized product.
    scale = log_fG.max(0).values
    prod = torch.log(K @ torch.exp(log_fG - scale)) + scale

    # Compute the dot product with A.
    return eps * torch.sum(mod_weights * A * prod, dim=dim)


def early_stop(history: List, tol: float, nonincreasing: bool = False) -> bool:
    """Based on a history and a tolerance, whether to stop early or not.

    Args:
        history (List):
            The loss history.
        tol (float):
            The tolerance before early stopping.
        nonincreasing (bool, optional):
            When False, throws an error if the loss goes up. Defaults to False.

    Raises:
        ValueError: When the loss goes up.

    Returns:
        bool: Whether to stop early.
    """
    # If we have a nan or infinite, die.
    if len(history) > 0 and not torch.isfinite(history[-1]):
        raise ValueError("Error: Loss is not finite!")

    # If the history is too short, continue.
    if len(history) < 3:
        return False

    # If the next value is worse, stop (not normal!).
    if nonincreasing and (history[-1] - history[-3]) > tol:
        return True

    # If the next value is close enough, stop.
    if abs(history[-1] - history[-2]) < tol:
        return True

    # Otherwise, keep on going.
    return False

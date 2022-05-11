from typing import Dict, Iterable, List
import torch
from scipy.spatial.distance import cdist
import numpy as np

def reference_dataset(
    X, dtype: torch.dtype, device: torch.device,
    keep_idx: Iterable) -> torch.Tensor:
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

    # If the dataset is sparse, make it dense.
    try:
        A = A.todense()
    except:
        pass

    # Send the matrix `A` to PyTorch.
    return torch.from_numpy(A).to(device=device, dtype=dtype).contiguous()


def compute_ground_cost(
    features, cost: str, eps: float,
    force_recompute: bool, cost_path: str,
    dtype: torch.dtype, device: torch.device
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
            if cost == 'ones':
                K = 1 - np.eye(features.shape[0])
            else:
                K = cdist(features, features, metric=cost)
            recomputed = True

    # If we did recompute the cost, save it.
    if recomputed and cost_path:
        np.save(cost_path, K)

    K = torch.from_numpy(K).to(device=device, dtype=dtype)
    K /= eps*K.max()

    # Compute the kernel K
    K = torch.exp(-K).to(device=device, dtype=dtype)
    
    return K


def normalize_tensor(X: torch.Tensor, normalize: str) -> torch.Tensor:
    """Normalize a tensor along columns, rows or nothing.

    Args:
        X (torch.Tensor): The tensor to normalize.
        normalize (str): The direction to normalize on.

    Returns:
        torch.Tensor: The normalized tensor.
    """    
    if normalize == 'cols':
        return X/X.sum(0)
    elif normalize == 'rows':
        return (X.T/X.sum(1)).T
    elif normalize == 'full':
        return X/X.sum()
    else:
        return X/X.sum(0).mean()


def entropy(
    X: torch.Tensor,
    min_one: bool = False,
    rescale: bool = False) -> torch.Tensor:
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
    scale = X.shape[1]*np.log(X.shape[0]) if rescale else 1
    return -torch.sum(X*(torch.nan_to_num(X.log()) - offset))/scale



def entropy_dual_loss(Y: torch.Tensor, normalize: str) -> torch.Tensor:
    """Compute the Legendre dual of the entropy. This depends on the
    normalization constraint.

    Args:
        Y (torch.Tensor): The input parameter.
        normalize (str): The normalization constraint for the input parameter.

    Returns:
        torch.Tensor: The loss.
    """    
    if normalize == 'cols':
        return -torch.logsumexp(Y, dim=0).sum()
    elif normalize == 'rows':
        return -torch.logsumexp(Y, dim=1).sum()
    elif normalize == 'full':
        return -torch.logsumexp(Y, dim=(0,1)).sum()
    else:
        return -torch.exp(Y).sum()


def mass_transported(
    A: dict, G: dict, K: dict,
    eps: float, lbda: float,
    per_cell: bool = False, per_mod: bool = False):
    """Compute the amount of mass transported, i.e. the mass outside of
    the diagonal in the transport plan.

    Args:
        A (dict): The dataset.
        G (dict): The dual variable.
        K (dict): The kernel.
        eps (_type_): The OT entropic regularization.
        lbda (_type_): The unbalanced relaxation.
        per_cell (bool, optional): Whether to return for each cell separately. Defaults to False.
        per_mod (bool, optional): Whether to return for each modality separately. Defaults to False.

    Returns: The mass transported (dictionary of array).
    """    
    score = {} if per_mod else 0
    for mod in A:

        # When lbda tends to infinity, these lines are equivalent.
        if lbda:
            prod = torch.exp(-lbda*torch.log1p(-G[mod]/lbda)/eps)
        else:
            prod = torch.exp(G[mod]/eps)
        prod /= K[mod]@prod

        # Compute per cell or summing everything.
        if per_cell:
            s = torch.sum(A[mod]*prod, dim=0)
        else:
            s = torch.sum(A[mod]*prod)/A[mod].shape[1]
        
        # Compute the modality of summing everything.
        if per_mod:
            score[mod] = 1 - s.detach().cpu().numpy()
        else:
            score += (1 - s.detach().cpu().numpy())/len(A)
        
    return score


def ot_dual_loss(
    A: dict, G: dict, K: dict,
    eps: float, lbda: float,
    mod_weights: torch.Tensor,
    dim=(0, 1)
    ) -> torch.Tensor:
    """Compute the Legendre dual of the entropic unbalanced OT loss.

    Args:
        A (dict): The input data.
        G (dict): The dual variable.
        K (dict): The kernel.
        eps (float): The entropic regularization.
        lbda (float): The unbalanced relaxation.
        mod_weights (torch.Tensor): The weights per cell and modality.
        dim (tuple, optional): How to sum the loss. Defaults to (0, 1).

    Returns:
        torch.Tensor: The loss
    """    

    # When lambda tends to infinity, these lines are quivalent.
    if lbda == None:
        log_fG = G/eps
    else:
        log_fG = -(lbda/eps)*torch.log1p(-G/lbda)

    # Compute the non stabilized product.
    scale = log_fG.max(0).values
    prod = torch.log(K@torch.exp(log_fG - scale)) + scale

    # Compute the dot product with A.
    return eps*torch.sum(mod_weights*A*prod, dim=dim)


def unbalanced_scores(
    A: dict, K: dict, G: dict,
    H: dict, W: torch.Tensor,
    eps: float, lbda: float) -> dict:
    """Compute the "unbalancedness" of the result. The score is between 0
    (completely balanced transport) and 1 (all the possible mass is created).

    Args:
        A (dict): The input data.
        K (dict): The kernel.
        G (dict): The dual variable.
        H (dict): The dictionary.
        W (torch.Tensor): The embeddings.
        eps (float): The entropic regularization.
        lbda (float): The unbalanced relaxation.

    Returns:
        dict: The score for each modality.
    """    

    if lbda == 0:
        return {mod: 0 for mod in H}
    
    scores = {}
    
    for mod in H:
        # For large \lambda, \phi(G) is equal to \exp(G/\epsilon).
        phi_G = torch.exp(-lbda*torch.log1p(-G[mod]/lbda)/eps)
        
        # Compute the second marginal of the transport plan.
        # Ideally it should be close to HW
        B_tilde = phi_G * (K[mod].T @ (A[mod] / (K[mod] @ phi_G)))
        
        # Check the conservation of mass.
        mass_cons = torch.abs(B_tilde.sum(0) - A[mod].sum(0)).mean()
        if mass_cons > 1e-5:
            print('Warning. Check conservation of mass: ', mass_cons)
        
        # The distance between the two measures HW and \tilde B.
        # Smaller is better!
        mass_difference = torch.abs(H[mod]@W - B_tilde).sum(0)
        
        # At most, we'll destroy and create this much mass
        # (in case of disjoint supports).
        # It's a worst case scenario, and probably quite a loose upper bound.
        maximum_error = (A[mod] + H[mod]@W).sum(0)
        
        # A and HW don't necessarily have the same mass, so we need to
        # create or destroy at least this amount.
        minimum_error = torch.abs(A[mod].sum(0) - (H[mod]@W).sum(0))
        
        # This is a score between 0 and 1. 0 means we're in the balanced
        # case. 1 means we destroy or create everything.
        scores[mod] = mass_difference - minimum_error
        scores[mod] /= maximum_error - minimum_error
        scores[mod] = torch.median(scores[mod]).detach()
    
    return scores


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
        raise ValueError('Error: Loss is not finite!')

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
# PyTorch
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.decomposition import PCA

# Form cost matrices
import numpy as np
from scipy.spatial.distance import cdist

# Typing
from typing import List, Set, Dict, Tuple, Optional
from typing import Callable, Iterator, Union, Optional, List

# Biology
import muon as mu

# Progress bar
from tqdm import tqdm

import matplotlib.pyplot as plt

class OTintNMF():
    """Integrative Nonnegative Matrix Factorization with an Optimal Transport loss.

    Args:
        latent_dim (int, optional): Dimension of latent space. Defaults to 15.
        rho_h (float, optional): Entropic regularization parameter for :math:`H`. Defaults to 1e-1.
        rho_w (float, optional): Entropic regularization parameter for :math:`W`. Defaults to 1e-1.
        eps (float, optional): Entropic regularization parameter for the optimal transport loss. Defaults to 5e-2.
        cost (str, optional): Distance function compatible with `scipy.spatial.distance.cdist`, used to compute an empirical cost between features. If a dictionary is passed, different functions can be used for each modality. Defaults to 'correlation'.
        pca (bool, optional): If `True`, the cost is computed on a PCA embedding of the features, of size `latent_dim`. Defaults to False.
    """

    def __init__(self, latent_dim: int = 15, rho_h: float = 1e-1, rho_w: float = 1e-1,
                 eps: float = 5e-2, cost: str = 'correlation', pca: bool = False,
                 cost_path: dict = None):

        # Check that the user-defined parameters are valid.
        assert(latent_dim > 0)
        assert(rho_h > 0)
        assert(rho_w > 0)
        assert(eps > 0)

        # Save arguments as attributes.
        self.latent_dim = latent_dim
        self.rho_h = rho_h
        self.rho_w = rho_w
        self.eps = eps
        self.cost = cost
        self.cost_path = cost_path
        self.pca = pca

        # Initialize the loss histories.
        self.losses_w, self.losses_h, self.losses = [], [], []

        # Initialize the dictionaries containing matrices for each omics.
        self.A, self.H, self.G, self.K = {}, {}, {}, {}

    def init_parameters(self, mdata: mu.MuData, dtype: torch.dtype,
        device: torch.device, force_recompute: bool = False) -> None:
        """Initialize parameters before optimization

        Args:
            mdata (mu.MuData): Input dataset
            dtype (torch.dtype): Data type (e.g. double)
            device (torch.device): Device (e.g. cuda)
        """

        # For each modality,
        for mod in mdata.mod:

            ################ Generate the reference dataset A. ################

            # Select the highly variable features.
            idx = mdata[mod].var['highly_variable'].to_numpy()

            # Keep only the highly variable features.
            self.A[mod] = mdata[mod].X[:,idx]

            # If the dataset is sparse, make it dense.
            try:
                self.A[mod] = self.A[mod].todense()
            except:
                pass
            
            ####################### Compute ground cost #######################

            # If `pca`, then compute the ground cost on PCA embeddings.
            if self.pca: 
                pca = PCA(n_components=self.latent_dim)
                X = pca.fit_transform(self.A[mod].T)
            # Else, compute the cost on the transposed matrix.
            else: 
                X = 1e-6 + self.A[mod].T
            
            # Use the specified cost function to compute ground cost.
            cost = self.cost if isinstance(self.cost, str) else self.cost[mod]
            
            # Initialized the `recomputed variable`.
            recomputed = False

            # If we force recomputing, then compute the ground cost.
            if force_recompute:
                C = cdist(X, X, metric=cost)
                recomputed = True
            
            # If the cost is not yet computed, try to load it or compute it.
            if not recomputed:
                try:
                    C = np.load(self.cost_path[mod])
                    recomputed = False
                except:
                    C = cdist(X, X, metric=cost)
                    recomputed = True
            
            # If we did recompute the cost, save it.
            if recomputed and self.cost_path:
                np.save(self.cost_path[mod], C)
            
            # Normalize the cost with infinity norm.
            C = torch.from_numpy(C).to(device=device, dtype=dtype)
            C /= C.max()

            # Compute the kernel K
            self.K[mod] = torch.exp(-C/self.eps).to(device=device, dtype=dtype)

            ######################## Normalize dataset ########################

            # Add a small value for numerical stability, and normalize `A`.
            self.A[mod] = 1e-6 + self.A[mod].T
            self.A[mod] /= self.A[mod].sum(0)

            # Send the matrix `A` to PyTorch.
            self.A[mod] = torch.from_numpy(self.A[mod]).to(
                device=device, dtype=dtype)

            ####################### Initialize matrices #######################

            # Initialize the factor `H`.
            n_vars = idx.sum()
            self.H[mod] = torch.rand(
                n_vars, self.latent_dim, device=device, dtype=dtype)
            self.H[mod] /= self.H[mod].sum(0)

            # Initialize the dual variable `G`
            self.G[mod] = torch.rand(n_vars, mdata[mod].n_obs,
                requires_grad=True, device=device, dtype=dtype)

        # Initialize the shared factor `W`
        self.W = torch.rand(
            self.latent_dim, mdata.n_obs, device=device, dtype=dtype)
        self.W /= self.W.sum(0)

    def fit_transform(self, mdata: mu.MuData, max_iter_inner: int = 25,
        max_iter: int = 25, device: torch.device = 'cpu', lr: float = 1e-2,
        dtype: torch.dtype = torch.float, tol_inner: float = 1e-5,
        tol_outer: float = 1e-3, optim_name: str = "lbfgs") -> None:
        """Fit the model to the input multiomics dataset, and add the learned
        factors to the Muon object.

        Args:
            mdata (mu.MuData): Input dataset
            max_iter_inner (int, optional): Maximum number of iterations for the inner loop. Defaults to 25.
            max_iter (int, optional): Maximum number of iterations for the outer loop. Defaults to 25.
            device (torch.device, optional): Device to do computations on. Defaults to 'cpu'.
            lr (float, optional): Learning rate. Defaults to 1e-2.
            dtype (torch.dtype, optional): Dtype of tensors. Defaults to torch.float.
            tol_inner (float, optional): Tolerance for the inner loop convergence. Defaults to 1e-5.
            tol_outer (float, optional): Tolerance for the outer loop convergence (more tolerance is advised in the outer loop). Defaults to 1e-3.
            optim_name (str, optional): Name of optimizer. See `build_optimizer`. Defaults to "lbfgs".
        """

        # First, initialize the different parameters.
        self.init_parameters(mdata, dtype=dtype, device=device)

        # Initialize the loss histories.
        self.losses_w, self.losses_h, self.losses = [], [], []

        # Build the optimizer.
        optimizer = self.build_optimizer(
            [self.G[mod] for mod in mdata.mod], lr=lr, optim_name=optim_name)

        # Set up the progress bar.
        pbar = tqdm(total=2*max_iter, position=0, leave=True)

        # This is the main loop, with at most `max_iter` iterations.
        for _ in range(max_iter):

            ############################## W step #############################

            # Optimize the dual variable `G`.
            self.optimize(optimizer=optimizer, loss_fn=self.loss_fn_w,
                max_iter=max_iter_inner, history=self.losses_w,
                tol=tol_inner, pbar=pbar)
            
            # Update the shared factor `W`.
            htgw = 0
            for mod in mdata.mod:
                htgw += self.H[mod].T@self.G[mod]/len(mdata.mod)
            self.W = F.softmin(htgw/self.rho_w, dim=0).detach()
            
            # Update the progress bar.
            pbar.update(1)

            # Save the total dual loss.
            self.losses.append(self.total_dual_loss())

            ############################## H step #############################

            # Optimize the dual variable `G`.
            self.optimize(optimizer=optimizer, loss_fn=self.loss_fn_h,
                max_iter=max_iter_inner, history=self.losses_h,
                tol=tol_inner, pbar=pbar)
            
            # Update the omic specific factors `H[mod]`.
            for mod in mdata.mod:
                self.H[mod] = F.softmin(
                    self.G[mod]@self.W.T/self.rho_h, dim=0).detach()
            
            # Update the progress bar.
            pbar.update(1)

            # Save the total dual loss.
            self.losses.append(self.total_dual_loss())

            # Early stopping
            if self.early_stop(self.losses, tol_outer):
                break

        # Add H and W to the MuData object.
        for mod in mdata.mod:
            mdata[mod].uns['H_OT'] = self.H[mod].cpu().numpy()
        mdata.obsm['W_OT'] = self.W.T.cpu().numpy()

    def build_optimizer(self, params, lr: float, optim_name: str) -> torch.optim.Optimizer:
        """Generates the optimizer

        Args:
            params (Iterable of Tensors): The parameters to be optimized
            lr (float): Learning rate of the optimizer
            optim_name (str): Name of the optimizer, among `'lbfgs'`, `'sgd'`, `'adam'`

        Returns:
            torch.optim.Optimizer: The optimizer
        """
        if optim_name == 'lbfgs':
            return optim.LBFGS(params, lr=lr, history_size=5, max_iter=1, line_search_fn='strong_wolfe')
        elif optim_name == 'sgd':
            return optim.SGD(params, lr=lr)
        elif optim_name == 'adam':
            return optim.Adam(params, lr=lr)

    def optimize(self, optimizer: torch.optim.Optimizer, loss_fn: Callable,
                 max_iter: int, history: List, tol: float, pbar: None) -> None:
        """Optimize the dual variable based on the provided loss function

        Args:
            optimizer (torch.optim.Optimizer): Optimizer used
            loss_fn (Callable): Loss function to optimize
            max_iter (int): Max number of iterations
            history (List): List to update with the values of the loss
            tol (float): Tolerance for the convergence
            pbar (None): `tqdm` progress bar
        """

        # This value will be displayed in the progress bar
        total_loss = self.losses[-1].cpu().numpy() if len(self.losses) > 0 else '?'

        # This is the main optimization loop.
        for i in range(max_iter):
            
            # Define the closure function required by the optimizer.
            def closure():
                optimizer.zero_grad()
                loss = loss_fn()
                loss.backward()
                return loss

            # Add a value to the loss history.
            history.append(closure().detach())

            # Perform an optimization step.
            optimizer.step(closure)

            # Every 10 steps, update the progress bar.
            if i % 10 == 0:
                pbar.set_postfix({
                    'loss': total_loss,
                    'loss_inner': history[-1].cpu().numpy()
                })

            # Attempt early stopping
            if self.early_stop(history, tol):
                break
    
    def early_stop(self, history: List, tol: float) -> bool:
        """Check if the early soptting criterion is valid

        Args:
            history (List): The history of values to check.
            tol (float): Tolerance of the value.

        Returns:
            bool: Whether one can stop early
        """
        if len(history) > 2 and abs(history[-1] - history[-2]) < tol:
            return True
        else:
            return False

    def entropy(self, X: torch.Tensor, min_one: bool = False) -> torch.Tensor:
        """Entropy function, :math:`E(X) = \langle X, \log X - 1 \rangle`.

        Args:
            X (torch.Tensor): The parameter to compute the entropy of.
            min_one (bool, optional): Whether to inclue the :math:`-1` in the formula.. Defaults to False.

        Returns:
            torch.Tensor:  The entropy of X.
        """

        # TODO: KL div can take log input as well! see log_softmax ?

        if min_one:
            # return -torch.nan_to_num(X*(X.log()-1)).sum()
            return F.kl_div(torch.ones_like(X), X, reduction='sum')
        else:
            # return -torch.nan_to_num(X*X.log()).sum()
            return F.kl_div(torch.zeros_like(X), X, reduction='sum')

    def entropy_dual_loss(self, Y: torch.Tensor) -> torch.Tensor:
        """The dual of the entropy function.
        
        Implies a simplex constraint for the entropy,
        and no -1 in its expression!

        Args:
            Y (torch.Tensor): The dual parameter.

        Returns:
            torch.Tensor: The dual entropy loss of Y
        """
        return -torch.logsumexp(Y, dim=0).sum()

    def ot_dual_loss(self, A: torch.Tensor, K: torch.Tensor,
                     Y: torch.Tensor) -> torch.Tensor:
        """The dual optimal transport loss. We omit the constant entropy term.

        Args:
            A (torch.Tensor): The reference dataset
            K (torch.Tensor): The exponentiated ground cost :math:`K=e^{-C/\epsilon}`
            Y (torch.Tensor): The dual parameter

        Returns:
            torch.Tensor: The dual optimal transport loss of Y.
        """
        loss = torch.sum(A*torch.log(K@torch.exp(Y/self.eps)))
        return self.eps*loss

    def total_dual_loss(self) -> torch.Tensor:
        """Compute total dual loss

        Returns:
            torch.Tensor: The loss
        """

        # Initialize the loss to zero.
        loss = 0

        # Recover the modalities (omics).
        modalities = self.A.keys()

        # For each modality,
        for mod in modalities:
            # Add the OT dual loss.
            loss -= self.ot_dual_loss(self.A[mod], self.K[mod], self.G[mod])

            # Add the Lagranger multiplier term.
            loss += ((self.H[mod] @ self.W) * self.G[mod]).sum()

            # Add the `H[mod]` entropy term.
            loss -= self.rho_h*self.entropy(self.H[mod])
        
        # Add the `W` entropy term.
        loss -= len(modalities)*self.rho_w*self.entropy(self.W)

        # Return the full loss.
        return loss.detach()
    
    def loss_fn_h(self) -> torch.Tensor:
        """The loss for the optimization of :math:`H`

        Returns:
            torch.Tensor: The loss
        """
        loss_h = 0
        modalities = self.A.keys()
        for mod in modalities:
            # OT dual loss term
            loss_h += self.ot_dual_loss(
                self.A[mod], self.K[mod], self.G[mod])
            # Entropy dual loss term
            coef = self.rho_h
            loss_h -= coef*self.entropy_dual_loss(-self.G[mod]@self.W.T/coef)
        return loss_h

    def loss_fn_w(self) -> torch.Tensor:
        """Return the loss for the optimization of W

        Returns:
            torch.Tensor: The loss
        """
        loss_w = 0
        htgw = 0
        modalities = self.A.keys()
        for mod in modalities:
            htgw += self.H[mod].T@self.G[mod]
            loss_w += self.ot_dual_loss(
                self.A[mod], self.K[mod], self.G[mod])
        coef = len(modalities)*self.rho_w
        loss_w -= coef*self.entropy_dual_loss(-htgw/coef)
        return loss_w

# PyTorch
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.decomposition import PCA

# Form cost matrices
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
                 eps: float = 5e-2, cost: str = 'correlation', pca: bool = False):

        # Check the user-defined parameters
        assert(latent_dim > 0)
        assert(rho_h > 0)
        assert(rho_w > 0)
        assert(eps > 0)

        # Save args as attributes
        self.latent_dim = latent_dim
        self.rho_h = rho_h
        self.rho_w = rho_w
        self.eps = eps
        self.cost = cost
        self.pca = pca

        # Init other attributes
        self.losses_w, self.losses_h, self.losses = [], [], []
        self.A, self.H, self.G, self.K = {}, {}, {}, {}

    def init_parameters(self, mdata: mu.MuData, dtype: torch.dtype,
        device: torch.device) -> None:
        """Initialize parameters before optimization

        Args:
            mdata (mu.MuData): Input dataset
            dtype (torch.dtype): Data type (e.g. double)
            device (torch.device): Device (e.g. cuda)
        """

        # For each modality
        for mod in mdata.mod:

            # Generate reference dataset A
            idx = mdata[mod].var['highly_variable'].to_numpy()
            self.A[mod] = mdata[mod].X[:,idx]
            try:
                self.A[mod] = self.A[mod].todense()
            except:
                pass
            
            if self.pca:
                X = PCA(n_components=self.latent_dim).fit_transform(self.A[mod].T)
            else:
                X = 1e-6 + self.A[mod].T
            # Compute K
            if isinstance(self.cost, str):
                C = cdist(X, X, metric=self.cost)
            else:
                C = cdist(X, X, metric=self.cost[mod])
            C = torch.from_numpy(C).to(device=device, dtype=dtype)
            C /= C.max()
            self.K[mod] = torch.exp(-C/self.eps).to(device=device, dtype=dtype)

            # Normalize datasets
            self.A[mod] = 1e-6 + self.A[mod].T
            self.A[mod] /= self.A[mod].sum(0)

            # Send A to PyTorch
            self.A[mod] = torch.from_numpy(self.A[mod]).to(
                device=device, dtype=dtype)

            # Init H
            n_vars = idx.sum()
            self.H[mod] = torch.rand(
                n_vars, self.latent_dim, device=device, dtype=dtype)
            self.H[mod] /= self.H[mod].sum(0)

            # Init G
            self.G[mod] = torch.rand(n_vars, mdata[mod].n_obs,
                requires_grad=True, device=device, dtype=dtype)

        # Init W
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

        # Initialization
        self.init_parameters(mdata, dtype=dtype, device=device)
        self.losses_w, self.losses_h, self.losses = [], [], []

        # Building the optimizer
        optimizer = self.build_optimizer(
            [self.G[mod] for mod in mdata.mod], lr=lr, optim_name=optim_name)

        # Progress bar
        pbar = tqdm(total=2*max_iter, position=0, leave=True)

        # Main loop
        for _ in range(max_iter):

            # Optimize W
            self.optimize(optimizer=optimizer, loss_fn=self.loss_fn_w,
                max_iter=max_iter_inner, history=self.losses_w, tol=tol_inner, pbar=pbar)
            # Update W
            htgw = 0
            for mod in mdata.mod:
                htgw += self.H[mod].T@self.G[mod]/len(mdata.mod)
            self.W = F.softmin(htgw/self.rho_w, dim=0).detach()
            # Update progress bar
            pbar.update(1)

            # Save total dual loss, and add it in progress bar
            self.losses.append(self.total_dual_loss())

            # Optimize H
            self.optimize(optimizer=optimizer, loss_fn=self.loss_fn_h,
                max_iter=max_iter_inner, history=self.losses_h, tol=tol_inner, pbar=pbar)
            
            # Update H
            for mod in mdata.mod:
                self.H[mod] = F.softmin(
                    self.G[mod]@self.W.T/self.rho_h, dim=0).detach()
            # Update progress bar
            pbar.update(1)

            # Save total dual loss, and add it in progress bar
            self.losses.append(self.total_dual_loss())

            # Early stopping
            if self.early_stop(self.losses, tol_outer):
                break

        # Add H and W to mdata
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
        if len(self.losses) > 0:
            total_loss = self.losses[-1].cpu().numpy()
        else:
            total_loss = '?'

        for i in range(max_iter):

            def closure():
                optimizer.zero_grad()
                loss = loss_fn()
                loss.backward()
                return loss

            history.append(closure().detach())
            optimizer.step(closure)

            if i % 10 == 0:
                pbar.set_postfix({
                    'loss': total_loss,
                    'loss_inner': history[-1].cpu().numpy()
                })

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
        if min_one:
            return -torch.nan_to_num(X*(X.log()-1)).sum()
        else:
            return -torch.nan_to_num(X*X.log()).sum()

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
        loss = 0
        for mod in self.A:
            loss -= self.ot_dual_loss(self.A[mod], self.K[mod], self.G[mod])
            loss += ((self.H[mod] @ self.W) * self.G[mod]).sum()
            loss -= self.rho_w*self.entropy(self.W)
            loss -= self.rho_h*self.entropy(self.H[mod])
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

# PyTorch
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

# Form cost matrices
from scipy.spatial.distance import cdist

# Typing
from typing import List, Set, Dict, Tuple, Optional
from typing import Callable, Iterator, Union, Optional, List

# Numpy
import numpy as np

# Biology
import scanpy as sc
import anndata as ad
import muon as mu

# Nonnegative least squares
import nn_fac.nnls

# Progress bar
from tqdm import tqdm

import matplotlib.pyplot as plt

class OTintNMF():
    """Optimal Transport Nonnegative Matrix Factorization model

    Parameters
    ----------
    latent_dim : int
        Number of latent dimensions.
    rho_h : float
        Regularization parameter of the entropic positivity term on H
    rho_w : float
        Regularization parameter of the entropic positivity term on W
    eps : float
        Entropic regularization parameter for the Optimal Transport distance.
    cost : str or Dict
        Name of function to compute the ground cost.

    Attributes
    ----------
    latent_dim
    rho_h
    rho_w
    eps
    losses_w
    losses_h
    losses
    cost
    A
    H
    G
    K
    """
    def __init__(self, latent_dim: int = 15, rho_h: float = 1e-1,
                 rho_w: float = 1e-1, eps: float = 5e-2, cost='correlation'):
        # Check the user-defined parameters
        assert(latent_dim > 0)
        assert(rho_h > 0)
        assert(rho_w > 0)
        assert(eps > 0)

        # Save as attributes
        self.latent_dim = latent_dim
        self.rho_h = rho_h
        self.rho_w = rho_w
        self.eps = eps
        self.cost = cost

        # Other attributes
        self.losses_w, self.losses_h, self.losses = [], [], []
        self.A, self.H, self.G, self.K = {}, {}, {}, {}

    def build_optimizer(self, params, lr: float) -> torch.optim.Optimizer:
        """Generates the optimizer

        Parameters
        ----------
        params
            parameters
        lr : float
            Learning rate

        Returns
        -------
        torch.optim.Optimizer
            An optimizer

        """
        return optim.LBFGS(params, lr=lr, history_size=10, max_iter=4)
        # return optim.SGD(params, lr=lr)
        # return optim.Adam(params, lr=lr)

    def entropy(self, X: torch.Tensor, min_one: bool = False) -> torch.Tensor:
        """Entropy function, $E(X) = \langle X, \log X - 1\rangle$.
        The 1 is optional.

        Parameters
        ----------
        X : torch.Tensor
            The parameter to compute the entropy of.
        min_one : bool
            Whether to inclue the $-1$ in the formula.

        Returns
        -------
        torch.Tensor
            The entropy of X.

        """
        if min_one:
            return -torch.nan_to_num(X*(X.log()-1)).sum()
        else:
            return -torch.nan_to_num(X*X.log()).sum()

    def entropy_dual_loss(self, Y: torch.Tensor) -> torch.Tensor:
        """The dual of the entropy function.
        Implies a simplex constraint for the entropy,
        and no -1 in its expression!

        Parameters
        ----------
        Y : torch.Tensor
            The dual parameter.

        Returns
        -------
        torch.Tensor
            The dual entropy loss of Y

        """
        return -torch.logsumexp(Y, dim=0).sum()

    def ot_dual_loss(self, A: torch.Tensor, K: torch.Tensor,
                     Y: torch.Tensor) -> torch.Tensor:
        """The dual optimal transport loss.

        Parameters
        ----------
        A : torch.Tensor
            The reference dataset
        K : torch.Tensor
            The exponentiated ground cost $K=e^{-C/\epsilon}$
        Y : torch.Tensor
            The dual parameter

        Returns
        -------
        torch.Tensor
            The dual optimal transport loss of Y.
        """
        loss = self.entropy(A, min_one=True)
        loss += torch.sum(A*torch.log(K@torch.exp(Y/self.eps)))
        return self.eps*loss

    def early_stop(self, history: List, tol: float) -> bool:
        """Check if the early stopping criterion is valid

        Parameters
        ----------
        history : List
            The history of values to check.
        tol : float
            Tolerance of the value.

        Returns
        -------
        bool
            Whether one can stop early

        """
        if len(history) > 2 and abs(history[-1] - history[-2]) < tol:
            return True
        else:
            return False

    def total_dual_loss(self) -> torch.Tensor:
        """Compute total dual loss

        Returns
        -------
        torch.Tensor
            The loss

        """
        loss = 0
        for mod in self.A:
            loss -= self.ot_dual_loss(self.A[mod], self.K[mod], self.G[mod])
            loss += ((self.H[mod] @ self.W) * self.G[mod]).sum()
            loss -= self.rho_w*self.entropy(self.W)
            loss -= self.rho_h*self.entropy(self.H[mod])
        return loss.detach()/len(self.A.keys())

    def optimize(self, optimizer: torch.optim.Optimizer, loss_fn: Callable,
                 max_iter: int, history: List, tol: float) -> None:
        """Optimize the dual variable based on the provided loss function

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer used
        loss_fn : Callable
            Loss function to optimize
        max_iter : int
            Max number of iterations
        history : List
            List to update with the values of the loss
        tol : float
            Tolerance for the convergence

        """
        for _ in range(max_iter):

            def closure():
                optimizer.zero_grad()
                loss = loss_fn()
                loss.backward()
                return loss

            history.append(closure().detach())
            optimizer.step(closure)

            if self.early_stop(history, tol):
                break

    def init_parameters(self, mdata: mu.MuData, dtype: torch.dtype,
        device: torch.device) -> None:
        """Init parameters before optimization

        Parameters
        ----------
        mdata : mu.MuData
            Input dataset

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

            # Normalize datasets
            self.A[mod] = 1e-6 + self.A[mod].T
            self.A[mod] /= self.A[mod].sum(0)

            # Compute K
            if isinstance(self.cost, str):
                C = cdist(self.A[mod], self.A[mod], metric=self.cost)
            else:
                C = cdist(self.A[mod], self.A[mod], metric=self.cost[mod])
            C = torch.from_numpy(C).to(device=device, dtype=dtype)
            C /= C.max()
            self.K[mod] = torch.exp(-C/self.eps).to(device=device, dtype=dtype)

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

    def loss_fn_h(self) -> torch.Tensor:
        """Return the loss for the optimization of H

        Returns
        -------
        torch.Tensor
            The loss

        """
        loss_h = 0
        modalities = self.A.keys()
        for mod in modalities:
            # OT dual loss term
            loss_h += self.ot_dual_loss(
                self.A[mod], self.K[mod], self.G[mod])
            # Entropy dual loss term
            loss_h -= self.rho_h*self.entropy_dual_loss(
                -self.G[mod]@self.W.T/self.rho_h)
        return loss_h

    def loss_fn_w(self) -> torch.Tensor:
        """Return the loss for the optimization of W

        Returns
        -------
        torch.Tensor
            The loss

        """
        loss_w = 0
        htgw = 0
        modalities = self.A.keys()
        for mod in modalities:
            htgw += self.H[mod].T@self.G[mod]
            loss_w += self.ot_dual_loss(
                self.A[mod], self.K[mod], self.G[mod])
        loss_w -= len(modalities)*self.rho_w*self.entropy_dual_loss(
            -htgw/(self.rho_w*len(modalities)))
        return loss_w

    def fit_transform(self, mdata: mu.MuData, max_iter_inner: int = 25,
        max_iter: int = 25, device: torch.device = 'cpu', lr: float = 1e-2,
        dtype: torch.dtype = torch.float, tol_inner: float = 1e-5,
        tol_outer: float = 1e-3) -> None:
        """Fit the model to the input multiomics dataset, and add the learned
        factors to the Muon object.

        Parameters
        ----------
        mdata : mu.MuData
            Input dataset
        max_iter_inner : int
            Maximum number of iterations for the inner loop
        max_iter : int
            Maximum number of iterations for the outer loop
        device : torch.device
            Device to do computations on
        lr : float
            Learning rate
        dtype : torch.dtype
            Dtype of tensors
        tol_inner : float
            Tolerance for the inner loop convergence
        tol_outer : float
            Tolerance for the outer loop convergence (more tolerance is
            advised in the outer loop)

        """

        # Initialization
        self.init_parameters(mdata, dtype=dtype, device=device)
        self.losses_w, self.losses_h, self.losses = [], [], []

        # Building the optimizer
        optimizer = self.build_optimizer(
            [self.G[mod] for mod in mdata.mod], lr=lr)

        # Progress bar
        pbar = tqdm(total=2*max_iter, position=0, leave=True)

        # Main loop
        for _ in range(max_iter):

            # Optimize H
            self.optimize(optimizer=optimizer, loss_fn=self.loss_fn_h,
                max_iter=max_iter_inner, history=self.losses_h, tol=tol_inner)
            # Update H
            for mod in mdata.mod:
                self.H[mod] = F.softmin(
                    self.G[mod]@self.W.T/self.rho_h, dim=0).detach()
            # Update progress bar
            pbar.update(1)


            # Save total dual loss, and add it in progress bar
            self.losses.append(self.total_dual_loss())
            pbar.set_postfix(loss=self.losses[-1].cpu().numpy())


            # Optimize W
            self.optimize(optimizer=optimizer, loss_fn=self.loss_fn_w,
                max_iter=max_iter_inner, history=self.losses_w, tol=tol_inner)
            # Update W
            htgw = 0
            for mod in mdata.mod:
                htgw += self.H[mod].T@self.G[mod]/len(mdata.mod)
            self.W = F.softmin(htgw/self.rho_w, dim=0).detach()
            # Update progress bar
            pbar.update(1)


            # Save total dual loss, and add it in progress bar
            self.losses.append(self.total_dual_loss())
            pbar.set_postfix(loss=self.losses[-1].cpu().numpy())


            # Early stopping
            if self.early_stop(self.losses, tol_outer):
                break

        # Add H and W to mdata
        for mod in mdata.mod:
            mdata[mod].uns['H_OT'] = self.H[mod]
        mdata.obsm['W_OT'] = self.W.T

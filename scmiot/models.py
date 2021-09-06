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

# Nonnegative least squares
import nn_fac.nnls

# Progress bar
from tqdm import tqdm

import matplotlib.pyplot as plt

class OTintNMF():
    def __init__(self, latent_dim=15, rho_h=1e-1, rho_w=1e-1, lr=1e-2, eps=5e-2, tol=1e-2):
        self.latent_dim = latent_dim

        self.lr = lr

        self.rho_h = rho_h
        self.rho_w = rho_w
        self.eps = eps

        self.tol = tol

    def build_optimizer(self, params, lr):
        #return optim.LBFGS(params, lr=lr, history_size=10, max_iter=4)
        #return optim.SGD(params, lr=lr)
        return optim.Adam(params, lr=lr)

    def entropy(self, X, min_one=False):
        if min_one:
            return -torch.nan_to_num(X*(X.log()-1)).sum()
        else:
            return -torch.nan_to_num(X*X.log()).sum()

    def entropy_dual_loss(self, X):
        # For entropy defined on the simplex!
        # min_one=False
        return -torch.logsumexp(X, dim=0).sum()

    def ot_dual_loss(self, A, K, G):
        loss = self.entropy(A, min_one=True)
        loss += (A*torch.log(K@torch.exp(G/self.eps))).sum()
        return self.eps*loss

    def optimize(self, modalities, n_iter_inner, n_iter, device, K):
        optimizer_h, self.losses_h  = {}, {}
        self.losses_w, self.losses_hw = [], []
        for mod in modalities: # For each modality...
            optimizer_h[mod] = self.build_optimizer([self.GH[mod]], lr=self.lr)
            self.losses_h[mod] = []
        optimizer_w = self.build_optimizer([self.GW[mod] for mod in modalities], lr=self.lr)

        # Progress bar
        pbar = tqdm(total=2*len(modalities)*n_iter, position=0, leave=True)

        # Losses
        loss_h = {}

        # Main loop
        for k in range(n_iter):
            # Dual solver for H_i
            for mod in modalities:
                for _ in range(n_iter_inner):
                    optimizer_h[mod].zero_grad()
                    loss_h[mod] = self.ot_dual_loss(self.A[mod], K[mod], self.GH[mod])
                    loss_h[mod] -= self.rho_h*self.entropy_dual_loss(-self.GH[mod]@self.W.T/self.rho_h)
                    self.losses_h[mod].append(loss_h[mod].detach())
                    loss_h[mod].backward()
                    optimizer_h[mod].step()
                    if len(self.losses_h[mod]) > 2 and abs(self.losses_h[mod][-1] - self.losses_h[mod][-2]) < self.tol:
                        break
                pbar.update(1)

                self.H[mod] = F.softmin(self.GH[mod]@self.W.T/self.rho_h, dim=0).detach()

            # Dual solver for W
            for _ in range(n_iter_inner):
                optimizer_w.zero_grad()
                loss_w = 0
                htgw = 0
                for mod in modalities:
                    htgw += self.H[mod].T@self.GW[mod]
                    loss_w += self.ot_dual_loss(self.A[mod], K[mod], self.GW[mod])
                loss_w -= self.rho_w*self.entropy_dual_loss(-htgw/(len(modalities)*self.rho_w))#*len(modalities)
                self.losses_w.append(loss_w.detach())
                loss_w.backward()
                optimizer_w.step()
                if len(self.losses_w) > 2 and abs(self.losses_w[-1] - self.losses_w[-2]) < self.tol:
                    break
            pbar.update(len(modalities))

            self.losses_hw.append(self.losses_w[-1])
            for mod in modalities:
                self.losses_hw[-1] += self.rho_h*self.entropy(self.H[mod])

            htgw = 0
            for mod in modalities:
                htgw += self.H[mod].T@self.GW[mod]
            self.W = F.softmin(htgw/(len(modalities)*self.rho_w), dim=0).detach()

            pbar.set_postfix(loss=self.losses_hw[-1].cpu().numpy())

    def plot_convergence(self):
        plt.title('Dual loss for HW')
        plt.plot(self.losses_hw)
        plt.show()

        plt.title('Dual losses for H')
        for mod in self.H:
            plt.plot(self.losses_h[mod])
        plt.legend(self.H.keys())
        plt.show()

        plt.title('Dual loss for W')
        plt.plot(self.losses_w)
        plt.show()

    def fit_transform(self, mdata, cost='cosine', n_iter_inner=25, n_iter=25, device='cpu', dtype=torch.float):
        self.A, self.H, self.GH, self.GW, K = {}, {}, {}, {}, {}
        self.cost = cost
        if isinstance(self.cost, str):
            self.cost = {mod:self.cost for mod in mdata.mod}

        for mod in mdata.mod: # For each modality...

            # ... Generate datasets
            self.A[mod] = mdata[mod].X[:,mdata[mod].var['highly_variable'].to_numpy()]
            try:
                self.A[mod] = self.A[mod].todense()
            except:
                pass

            # Normalize datasets
            self.A[mod] = 1e-6 + self.A[mod].T
            self.A[mod] /= self.A[mod].sum(0)

            # Compute K
            C = torch.from_numpy(cdist(self.A[mod], self.A[mod], metric=self.cost[mod])).to(device=device, dtype=dtype)
            C /= C.max()
            K[mod] = torch.exp(-C/self.eps).to(device=device, dtype=dtype)

            # send to PyTorch
            self.A[mod] = torch.from_numpy(self.A[mod]).to(device=device, dtype=dtype)

            # ... Generate H_i
            n_vars = mdata[mod].var['highly_variable'].sum()
            self.H[mod] = torch.rand(n_vars, self.latent_dim, device=device, dtype=dtype)
            self.H[mod] /= self.H[mod].sum(0)

            # ... Generate G_{H_i}
            self.GH[mod] = torch.rand(n_vars, mdata[mod].n_obs, requires_grad=True, device=device, dtype=dtype)

            # ... Generate G_{W_i}
            self.GW[mod] = torch.rand(n_vars, mdata[mod].n_obs, requires_grad=True, device=device, dtype=dtype)

        # Generate W
        self.W = torch.rand(self.latent_dim, mdata.n_obs, device=device, dtype=dtype)
        self.W /= self.W.sum(0)

        self.optimize(modalities=mdata.mod, n_iter_inner=n_iter_inner, n_iter=n_iter, device=device, K=K)

        for mod in mdata.mod:
            mdata[mod].uns['H_OT'] = self.H[mod]
        mdata.obsm['W_OT'] = self.W.T

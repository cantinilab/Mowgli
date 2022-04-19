import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from sklearn.decomposition import PCA

from typing import Callable, List
import muon as mu
from tqdm import tqdm

import utils

class MowgliModel():
    
    def __init__(
        self,
        latent_dim: int = 15,
        use_mod_weight: bool = False,
        rho_h: float = 5e-2, rho_w: float = 5e-2,
        eps: float = 5e-2, lbda: float = None,
        cost: str = 'cosine', pca_cost: bool = False,
        cost_path: dict = None,
        normalize_A: str = 'cols',
        normalize_H: str = 'cols',
        normalize_W: str = 'cols'):

        # Check that the user-defined parameters are valid.
        assert(latent_dim > 0)
        assert(rho_h > 0)
        assert(rho_w > 0)
        assert(eps > 0)
        if lbda != None:
            assert(lbda > 0)
        else:
            assert(normalize_A == 'cols')
            assert(normalize_H == 'cols')
            assert(normalize_W == 'cols')

        # Save arguments as attributes.
        self.latent_dim = latent_dim
        self.rho_h = rho_h
        self.rho_w = rho_w
        self.eps = eps
        self.lbda = lbda
        self.use_mod_weight = use_mod_weight
        self.normalize_H = normalize_H
        self.normalize_W = normalize_W
        self.normalize_A = normalize_A
        self.cost = cost
        self.cost_path = cost_path
        self.pca_cost = pca_cost

        # Create new attributes.
        self.mod_weight = {}

        # Initialize the loss and statistics histories.
        self.losses_w, self.losses_h, self.losses = [], [], []
        self.scores_history = [0]

        # Initialize the dictionaries containing matrices for each omics.
        self.A, self.H, self.G, self.K = {}, {}, {}, {}

    def init_parameters(
        self,
        mdata: mu.MuData,
        dtype: torch.dtype, device: torch.device,
        force_recompute: bool = False
        ) -> None:
        """Initialize the parameters for the model

        Args:
            mdata (mu.MuData): Input dataset
            dtype (torch.dtype): Dtype for the output
            device (torch.device): Device for the output
            force_recompute (bool, optional):
                Where to recompute the cost even if there
                is a matrix precomputed. Defaults to False.
        """

        # Set some attributes.
        self.mod = mdata.mod
        self.n_mod = mdata.n_mod
        self.n_obs = mdata.n_obs
        self.n_var = {}

        # For each modality,
        for mod in self.mod:
            
            ################### Define the modality weights ###################

            if self.use_mod_weight:
                mod_weight = mdata.obs[mod + ':mod_weight'].to_numpy()
                mod_weight = torch.Tensor(mod_weight).reshape(1, -1)
                mod_weight = mod_weight.to(dtype=dtype, device=device)
                self.mod_weight[mod] = mod_weight
            else:
                self.mod_weight[mod] = torch.ones(
                    1, self.n_obs, dtype=dtype, device=device)

            ################ Generate the reference dataset A. ################

            # Select the highly variable features.
            keep_idx = mdata[mod].var['highly_variable'].to_numpy()

            # Make the reference dataset.
            self.A[mod] = utils.reference_dataset(
                mdata[mod].X, dtype, device, keep_idx)
            self.n_var[mod] = self.A[mod].shape[0]
            

            ################# Normalize the reference dataset #################

            # Add a small value for numerical stability, and normalize `A^T`.
            self.A[mod] += 1e-6
            if self.normalize_A == 'cols':
                self.A[mod] /= self.A[mod].sum(0)
            else:
                self.A[mod] /= self.A[mod].sum(0).mean()

            ####################### Compute ground cost #######################

            # Use the specified cost function to compute ground cost.
            cost = self.cost if isinstance(self.cost, str) else self.cost[mod]
            try:
                cost_path = self.cost_path[mod]
            except:
                cost_path = None

            features = 1e-6 + self.A[mod].cpu().numpy()
            if self.pca_cost:
                pca = PCA(n_components=self.latent_dim)
                features = pca.fit_transform(features)

            self.K[mod] = utils.compute_ground_cost(
                features, cost, self.eps, force_recompute,
                cost_path, dtype, device)

            ####################### Initialize matrices #######################

            # Initialize the factor `H`.
            self.H[mod] = torch.rand(
                self.n_var[mod], self.latent_dim, device=device, dtype=dtype)
            
            self.H[mod] = utils.normalize_tensor(self.H[mod], self.normalize_H)

            # Initialize the dual variable `G`
            self.G[mod] = torch.zeros_like(self.A[mod], requires_grad=True)

        # Initialize the shared factor `W`
        self.W = torch.rand(
            self.latent_dim, self.n_obs, device=device, dtype=dtype)
        self.W = utils.normalize_tensor(self.W, self.normalize_W)
        
        del keep_idx, features


    def train(
        self, mdata: mu.MuData,
        max_iter_inner: int = 1_000, max_iter: int = 100,
        device: torch.device = 'cpu', dtype: torch.dtype = torch.float,
        lr: float = 1, optim_name: str = "lbfgs",
        tol_inner: float = 1e-9, tol_outer: float = 1e-3
        ) -> None:
        """Fit the model to the input multiomics dataset, and add the learned
        factors to the Muon object.

        Args:
            mdata (mu.MuData): Input dataset
            max_iter_inner (int, optional):
                Maximum number of iterations for the inner loop.
                Defaults to 1_000.
            max_iter (int, optional):
                Maximum number of iterations for the outer loop.
                Defaults to 100.
            device (torch.device, optional):
                Device to do computations on. Defaults to 'cpu'.
            dtype (torch.dtype, optional):
                Dtype of tensors. Defaults to torch.float.
            lr (float, optional): Learning rate. Defaults to 1e-2.
            optim_name (str, optional):
                Name of optimizer. See `build_optimizer`. Defaults to "lbfgs".
            tol_inner (float, optional):
                Tolerance for the inner loop convergence.
                Defaults to 1e-5.
            tol_outer (float, optional):
                Tolerance for the outer loop convergence (more
                tolerance is advised in the outer loop). Defaults to 1e-3.
        """

        # First, initialize the different parameters.
        self.init_parameters(mdata, dtype=dtype, device=device)
        
        self.lr = lr
        self.optim_name = optim_name

        # Initialize the loss histories.
        self.losses_w, self.losses_h, self.losses = [], [], []

        # Set up the progress bar.
        pbar = tqdm(total=2*max_iter, position=0, leave=True)

        # This is the main loop, with at most `max_iter` iterations.
        try:
            for _ in range(max_iter):


                ############################## W step #########################

                # Optimize the dual variable `G`.
                # for mod in self.G:
                #     nn.init.zeros_(self.G[mod])
                
                self.optimize(
                    loss_fn=self.loss_fn_w,
                    max_iter=max_iter_inner, tol=tol_inner,
                    history=self.losses_h, pbar=pbar,
                    device=device)
                
                # Update the shared factor `W`.
                htgw = 0
                for mod in self.mod:
                    htgw += self.H[mod].T@(self.mod_weight[mod]*self.G[mod])
                coef = np.log(self.latent_dim)/(self.n_mod*self.rho_w)
                if self.normalize_W == 'cols':
                    self.W = F.softmin(coef*htgw.detach(), dim=0)
                else:
                    self.W = torch.exp(-coef*htgw.detach())
                del htgw

                # Update the progress bar.
                pbar.update(1)

                # Save the total dual loss.
                self.losses.append(self.total_dual_loss().cpu().detach())

                if self.lbda:
                    scores = utils.unbalanced_scores(
                        self.A, self.K, self.G,
                        self.H, self.W,
                        self.eps, self.lbda)
                    self.scores_history.append(scores)
                else:
                    scores = utils.mass_transported(
                        self.A, self.G, self.K, self.eps, self.lbda)
                    self.scores_history.append(scores)

                ############################## H step #########################

                # Optimize the dual variable `G`.
                # for mod in self.G:
                #     nn.init.zeros_(self.G[mod])
                
                self.optimize(
                    loss_fn=self.loss_fn_h, device=device,
                    max_iter=max_iter_inner, tol=tol_inner,
                    history=self.losses_h, pbar=pbar)

                # Update the omic specific factors `H[mod]`.
                for mod in self.mod:
                    coef = self.latent_dim*np.log(self.n_var[mod])
                    coef /= self.n_obs*self.rho_h
                    if self.normalize_H == 'cols':
                        self.H[mod] = self.mod_weight[mod]*self.G[mod].detach()
                        self.H[mod] = self.H[mod]@self.W.T
                        self.H[mod] = F.softmin(coef*self.H[mod], dim=0)
                    else:
                        self.H[mod] = self.mod_weight[mod]*self.G[mod].detach()
                        self.H[mod] = self.H[mod]@self.W.T
                        self.H[mod] = torch.exp(-coef*self.H[mod])

                # Update the progress bar.
                pbar.update(1)

                # Save the total dual loss.
                self.losses.append(self.total_dual_loss().cpu().detach())

                # TODO refactor this
                if self.lbda:
                    scores = utils.unbalanced_scores(
                        self.A, self.K, self.G,
                        self.H, self.W,
                        self.eps, self.lbda)
                    self.scores_history.append(scores)
                else:
                    scores = utils.mass_transported(
                        self.A, self.G, self.K, self.eps, self.lbda)
                    self.scores_history.append(scores)

                # Early stopping
                if utils.early_stop(
                    self.losses, tol_outer, nonincreasing=True):
                    break

        except KeyboardInterrupt:
            print('Training interrupted.')

        # Add H and W to the MuData object.
        for mod in self.mod:
            mdata[mod].uns['H_OT'] = self.H[mod].cpu().numpy()
        mdata.obsm['W_OT'] = self.W.T.cpu().numpy()

    def build_optimizer(
        self, params,
        lr: float, optim_name: str
        ) -> torch.optim.Optimizer:
        """Generates the optimizer

        Args:
            params (Iterable of Tensors): The parameters to be optimized
            lr (float): Learning rate of the optimizer
            optim_name (str):
                Name of the optimizer, among `'lbfgs'`, `'sgd'`, `'adam'`

        Returns:
            torch.optim.Optimizer: The optimizer
        """
        if optim_name == 'lbfgs':
            # https://discuss.pytorch.org/t/unclear-purpose-of-max-iter
            # -kwarg-in-the-lbfgs-optimizer/65695
            return optim.LBFGS(
                params, lr=lr,
                history_size=5,
                max_iter=1,
                line_search_fn='strong_wolfe')
        elif optim_name == 'sgd':
            return optim.SGD(params, lr=lr)
        elif optim_name == 'adam':
            return optim.Adam(params, lr=lr)

    def optimize(
        self, loss_fn: Callable,
        max_iter: int, history: List,
        tol: float, pbar: None, device: str) -> None:
        """Optimize the dual variable based on the provided loss function

        Args:
            optimizer (torch.optim.Optimizer): Optimizer used
            loss_fn (Callable): Loss function to optimize
            max_iter (int): Max number of iterations
            history (List): List to update with the values of the loss
            tol (float): Tolerance for the convergence
            pbar (None): `tqdm` progress bar
        """

        optimizer = self.build_optimizer(
            [self.G[mod] for mod in self.G],
            lr=self.lr, optim_name=self.optim_name)

        # This value will be displayed in the progress bar
        if len(self.losses) > 0:
            total_loss = self.losses[-1].cpu().numpy()
        else:
            total_loss = '?'

        # This is the main optimization loop.
        for i in range(max_iter):

            # Define the closure function required by the optimizer.
            def closure():
                optimizer.zero_grad()
                loss = loss_fn()
                loss.backward()
                return loss.detach()

            # Perform an optimization step.
            optimizer.step(closure)

            # Every x steps, update the progress bar.
            if i % 10 == 0:
                # Add a value to the loss history.
                history.append(loss_fn().cpu().detach())
                gpu_mem_alloc = torch.cuda.memory_allocated(device=device)

                pbar.set_postfix({
                    'loss': total_loss,
                    'mass_transported': self.scores_history[-1],
                    'loss_inner': history[-1].cpu().numpy(),
                    'inner_steps': i,
                    'gpu_memory_allocated': gpu_mem_alloc
                })

                # Attempt early stopping
                if utils.early_stop(history, tol):
                    break

    @torch.no_grad()
    def total_dual_loss(self) -> torch.Tensor:
        """Compute total dual loss

        Returns:
            torch.Tensor: The loss
        """

        # Initialize the loss to zero.
        loss = 0

        # Recover the modalities (omics).
        modalities = self.mod

        # For each modality,
        for mod in modalities:

            # Add the OT dual loss.
            loss -= utils.ot_dual_loss(
                self.A[mod], self.G[mod], self.K[mod],
                self.eps, self.lbda,
                self.mod_weight[mod])/self.n_obs

            # Add the Lagrange multiplier term.
            lagrange = self.H[mod]@self.W
            lagrange *= self.mod_weight[mod]*self.G[mod]
            lagrange = lagrange.sum()
            loss += lagrange/self.n_obs

            # Add the `H[mod]` entropy term.
            coef = self.rho_h/(self.latent_dim*np.log(self.n_var[mod]))
            loss -= coef*utils.entropy(self.H[mod], min_one=True)

        # Add the `W` entropy term.
        coef = self.n_mod*self.rho_w/(self.n_obs*np.log(self.latent_dim))
        loss -= coef*utils.entropy(self.W, min_one=True)

        # Return the full loss.
        return loss

    def loss_fn_h(self) -> torch.Tensor:
        """The loss for the optimization of :math:`H`

        Returns:
            torch.Tensor: The loss
        """
        loss_h = 0
        for mod in self.mod:

            # OT dual loss term
            loss_h += utils.ot_dual_loss(
                self.A[mod], self.G[mod], self.K[mod],
                self.eps, self.lbda,
                self.mod_weight[mod])/self.n_obs
            
            # Entropy dual loss term
            coef = self.rho_h/(self.latent_dim*np.log(self.n_var[mod]))
            gwt = self.mod_weight[mod]*self.G[mod]@self.W.T
            gwt /= self.n_obs*coef
            loss_h -= coef*utils.entropy_dual_loss(-gwt, self.normalize_H)
        return loss_h

    def loss_fn_w(self) -> torch.Tensor:
        """Return the loss for the optimization of W

        Returns:
            torch.Tensor: The loss
        """
        loss_w, htgw = 0, 0

        for mod in self.mod:

            # For the entropy dual loss term.
            htgw += self.H[mod].T@(self.mod_weight[mod]*self.G[mod])

            # OT dual loss term.
            loss_w += utils.ot_dual_loss(
                self.A[mod], self.G[mod], self.K[mod],
                self.eps, self.lbda,
                self.mod_weight[mod])/self.n_obs
        
        # Entropy dual loss term.
        coef = self.n_mod*self.rho_w
        coef /= self.n_obs*np.log(self.latent_dim)
        htgw /= coef*self.n_obs
        loss_w -= coef*utils.entropy_dual_loss(-htgw, self.normalize_W)

        del htgw

        # Return the loss.
        return loss_w

# PyTorch
import torch
from torch import nn, optim
import torch.nn.functional as F

# For cost matrices
import numpy as np
from scipy.spatial.distance import cdist

# sklearn
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Typing
from typing import Iterable, List, Set, Dict, Tuple, Optional
from typing import Callable, Iterator, Union, Optional, List

# Biology
import muon as mu

# Progress bar
from tqdm import tqdm

class OTintNMF():
    
    def __init__(
        self, latent_dim: int = 15, rho_h: float = 5e-2, rho_w: float = 5e-2,
        eps: float = 5e-2, cost: str = 'cosine', pca_cost: bool = False,
        cost_path: dict = None, lbda: float = None, normalize_A: str = 'cols', normalize_H: str = 'cols',
        normalize_W: str = 'cols', use_mod_weight: bool = False):
        """Optimal Transport Integrative NMF.

        Args:
            latent_dim (int, optional): Latent dimension of the model. Defaults to 15.
            rho_h (float, optional): Entropic regularization parameter for the disctionary. Defaults to 1e-1.
            rho_w (float, optional): Entropic regularization parameter for the embeddings. Defaults to 1e-1.
            eps (float, optional): Entropic regularization parameter for the Sinkhorn loss. Defaults to 5e-2.
            cost (str, optional): Function to compute the ground cost matrix. Defaults to 'cosine'.
            pca_cost (bool, optional): Whether to compute the ground cost on a PCA reduction instead of full data. Defaults to False.
            cost_path (dict, optional): If specified, save the computed ground cost to this path. Defaults to None.
        """

        # Check that the user-defined parameters are valid.
        assert(latent_dim > 0)
        assert(rho_h > 0)
        assert(rho_w > 0)
        assert(eps > 0)
        if lbda != None:
            assert(lbda > 0)
        else:
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

        # Initialize the loss histories.
        self.losses_w, self.losses_h, self.losses = [], [], []

        # Initialize the dictionaries containing matrices for each omics.
        self.A, self.H, self.G, self.C, self.K = {}, {}, {}, {}, {}
    
    def reference_dataset(self, X, dtype: torch.dtype, device: torch.device,
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
    
    def compute_ground_cost(self, features, cost: str,
        force_recompute: bool, cost_path: str, dtype: torch.dtype,
        device: torch.device) -> torch.Tensor:
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
            C = cdist(features, features, metric=cost)
            recomputed = True

        # If the cost is not yet computed, try to load it or compute it.
        if not recomputed:
            try:
                C = np.load(cost_path)
            except:
                C = cdist(features, features, metric=cost)
                recomputed = True

        # If we did recompute the cost, save it.
        if recomputed and self.cost_path:
            np.save(cost_path, C)

        C = torch.from_numpy(C).to(device=device, dtype=dtype)
        C /= C.max()

        # Compute the kernel K
        K = torch.exp(-C/self.eps).to(device=device, dtype=dtype)

        del C
        
        return K

    def update_latent_dim(self, latent_dim: int) -> None:
        """Change the latent dimension. This allows you to keep the
        previous dual variable as a good initialization for the new
        optimization.

        Args:
            latent_dim (int): The new latent_dimension.
        """

        assert(latent_dim > 0)

        self.latent_dim = latent_dim

        # Initialize the factor `H`.
        for mod in self.H.keys():
            # Get the dimensions, device and dtype for H.
            n_var = self.H[mod].shape[0]
            device = self.H[mod].device
            dtype = self.H[mod].dtype

            # Create a new random H.
            self.H[mod] = torch.rand(
                n_var, self.latent_dim, device=device, dtype=dtype)
            self.H[mod] = self.normalize_tensor(self.H[mod])

        # Initialize the shared factor `W`.
        self.W = torch.rand(
            self.latent_dim, self.W.shape[1], device=device, dtype=dtype)
        self.W = self.normalize_tensor(self.W)
    
    def normalize_tensor(self, X: torch.Tensor, normalize: str):
        if normalize == 'cols':
            return X/X.sum(0)
        elif normalize == 'rows':
            return (X.T/X.sum(1)).T
        elif normalize == 'full':
            return X/X.sum()
        else:
            return X

    @ignore_warnings(category=ConvergenceWarning)
    def init_parameters(
        self, mdata: mu.MuData, dtype: torch.dtype,
        device: torch.device, force_recompute=False) -> None:
        """Initialize the parameters for the model

        Args:
            mdata (mu.MuData): Input dataset
            dtype (torch.dtype): Dtype for the output
            device (torch.device): Device for the output
            force_recompute (bool, optional): Where to recompute the cost even if there is a matrix precomputed. Defaults to False.
        """

        # classical_nmf = NMF(n_components=self.latent_dim, init = "nndsvd", max_iter=1)

        self.mod_weight = {}

        # For each modality,
        for mod in mdata.mod:

            if self.use_mod_weight:
                self.mod_weight[mod] = torch.Tensor(mdata.obs[mod + ':mod_weight'].to_numpy()).to(dtype=dtype, device=device).reshape(1, -1)

            ################ Generate the reference dataset A. ################

            # Select the highly variable features.
            keep_idx = mdata[mod].var['highly_variable'].to_numpy()

            # Make the reference dataset.
            self.A[mod] = self.reference_dataset(
                mdata[mod].X, dtype, device, keep_idx)

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

            self.K[mod] = self.compute_ground_cost(
                features, cost, force_recompute,
                cost_path, dtype, device)

            ################# Normalize the reference dataset #################

            # Add a small value for numerical stability, and normalize `A^T`.
            self.A[mod] += 1e-6
            if self.normalize_A == 'cols':
                self.A[mod] /= self.A[mod].sum(0)
            else:
                self.A[mod] /= self.A[mod].sum(0).mean()


            ####################### Initialize matrices #######################

            # Initialize the factor `H`.
            # self.H[mod] = torch.Tensor(classical_nmf.fit_transform(self.A[mod].cpu())).to(device=device, dtype=dtype)
            self.H[mod] = torch.rand(self.A[mod].shape[0], self.latent_dim, device=device, dtype=dtype)
            self.H[mod] = self.normalize_tensor(self.H[mod], self.normalize_H)

            # Initialize the dual variable `G`
            self.G[mod] = torch.zeros_like(self.A[mod], requires_grad=True)

        # Initialize the shared factor `W`
        # self.W = torch.Tensor(classical_nmf.components_).to(device=device, dtype=dtype)
        self.W = torch.rand(self.latent_dim, self.A[mod].shape[1], device=device, dtype=dtype)
        self.W = self.normalize_tensor(self.W, self.normalize_W)
        
        del keep_idx, features


    def fit_transform(self, mdata: mu.MuData, max_iter_inner: int = 100,
                      max_iter: int = 25, device: torch.device = 'cpu',
                      lr: float = 1, dtype: torch.dtype = torch.float,
                      tol_inner: float = 1e-9, tol_outer: float = 1e-3,
                      optim_name: str = "lbfgs") -> None:
        """Fit the model to the input multiomics dataset, and add the learned
        factors to the Muon object.

        Args:
            mdata (mu.MuData): Input dataset
            max_iter_inner (int, optional): Maximum number of iterations for the inner loop. Defaults to 100.
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
        
        self.lr = lr
        self.optim_name = optim_name

        # Initialize the loss histories.
        self.losses_w, self.losses_h, self.losses = [], [], []

        # Set up the progress bar.
        pbar = tqdm(total=2*max_iter, position=0, leave=True)

        # This is the main loop, with at most `max_iter` iterations.
        try:
            for _ in range(max_iter):


                ############################## W step #############################

                # Optimize the dual variable `G`.
                for mod in self.G:
                    nn.init.zeros_(self.G[mod])
                
                self.optimize(loss_fn=self.loss_fn_w, max_iter=max_iter_inner,
                    history=self.losses_h, tol=tol_inner, pbar=pbar, device=device)
                
                # Update the shared factor `W`.
                htgw = 0
                for mod in mdata.mod:
                    if self.use_mod_weight:
                        htgw += self.H[mod].T@(self.mod_weight[mod]*self.G[mod])/len(mdata.mod)
                    else:
                        htgw += self.H[mod].T@self.G[mod]/len(mdata.mod)
                self.W = torch.exp(-htgw.detach()/self.rho_w)
                self.W = self.normalize_tensor(self.W, self.normalize_W)
                del htgw

                # Update the progress bar.
                pbar.update(1)

                # Save the total dual loss.
                self.losses.append(self.total_dual_loss().cpu().detach())

                ############################## H step #############################

                # Optimize the dual variable `G`.
                for mod in self.G:
                    nn.init.zeros_(self.G[mod])
                
                self.optimize(loss_fn=self.loss_fn_h, max_iter=max_iter_inner,
                    history=self.losses_h, tol=tol_inner, pbar=pbar, device=device)

                # Update the omic specific factors `H[mod]`.
                for mod in mdata.mod:
                    if self.use_mod_weight:
                        self.H[mod] = torch.exp(-((self.mod_weight[mod]*self.G[mod])@self.W.T).detach()/self.rho_h)
                    else:
                        self.H[mod] = torch.exp(-(self.G[mod]@self.W.T).detach()/self.rho_h)
                    self.H[mod] = self.normalize_tensor(self.H[mod], self.normalize_H)

                # Update the progress bar.
                pbar.update(1)

                # Save the total dual loss.
                self.losses.append(self.total_dual_loss().cpu().detach())

                # Early stopping
                if self.early_stop(self.losses, tol_outer, nonincreasing=True):
                    break

        except KeyboardInterrupt:
            print('Training interrupted.')

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
            # https://discuss.pytorch.org/t/unclear-purpose-of-max-iter-kwarg-in-the-lbfgs-optimizer/65695
            return optim.LBFGS(params, lr=lr, history_size=5, max_iter=1, line_search_fn='strong_wolfe')
        elif optim_name == 'sgd':
            return optim.SGD(params, lr=lr)
        elif optim_name == 'adam':
            return optim.Adam(params, lr=lr)

    def optimize(self, loss_fn: Callable,
                 max_iter: int, history: List, tol: float, pbar: None, device: str) -> None:
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
        total_loss = self.losses[-1].cpu().numpy() if len(self.losses) > 0 else '?'

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

                pbar.set_postfix({
                    'loss': total_loss,
                    'loss_inner': history[-1].cpu().numpy(),
                    'inner_steps': i,
                    'gpu_memory_allocated': torch.cuda.memory_allocated(device=device)
                })

                # Attempt early stopping
                if self.early_stop(history, tol):
                    break

    @torch.no_grad()
    def early_stop(self, history: List, tol: float, nonincreasing: bool = False) -> bool:
        """Check if the early stopping criterion is valid

        Args:
            history (List): The history of values to check.
            tol (float): Tolerance of the value.

        Returns:
            bool: Whether one can stop early
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

    @torch.no_grad()
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

    def entropy_dual_loss(self, Y: torch.Tensor, normalize: str) -> torch.Tensor:
        """The dual of the entropy function.

        Args:
            Y (torch.Tensor): The dual parameter.

        Returns:
            torch.Tensor: The dual entropy loss of Y
        """
        if normalize == 'cols':
            return -torch.logsumexp(Y, dim=0).sum()
        elif normalize == 'rows':
            return -torch.logsumexp(Y, dim=1).sum()
        elif normalize == 'full':
            return -torch.logsumexp(Y, dim=(0,1)).sum()
        else:
            return -torch.exp(Y).sum()

    def ot_dual_loss(self, mod: str) -> torch.Tensor:
        """The dual optimal transport loss. We omit the constant entropy term.

        Args:
            A (torch.Tensor): The reference dataset
            K (torch.Tensor): The exponentiated ground cost :math:`K=e^{-C/\epsilon}`
            Y (torch.Tensor): The dual parameter

        Returns:
            torch.Tensor: The dual optimal transport loss of Y.
        """

        if self.lbda == None:
            log_fG = self.G[mod]/self.eps
        else:
            log_fG = -(self.lbda/self.eps)*torch.log1p(-self.G[mod]/self.lbda)

        # Compute the non stabilized product.
        scale = log_fG.max(0).values
        prod = torch.log(self.K[mod]@torch.exp(log_fG - scale)) + scale

        # Compute the dot product with A.
        if self.use_mod_weight:
            loss = self.eps*torch.sum(self.mod_weight[mod]*self.A[mod]*prod)
        else:
            loss = self.eps*torch.sum(self.A[mod]*prod)

        del scale
        del prod
        
        return loss

    @torch.no_grad()
    def total_dual_loss(self) -> torch.Tensor:
        """Compute total dual loss

        Returns:
            torch.Tensor: The loss
        """

        # TODO: add constantsssss

        # Initialize the loss to zero.
        loss = 0

        # Recover the modalities (omics).
        modalities = self.A.keys()

        # For each modality,
        for mod in modalities:

            # Add the OT dual loss.
            loss -= self.ot_dual_loss(mod)

            # Add the Lagrange multiplier term.
            if self.use_mod_weight:
                loss += ((self.H[mod] @ self.W) * (self.mod_weight[mod]*self.G[mod])).sum()
            else:
                loss += ((self.H[mod] @ self.W) * self.G[mod]).sum()

            # Add the `H[mod]` entropy term.
            loss -= self.rho_h*self.entropy(self.H[mod], min_one=True)

        # Add the `W` entropy term.
        loss -= len(modalities)*self.rho_w*self.entropy(self.W, min_one=True)

        # Return the full loss.
        return loss/self.W.shape[1]
    
    def unbalanced_scores(self):

        if self.lbda == 0:
            return {mod: 0 for mod in self.H}
        
        scores = {}
        
        for mod in self.H:
            # For large \lambda, \phi(G) is equal to \exp(G/\epsilon).
            phi_G = torch.exp(-self.lbda*torch.log1p(-self.G[mod]/self.lbda)/self.eps)
            
            # Compute the second marginal of the transport plan. Ideally it should be close to HW
            B_tilde = phi_G * (self.K[mod].T @ (self.A[mod] / (self.K[mod] @ phi_G)))
            
            # Check the conservation of mass.
            mass_cons = torch.abs(B_tilde.sum(0) - self.A[mod].sum(0)).mean()
            if mass_cons > 1e-5:
                print('Warning. Check conservation of mass: ', mass_cons)
            
            # The distance between the two measures HW and \tilde B. Smaller is better!
            mass_difference = torch.abs(self.H[mod]@self.W - B_tilde).sum(0)
            
            # At most, we'll destroy and create this much mass (in case of disjoint supports).
            # It's a worst case scenario, and probably quite a loose upper bound.
            maximum_error = (self.A[mod] + self.H[mod]@self.W).sum(0)
            
            # A and HW don't necessarily have the same mass, so we need to create or destroy at least this amount.
            minimum_error = torch.abs(self.A[mod].sum(0) - (self.H[mod]@self.W).sum(0))
            
            # This is a score between 0 and 1. 0 means we're in the balanced case. 1 means we destroy or create averything.
            scores[mod] = torch.mean((mass_difference - minimum_error)/(maximum_error - minimum_error)).detach()
        
        return scores

    def loss_fn_h(self) -> torch.Tensor:
        """The loss for the optimization of :math:`H`

        Returns:
            torch.Tensor: The loss
        """
        loss_h = 0
        for mod in self.A.keys():
            n = self.A[mod].shape[1]

            # OT dual loss term
            loss_h += self.ot_dual_loss(mod)
            
            # Entropy dual loss term
            coef = self.rho_h
            if self.use_mod_weight:
                loss_h -= coef*self.entropy_dual_loss(-(self.mod_weight[mod]*self.G[mod])@self.W.T/coef, self.normalize_H)
            else:
                loss_h -= coef*self.entropy_dual_loss(-self.G[mod]@self.W.T/coef, self.normalize_H)
        return loss_h/n

    def loss_fn_w(self) -> torch.Tensor:
        """Return the loss for the optimization of W

        Returns:
            torch.Tensor: The loss
        """
        loss_w = 0
        htgw = 0
        for mod in self.A.keys():
            n = self.A[mod].shape[1]

            # For the entropy dual loss term.
            if self.use_mod_weight:
                htgw += self.H[mod].T@(self.mod_weight[mod]*self.G[mod])
            else:
                htgw += self.H[mod].T@self.G[mod]

            # OT dual loss term.
            loss_w += self.ot_dual_loss(mod)
        
        # Entropy dual loss term.
        coef = len(self.A.keys())*self.rho_w
        loss_w -= coef*self.entropy_dual_loss(-htgw/coef, self.normalize_W)

        del htgw

        # Return the loss.
        return loss_w/n

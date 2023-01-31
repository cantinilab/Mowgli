import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from sklearn.decomposition import PCA

from typing import Callable, List
import muon as mu
from tqdm import tqdm

from mowgli import utils


class MowgliModel:
    def __init__(
        self,
        latent_dim: int = 15,
        use_mod_weight: bool = False,
        h_regularization: float = 5e-2,
        w_regularization: float = 5e-2,
        eps: float = 5e-2,
        cost: str = "cosine",
        pca_cost: bool = False,
        cost_path: dict = None,
    ):
        """Initialize the Mowgli model, which performs integrative NMF with an
        Optimal Transport loss.

        Args:
            latent_dim (int, optional):
                The latent dimension of the model. Defaults to 15.
            use_mod_weight (bool, optional):
                Whether to use a different weight for each modality and each
                cell. If `True`, the weights are expected in the `mod_weight`
                obs field of each modality. Defaults to False.
            h_regularization (float, optional):
                The entropy parameter for the dictionary. Small values mean
                sparse dictionaries. Defaults to 5e-2.
            w_regularization (float, optional):
                The entropy parameter for the embedding. Small values mean
                sparse vectors. Defaults to 5e-2.
            eps (float, optional):
                The entropy parameter for epsilon transport. Large values
                decrease importance of individual genes. Defaults to 5e-2.
            cost (str, optional):
                The function used to compute an emprical ground cost. All
                metrics from Scipy's `cdist` are allowed. Defaults to 'cosine'.
            pca_cost (bool, optional):
                If True, the emprical ground cost will be computed on PCA
                embeddings rather than raw data. Defaults to False.
            cost_path (dict, optional):
                Will look for an existing cost as a `.npy` file at this
                path. If not found, the cost will be computed then saved
                there. Defaults to None.
        """

        # Check that the user-defined parameters are valid.
        assert latent_dim > 0
        assert w_regularization > 0
        assert eps > 0

        if isinstance(h_regularization, dict):
            for mod in h_regularization:
                assert h_regularization[mod] > 0
        else:
            assert h_regularization > 0

        # Save arguments as attributes.
        self.latent_dim = latent_dim
        self.h_regularization = h_regularization
        self.w_regularization = w_regularization
        self.eps = eps
        self.use_mod_weight = use_mod_weight
        self.cost = cost
        self.cost_path = cost_path
        self.pca_cost = pca_cost

        # Create new attributes.
        self.mod_weight = {}

        # Initialize the loss and statistics histories.
        self.losses_w, self.losses_h, self.losses = [], [], []
        self.scores_history = [0]  # TODO: change this

        # Initialize the dictionaries containing matrices for each omics.
        self.A, self.H, self.G, self.K = {}, {}, {}, {}

    def init_parameters(
        self,
        mdata: mu.MuData,
        dtype: torch.dtype,
        device: torch.device,
        force_recompute: bool = False,
        normalize_rows: bool = False,
    ) -> None:
        """Initialize parameters based on input data.

        Args:
            mdata (mu.MuData):
                The input MuData object.
            dtype (torch.dtype):
                The dtype to work with.
            device (torch.device):
                The device to work on.
            force_recompute (bool, optional):
                Whether to recompute the ground cost. Defaults to False.
        """

        # Set some attributes.
        self.mod = mdata.mod
        self.n_mod = mdata.n_mod
        self.n_obs = mdata.n_obs
        self.n_var = {}

        if not isinstance(self.h_regularization, dict):
            self.h_regularization = {mod: self.h_regularization for mod in self.mod}

        # For each modality,
        for mod in self.mod:

            # Define the modality weights.
            if self.use_mod_weight:
                mod_weight = mdata.obs[mod + ":mod_weight"].to_numpy()
                mod_weight = torch.Tensor(mod_weight).reshape(1, -1)
                mod_weight = mod_weight.to(dtype=dtype, device=device)
                self.mod_weight[mod] = mod_weight
            else:
                self.mod_weight[mod] = torch.ones(
                    1, self.n_obs, dtype=dtype, device=device
                )

            # Select the highly variable features.
            keep_idx = mdata[mod].var["highly_variable"].to_numpy()

            # Make the reference dataset.
            self.A[mod] = utils.reference_dataset(mdata[mod].X, dtype, device, keep_idx)
            self.n_var[mod] = self.A[mod].shape[0]

            # Normalize the reference dataset, and add a small value
            # for numerical stability.
            self.A[mod] += 1e-6
            if normalize_rows:
                mean_row_sum = self.A[mod].sum(1).mean()
                self.A[mod] /= self.A[mod].sum(1).reshape(-1, 1) * mean_row_sum
            self.A[mod] /= self.A[mod].sum(0)

            # Determine which cost function to use.
            cost = self.cost if isinstance(self.cost, str) else self.cost[mod]
            try:
                cost_path = self.cost_path[mod]
            except:
                cost_path = None

            # Define the features that the ground cost will be computed on.
            features = 1e-6 + self.A[mod].cpu().numpy()
            if self.pca_cost:
                pca = PCA(n_components=self.latent_dim)
                features = pca.fit_transform(features)

            # Compute ground cost, using the specified cost function.
            self.K[mod] = utils.compute_ground_cost(
                features, cost, self.eps, force_recompute, cost_path, dtype, device
            )

            # Initialize the matrices `H`, which should be normalized.
            self.H[mod] = torch.rand(
                self.n_var[mod], self.latent_dim, device=device, dtype=dtype
            )
            self.H[mod] = utils.normalize_tensor(self.H[mod])

            # Initialize the dual variable `G`
            self.G[mod] = torch.zeros_like(self.A[mod], requires_grad=True)

        # Initialize the shared factor `W`, which should be normalized.
        self.W = torch.rand(self.latent_dim, self.n_obs, device=device, dtype=dtype)
        self.W = utils.normalize_tensor(self.W)

        # Clean up.
        del keep_idx, features

    def train(
        self,
        mdata: mu.MuData,
        max_iter_inner: int = 1_000,
        max_iter: int = 100,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float,
        lr: float = 1,
        optim_name: str = "lbfgs",
        tol_inner: float = 1e-9,
        tol_outer: float = 1e-3,
        normalize_rows: bool = False,
    ) -> None:
        """Train the Mowgli model on an input MuData object.

        Args:
            mdata (mu.MuData):
                The input MuData object.?
            max_iter_inner (int, optional):
                How many iterations for the inner optimization loop
                (optimizing H, or W). Defaults to 1_000.
            max_iter (int, optional):
                How many interations for the outer optimization loop (how
                many successive optimizations of H and W). Defaults to 100.
            device (torch.device, optional):
                The device to work on. Defaults to 'cpu'.
            dtype (torch.dtype, optional):
                The dtype to work with. Defaults to torch.float.
            lr (float, optional):
                The learning rate for the optimizer. The default is set
                for LBFGS and should be changed otherwise. Defaults to 1.
            optim_name (str, optional):
                The optimizer to use (`lbfgs`, `sgd` or `adam`). LBFGS
                is advised, but requires more memory. Defaults to "lbfgs".
            tol_inner (float, optional):
                The tolerance for the inner iterations before early stopping.
                Defaults to 1e-9.
            tol_outer (float, optional):
                The tolerance for the outer iterations before early stopping.
                Defaults to 1e-3.
        """

        # First, initialize the different parameters.
        self.init_parameters(
            mdata,
            dtype=dtype,
            device=device,
            normalize_rows=normalize_rows,
        )

        self.lr = lr
        self.optim_name = optim_name

        # Initialize the loss histories.
        self.losses_w, self.losses_h, self.losses = [], [], []

        # Set up the progress bar.
        pbar = tqdm(total=2 * max_iter, position=0, leave=True)

        # This is the main loop, with at most `max_iter` iterations.
        try:
            for _ in range(max_iter):

                # Perform the `W` optimization step.
                self.optimize(
                    loss_fn=self.loss_fn_w,
                    max_iter=max_iter_inner,
                    tol=tol_inner,
                    history=self.losses_h,
                    pbar=pbar,
                    device=device,
                )

                # Update the shared factor `W`.
                htgw = 0
                for mod in self.mod:
                    htgw += self.H[mod].T @ (self.mod_weight[mod] * self.G[mod])
                coef = np.log(self.latent_dim) / (self.n_mod * self.w_regularization)

                self.W = F.softmin(coef * htgw.detach(), dim=0)

                # Clean up.
                del htgw

                # Update the progress bar.
                pbar.update(1)

                # Save the total dual loss and statistics.
                self.losses.append(self.total_dual_loss().cpu().detach())
                scores = utils.mass_transported(self.A, self.G, self.K, self.eps)
                self.scores_history.append(scores)

                # Perform the `H` optimization step.
                self.optimize(
                    loss_fn=self.loss_fn_h,
                    device=device,
                    max_iter=max_iter_inner,
                    tol=tol_inner,
                    history=self.losses_h,
                    pbar=pbar,
                )

                # Update the omic specific factors `H[mod]`.
                for mod in self.mod:
                    coef = self.latent_dim * np.log(self.n_var[mod])
                    coef /= self.n_obs * self.h_regularization[mod]

                    self.H[mod] = self.mod_weight[mod] * self.G[mod].detach()
                    self.H[mod] = self.H[mod] @ self.W.T
                    self.H[mod] = F.softmin(coef * self.H[mod], dim=0)

                # Update the progress bar.
                pbar.update(1)

                # Save the total dual loss and statistics.
                self.losses.append(self.total_dual_loss().cpu().detach())
                scores = utils.mass_transported(self.A, self.G, self.K, self.eps)
                self.scores_history.append(scores)

                # Early stopping
                if utils.early_stop(self.losses, tol_outer, nonincreasing=True):
                    break

        except KeyboardInterrupt:
            print("Training interrupted.")

        # Add H and W to the MuData object.
        for mod in self.mod:
            mdata[mod].uns["H_OT"] = self.H[mod].cpu().numpy()
        mdata.obsm["W_OT"] = self.W.T.cpu().numpy()

    def build_optimizer(
        self, params, lr: float, optim_name: str
    ) -> torch.optim.Optimizer:
        """Generates the optimizer. The PyTorch LBGS implementation is
        parametrized following the discussion in https://discuss.pytorch.org/
        t/unclear-purpose-of-max-iter-kwarg-in-the-lbfgs-optimizer/65695.

        Args:
            params (Iterable of Tensors):
                The parameters to be optimized.
            lr (float):
                Learning rate of the optimizer.
            optim_name (str):
                Name of the optimizer, among `'lbfgs'`, `'sgd'`, `'adam'`

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        if optim_name == "lbfgs":
            return optim.LBFGS(
                params,
                lr=lr,
                history_size=5,
                max_iter=1,
                line_search_fn="strong_wolfe",
            )
        elif optim_name == "sgd":
            return optim.SGD(params, lr=lr)
        elif optim_name == "adam":
            return optim.Adam(params, lr=lr)

    def optimize(
        self,
        loss_fn: Callable,
        max_iter: int,
        history: List,
        tol: float,
        pbar,
        device: str,
    ) -> None:
        """Optimize a fiven function.

        Args:
            loss_fn (Callable): The function to optimize.
            max_iter (int): The maximum number of iterations.
            history (List): A list to append the losses to.
            tol (float): The tolerance before early stopping.
            pbar (A tqdm progress bar): The progress bar.
            device (str): The device to work on.
        """

        # Build the optimizer.
        optimizer = self.build_optimizer(
            [self.G[mod] for mod in self.G], lr=self.lr, optim_name=self.optim_name
        )

        # This value will be initially be displayed in the progress bar
        if len(self.losses) > 0:
            total_loss = self.losses[-1].cpu().numpy()
        else:
            total_loss = "?"

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

                # Populate the progress bar.
                pbar.set_postfix(
                    {
                        "loss": total_loss,
                        "mass_transported": self.scores_history[-1],
                        "loss_inner": history[-1].cpu().numpy(),
                        "inner_steps": i,
                        "gpu_memory_allocated": gpu_mem_alloc,
                    }
                )

                # Attempt early stopping.
                if utils.early_stop(history, tol):
                    break

    @torch.no_grad()
    def total_dual_loss(self) -> torch.Tensor:
        """Compute the total dual loss. This is only used by the user and for,
        early stopping, not by the optimization algorithm.

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
            loss -= (
                utils.ot_dual_loss(
                    self.A[mod],
                    self.G[mod],
                    self.K[mod],
                    self.eps,
                    self.mod_weight[mod],
                )
                / self.n_obs
            )

            # Add the Lagrange multiplier term.
            lagrange = self.H[mod] @ self.W
            lagrange *= self.mod_weight[mod] * self.G[mod]
            lagrange = lagrange.sum()
            loss += lagrange / self.n_obs

            # Add the `H[mod]` entropy term.
            coef = self.h_regularization[mod] / (
                self.latent_dim * np.log(self.n_var[mod])
            )
            loss -= coef * utils.entropy(self.H[mod], min_one=True)

        # Add the `W` entropy term.
        coef = (
            self.n_mod * self.w_regularization / (self.n_obs * np.log(self.latent_dim))
        )
        loss -= coef * utils.entropy(self.W, min_one=True)

        # Return the full loss.
        return loss

    def loss_fn_h(self) -> torch.Tensor:
        """Computes the loss for the update of `H`.

        Returns:
            torch.Tensor: The loss.
        """
        loss_h = 0
        for mod in self.mod:

            # OT dual loss term
            loss_h += (
                utils.ot_dual_loss(
                    self.A[mod],
                    self.G[mod],
                    self.K[mod],
                    self.eps,
                    self.mod_weight[mod],
                )
                / self.n_obs
            )

            # Entropy dual loss term
            coef = self.h_regularization[mod] / (
                self.latent_dim * np.log(self.n_var[mod])
            )
            gwt = self.mod_weight[mod] * self.G[mod] @ self.W.T
            gwt /= self.n_obs * coef
            loss_h -= coef * utils.entropy_dual_loss(-gwt)

            # Clean up.
            del gwt

        # Return the loss.
        return loss_h

    def loss_fn_w(self) -> torch.Tensor:
        """Return the loss for the optimization of W

        Returns:
            torch.Tensor: The loss
        """
        loss_w, htgw = 0, 0

        for mod in self.mod:

            # For the entropy dual loss term.
            htgw += self.H[mod].T @ (self.mod_weight[mod] * self.G[mod])

            # OT dual loss term.
            loss_w += (
                utils.ot_dual_loss(
                    self.A[mod],
                    self.G[mod],
                    self.K[mod],
                    self.eps,
                    self.mod_weight[mod],
                )
                / self.n_obs
            )

        # Entropy dual loss term.
        coef = self.n_mod * self.w_regularization
        coef /= self.n_obs * np.log(self.latent_dim)
        htgw /= coef * self.n_obs
        loss_w -= coef * utils.entropy_dual_loss(-htgw)

        # Clean up.
        del htgw

        # Return the loss.
        return loss_w

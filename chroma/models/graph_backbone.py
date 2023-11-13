# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for generating protein backbone structure via diffusion.
"""

from types import SimpleNamespace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from chroma.data.xcs import validate_XC
from chroma.layers import basic, graph
from chroma.layers.structure import backbone, diffusion, transforms
from chroma.models.graph_design import BackboneEncoderGNN
from chroma.utility.model import load_model as utility_load_model


class GraphBackbone(nn.Module):
    """Graph-based backbone generation for protein complexes.

    GraphBackbone parameterizes a generative model of the backbone coordinates
    of protein complexes.

    Args:
        See documention of `layers.structure.protein_graph.ProteinFeatureGraph`,
        `graph.GraphNN`, `layers.structure.backbone.GraphBackboneUpdate` and
        `layers.structure.diffusion.DiffusionChainCov` for more details on
        hyperparameters.

    Inputs:
        X (Tensor): Backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
        C (LongTensor): Chain map with shape `(num_batch, num_residues)`.

    Outputs:
        neglogp (Tensor): Sum of `neglogp_S` and `neglogp_chi`.
    """

    def __init__(
        self,
        dim_nodes: int = 128,
        dim_edges: int = 128,
        num_neighbors: int = 30,
        node_features: Tuple = (("internal_coords", {"log_lengths": True}),),
        edge_features: Tuple = (
            "distances_2mer",
            "orientations_2mer",
            "distances_chain",
        ),
        num_layers: int = 3,
        dropout: float = 0.1,
        node_mlp_layers: int = 1,
        node_mlp_dim: Optional[int] = None,
        edge_update: bool = True,
        edge_mlp_layers: int = 1,
        edge_mlp_dim: Optional[int] = None,
        skip_connect_input: bool = False,
        mlp_activation: str = "softplus",
        decoder_num_hidden: int = 512,
        graph_criterion: str = "knn",
        graph_random_min_local: int = 20,
        backbone_update_method: str = "neighbor",
        backbone_update_iterations: int = 1,
        backbone_update_num_weights: int = 1,
        backbone_update_unconstrained: bool = True,
        use_time_features: bool = True,
        time_feature_type: str = "t",
        time_log_feature_scaling: float = 0.05,
        noise_schedule: str = "log_snr",
        noise_covariance_model: str = "brownian",
        noise_beta_min: float = 0.2,
        noise_beta_max: float = 70.0,
        noise_log_snr_range: Tuple[float] = (-7.0, 13.5),
        noise_complex_scaling: bool = False,
        loss_scale: float = 10.0,
        loss_scale_ssnr_cutoff: float = 0.99,
        loss_function: str = "squared_fape",
        checkpoint_gradients: bool = False,
        prediction_type: str = "X0",
        num_graph_cycles: int = 1,
        **kwargs,
    ):
        """Initialize GraphBackbone network."""
        super(GraphBackbone, self).__init__()

        # Save configuration in kwargs
        self.kwargs = locals()
        self.kwargs.pop("self")
        for key in list(self.kwargs.keys()):
            if key.startswith("__") and key.endswith("__"):
                self.kwargs.pop(key)
        args = SimpleNamespace(**self.kwargs)

        # Important global options
        self.dim_nodes = args.dim_nodes
        self.dim_edges = args.dim_edges

        # Encoder GNN process backbone
        self.num_graph_cycles = args.num_graph_cycles
        self.encoders = nn.ModuleList(
            [
                BackboneEncoderGNN(
                    dim_nodes=args.dim_nodes,
                    dim_edges=args.dim_edges,
                    num_neighbors=args.num_neighbors,
                    node_features=args.node_features,
                    edge_features=args.edge_features,
                    num_layers=args.num_layers,
                    node_mlp_layers=args.node_mlp_layers,
                    node_mlp_dim=args.node_mlp_dim,
                    edge_update=args.edge_update,
                    edge_mlp_layers=args.edge_mlp_layers,
                    edge_mlp_dim=args.edge_mlp_dim,
                    mlp_activation=args.mlp_activation,
                    dropout=args.dropout,
                    skip_connect_input=args.skip_connect_input,
                    graph_criterion=args.graph_criterion,
                    graph_random_min_local=args.graph_random_min_local,
                    checkpoint_gradients=checkpoint_gradients,
                )
                for i in range(self.num_graph_cycles)
            ]
        )

        self.backbone_updates = nn.ModuleList(
            [
                backbone.GraphBackboneUpdate(
                    dim_nodes=args.dim_nodes,
                    dim_edges=args.dim_edges,
                    method=args.backbone_update_method,
                    iterations=args.backbone_update_iterations,
                    num_transform_weights=args.backbone_update_num_weights,
                    unconstrained=args.backbone_update_unconstrained,
                )
                for i in range(self.num_graph_cycles)
            ]
        )

        self.use_time_features = args.use_time_features
        self.time_feature_type = args.time_feature_type
        self.time_log_feature_scaling = time_log_feature_scaling
        if self.use_time_features:
            self.time_features = basic.FourierFeaturization(
                d_input=1, d_model=dim_nodes, trainable=False, scale=16.0
            )

        self.noise_perturb = diffusion.DiffusionChainCov(
            noise_schedule=args.noise_schedule,
            beta_min=args.noise_beta_min,
            beta_max=args.noise_beta_max,
            log_snr_range=args.noise_log_snr_range,
            covariance_model=args.noise_covariance_model,
            complex_scaling=args.noise_complex_scaling,
        )
        self.noise_schedule = self.noise_perturb.noise_schedule
        method = "symeig"
        self.loss_scale = args.loss_scale
        self.loss_scale_ssnr_cutoff = loss_scale_ssnr_cutoff
        self.loss_function = args.loss_function
        self.prediction_type = args.prediction_type
        self._loss_eps = 1e-5

        self.loss_diffusion = diffusion.ReconstructionLosses(
            diffusion=self.noise_perturb, rmsd_method=method, loss_scale=args.loss_scale
        )

        if self.prediction_type.startswith("scale"):
            self.mlp_W = graph.MLP(
                dim_in=args.dim_nodes, num_layers_hidden=args.node_mlp_layers, dim_out=1
            )

        # Wrap sampling functions
        _X0_func = lambda X, C, t: self.denoise(X, C, t)
        self.sample_sde = lambda C, **kwargs: self.noise_perturb.sample_sde(
            _X0_func, C, **kwargs
        )
        self.sample_baoab = lambda C, **kwargs: self.noise_perturb.sample_baoab(
            _X0_func, C, **kwargs
        )
        self.sample_ode = lambda C, **kwargs: self.noise_perturb.sample_ode(
            _X0_func, C, **kwargs
        )
        self.estimate_metrics = lambda X, C, **kwargs: self.loss_diffusion.estimate_metrics(
            _X0_func, X, C, **kwargs
        )
        self.estimate_elbo = lambda X, C, **kwargs: self.noise_perturb.estimate_elbo(
            _X0_func, X, C, **kwargs
        )
        self.estimate_pseudoelbo_X = lambda X, C, **kwargs: self.noise_perturb.estimate_pseudoelbo_X(
            _X0_func, X, C, **kwargs
        )

    def _time_features(self, t):
        h = {"t": lambda: t, "log_snr": lambda: self.noise_schedule.log_SNR(t)}[
            self.time_feature_type
        ]()

        if "log" in self.time_feature_type:
            h = self.time_log_feature_scaling * h

        time_h = self.time_features(h[:, None, None])
        return time_h

    @validate_XC()
    def denoise(
        self,
        X: torch.Tensor,
        C: torch.Tensor,
        t: Optional[Union[float, torch.Tensor]] = None,
        return_geometry: bool = False,
    ):
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float().to(X.device)
        if t.shape == torch.Size([]):
            t = t.unsqueeze(-1)

        time_h = self._time_features(t) if self.use_time_features else None
        node_h = time_h
        edge_h, edge_idx, mask_ij = [None] * 3

        # Normalize minimum average C-alpha distances
        X_update = X

        for i in range(self.num_graph_cycles):
            # Encode as graph
            node_h, edge_h, edge_idx, mask_i, mask_ij = self.encoders[i](
                X_update,
                C,
                node_h_aux=node_h,
                edge_h_aux=edge_h,
                edge_idx=edge_idx,
                mask_ij=mask_ij,
            )
            # Update backbone
            X_update, R_ji, t_ji, logit_ji = self.backbone_updates[i](
                X_update, C, node_h, edge_h, edge_idx, mask_i, mask_ij
            )

        # Shrink towards the input
        if time_h is None:
            time_h = torch.zeros(
                [node_h.shape[0], 1, node_h.shape[2]], device=node_h.device
            )
        if self.prediction_type == "scale":
            scale_shift = self.mlp_W(time_h)
            ssnr = self.noise_perturb.noise_schedule.SSNR(t)
            logit_bias = torch.logit(torch.sqrt(1 - ssnr))
            scale = torch.sigmoid(scale_shift + logit_bias[:, None, None])[..., None]
            X_update = scale * X_update + (1 - scale) * X
        elif self.prediction_type == "scale_cutoff":
            # Scale below a given hard-coded noise floor cutoff
            scale_shift = self.mlp_W(time_h)
            ssnr = self.noise_perturb.noise_schedule.SSNR(t)
            logit_bias = torch.logit(torch.sqrt(1 - ssnr))
            scale = torch.sigmoid(scale_shift + logit_bias[:, None, None])[..., None]

            # Skip connect for values of alpha close to 1
            skip = (1 - scale) * (ssnr > self.loss_scale_ssnr_cutoff).float().reshape(
                scale.shape
            )
            X_update = skip * X + (1 - skip) * X_update

        if not return_geometry:
            return X_update
        else:
            return X_update, R_ji, t_ji, logit_ji, edge_idx, mask_ij

    @validate_XC(all_atom=False)
    def _debug_plot_denoising_geometry(self, X, C, t=None):
        """Debug plots for analyzing denoising geometry"""
        if t is None:
            X_noise, t = self.noise_perturb(X, C)
        else:
            X_noise = self.noise_perturb(X, C, t=t)

        # Compute denoised geometry
        (
            X_denoise,
            R_ji_pred,
            t_ji_pred,
            logit_ji_pred,
            edge_idx,
            mask_ij,
        ) = self.denoise(X_noise, C, t, return_geometry=True)

        # Featurize other inputs and outpus
        R_ji_native, t_ji_native = self.backbone_updates[0]._inner_transforms(
            X, C, edge_idx
        )
        R_ji_noise, t_ji_noise = self.backbone_updates[0]._inner_transforms(
            X_noise, C, edge_idx
        )
        R_ji_denoise, t_ji_denoise = self.backbone_updates[0]._inner_transforms(
            X_denoise, C, edge_idx
        )

        R_ji = torch.cat([R_ji_native, R_ji_noise, R_ji_pred, R_ji_denoise], 0)
        t_ji = torch.cat([t_ji_native, t_ji_noise, t_ji_pred, t_ji_denoise], 0)
        logit_ji = torch.cat([mask_ij, mask_ij, logit_ji_pred[:, :, :, 0], mask_ij], 0)
        edge_idx = edge_idx.expand([4, -1, -1])
        from matplotlib import pyplot as plt

        transforms._debug_plot_transforms(R_ji, t_ji, logit_ji, edge_idx, mask_ij)
        plt.show()
        return X_denoise, X_noise

    @validate_XC(all_atom=False)
    def forward(
        self,
        X: torch.Tensor,
        C: torch.Tensor,
        t: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ):
        # If all atom structure is passed, discard side chains
        X = X[:, :, :4, :] if X.size(2) == 14 else X

        # Sample perturbed structure
        if t is None:
            X_t, t = self.noise_perturb(X, C)
        else:
            X_t = self.noise_perturb(X, C, t=t)

        X0_pred, R_ji_pred, t_ji_pred, logit_ji_pred, edge_idx, mask_ij = self.denoise(
            X_t, C, t, return_geometry=True
        )

        losses = self.loss_diffusion(X0_pred, X, C, t)

        # Per complex weights
        weights = (C > 0).float().sum(-1)

        ssnr = self.noise_perturb.noise_schedule.SSNR(t)
        prob_ssnr = self.noise_perturb.noise_schedule.prob_SSNR(ssnr)
        importance_weights = 1 / prob_ssnr

        _importance_weight = lambda h: h * importance_weights.reshape(
            [-1] + [1] * (len(h.shape) - 1)
        )
        _weighted_avg = lambda h: (weights * _importance_weight(h)).sum() / (
            weights.sum() + self._loss_eps
        )
        # Interresidue geometry predictions agreement
        if self.backbone_updates[0].method != "local":
            R_ij_mse, t_ij_mse = self.backbone_updates[0]._transform_loss(
                R_ji_pred, t_ji_pred, X, C, edge_idx, mask_ij
            )
            losses["batch_translate_mse"] = _weighted_avg(
                t_ij_mse / (self.loss_scale ** 2)
            )
            losses["batch_rotate_mse"] = _weighted_avg(R_ij_mse)
            losses["batch_transform_mse"] = (
                losses["batch_translate_mse"] + losses["batch_rotate_mse"]
            )

        losses_extend = {}
        for k, v in losses.items():
            if "elbo" in k:
                losses_extend[k.replace("elbo", "neg_elbo")] = -v
        losses.update(losses_extend)
        return losses


def load_model(
    weight_file: str,
    device: str = "cpu",
    strict: bool = False,
    strict_unexpected: bool = False,
    verbose: bool = True,
) -> GraphBackbone:
    """Load model `GraphBackbone`

    Args:
        weight_file (str): The destination path of the model weights to load.
            Compatible with files saved by `save_model`.
        device (str, optional): Pytorch device specification, e.g. `'cuda'` for
        GPU. Default is `'cpu'`.
        strict (bool): Whether to require that the keys match between the
            input file weights and the model created from the parameters stored
            in the model kwargs.
        strict_unexpected (bool): Whether to require that there are no
            unexpected keys when loading model weights, as distinct from the
            strict option which doesn't allow for missing keys either. By
            default, we use this option rather than strict for ease of
            development when adding model features.
        verbose (bool, optional): Show outputs from download and loading.
            Default True.

    Returns:
        model (GraphBackbone): Instance of `GraphBackbone` with loaded weights.
    """
    return utility_load_model(
        weight_file,
        GraphBackbone,
        device=device,
        strict=strict,
        strict_unexpected=strict_unexpected,
        verbose=verbose,
    )

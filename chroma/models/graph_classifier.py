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

"""Models for generating protein sequence and side chain conformations
given backbones. These can be used for sequence design and packing.
"""


from types import SimpleNamespace

import torch
import torch.nn as nn

from chroma.data.xcs import validate_XC
from chroma.layers import basic
from chroma.layers.attention import AttentionChainPool
from chroma.layers.basic import NodeProduct, NoOp
from chroma.layers.graph import MLP, MaskedNorm
from chroma.layers.structure import diffusion
from chroma.models.graph_design import BackboneEncoderGNN
from chroma.utility.model import load_model as utility_load_model


class GraphClassifier(nn.Module):
    """Graph-based protein classification

    Args:
        See documention of `structure.protein_graph.ProteinFeatureGraph`,
        and `graph.GraphNN` for more details.

    Inputs:
        X (Tensor): Backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
        C (LongTensor): Chain map with shape `(num_batch, num_residues)`.
        O (Tensor) (optional): One-hot sequence tensor of shape `(num_batch, num_residues)`

    Outputs:
        node_h (Tensor): residue-based representations that can be used to project various classification predictions
    """

    def __init__(
        self,
        dim_nodes=128,
        dim_edges=128,
        num_neighbors=30,
        node_features=(("internal_coords", {"log_lengths": True}),),
        edge_features=["random_fourier_2mer", "orientations_2mer", "distances_chain"],
        num_layers=3,
        dropout=0.1,
        node_mlp_layers=1,
        node_mlp_dim=None,
        edge_update=True,
        edge_mlp_layers=1,
        edge_mlp_dim=None,
        skip_connect_input=False,
        mlp_activation="softplus",
        graph_criterion="knn",
        graph_random_min_local=20,
        use_time_features=True,
        noise_schedule="log_snr",
        noise_beta_min=0.2,
        noise_beta_max=70.0,
        checkpoint_gradients=False,
        class_config={},
        out_mlp_layers=2,
        noise_covariance_model="globular",
        noise_log_snr_range=(-7.0, 13.5),
        time_feature_type="t",
        time_log_feature_scaling=0.05,
        fourier_scale=16.0,
        zero_grad_fix=False,
        **kwargs,
    ):
        """Initialize GraphBackbone network."""
        super().__init__()

        # Save configuration in kwargs
        self.kwargs = locals()
        self.kwargs.pop("self")
        for key in list(self.kwargs.keys()):
            if key.startswith("__") and key.endswith("__"):
                self.kwargs.pop(key)
        args = SimpleNamespace(**self.kwargs)

        self.class_config = class_config
        # Important global options
        self.dim_nodes = args.dim_nodes
        self.dim_edges = args.dim_edges
        self.mlp_activation = args.mlp_activation
        self.zero_grad_fix = zero_grad_fix

        if "random_fourier_2mer" in args.edge_features:
            index = args.edge_features.index("random_fourier_2mer")
            args.edge_features.pop(index)
            args.edge_features.append(
                (
                    "random_fourier_2mer",
                    {
                        "dim_embedding": args.dim_edges,
                        "trainable": False,
                        "scale": args.fourier_scale,
                    },
                )
            )

        # Encoder GNN process backbone
        self.encoder = BackboneEncoderGNN(
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

        self.time_feature_type = args.time_feature_type
        self.time_log_feature_scaling = time_log_feature_scaling

        self.use_time_features = use_time_features
        if self.use_time_features:
            self.time_features = basic.FourierFeaturization(
                d_input=1, d_model=dim_nodes, trainable=False, scale=16.0
            )

        self.sequence_embedding = nn.Embedding(20, dim_nodes)

        self.noise_perturb = diffusion.DiffusionChainCov(
            noise_schedule=args.noise_schedule,
            beta_min=args.noise_beta_min,
            beta_max=args.noise_beta_max,
            log_snr_range=args.noise_log_snr_range,
            covariance_model=args.noise_covariance_model,
        )

        self._init_heads(class_config, dim_nodes, out_mlp_layers, dropout)
        self.condition_sequence_frequency = 0.3

    def _init_heads(self, class_config, dim_nodes, out_mlp_layers, dropout):
        self.heads = {"chain": {}, "first_order": {}, "second_order": {}, "complex": {}}

        for label, config in class_config.items():
            group = config["level"]
            if label == "is_interface" or label == "contact":
                dim_out = 1
            else:
                dim_out = len(config["tokens"])
            if group == "chain":
                pool = AttentionChainPool(8, dim_nodes)
            elif group == "complex":
                raise NotImplementedError
            elif group == "second_order":
                pool = NoOp()
            else:
                pool = NoOp()

            if group != "second_order":
                if self.zero_grad_fix:
                    node_norm_layer = MaskedNorm(
                        dim=1, num_features=dim_nodes, affine=True, norm="layer"
                    )
                    mlp = MLP(
                        dim_nodes,
                        dim_hidden=None,
                        dim_out=dim_out,
                        num_layers_hidden=out_mlp_layers,
                        activation=self.mlp_activation,
                        dropout=dropout,
                    )
                    head = nn.Sequential(node_norm_layer, mlp)
                else:
                    mlp = MLP(
                        dim_nodes,
                        dim_hidden=None,
                        dim_out=dim_out,
                        num_layers_hidden=out_mlp_layers,
                        activation="relu",
                        dropout=dropout,
                    )
                    head = mlp
            else:
                head = nn.Sequential(nn.Linear(dim_nodes, 16), NodeProduct(16, 1))

            self.heads[group][label] = head, pool
            self.add_module(f"{label}_head", head)
            if pool is not None:
                self.add_module(f"{label}_pool", pool)

    def _time_features(self, t):
        h = {
            "t": lambda: t,
            "log_snr": lambda: self.noise_perturb.noise_schedule.log_SNR(t),
        }[self.time_feature_type]()

        if "log" in self.time_feature_type:
            h = self.time_log_feature_scaling * h

        time_h = self.time_features(h[:, None, None])
        return time_h

    @validate_XC()
    def encode(self, X, C, O=None, t=None):
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float().to(X.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0)

        node_h = self._time_features(t)

        if O is not None:
            if (not self.training) or (
                torch.rand(1,).item() < self.condition_sequence_frequency
            ):
                node_h = node_h + O @ self.sequence_embedding.weight

        edge_h, edge_idx, mask_ij = [None] * 3
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encoder(
            X,
            C,
            node_h_aux=node_h,
            edge_h_aux=edge_h,
            edge_idx=edge_idx,
            mask_ij=mask_ij,
        )

        return node_h, edge_h, edge_idx, mask_i, mask_ij

    @validate_XC()
    def gradient(
        self, X, C, t, label, mask=None, value=None, O=None, scale=1.0, max_norm=None
    ):
        """
        Args:
            X (torch.tensor): (batch, num_res, 4, 3) or (batch, num_res, 14, 3)
            C (torch.tensor): (batch, num_res)
            t (float): 0 < t <= 1
            label (string): class label to condition on, chosen from `self.class_config.keys()`
            mask (torch.tensor): (optional) bool tensor of shape (batch, num_res) for first order scores, (batch, num_chains) for
                                 chain-based scores, and (batch, num_res, num_res) for second order scores. The order of
                                 your score can be determined by inspecting self.class_config[label]['level']
            value (string): (optional) the token-based representation of the value you would like to condition `label` on,
                            you can select options from `self.class_config[label]['tokens']` for all scores except `is_interface`
                            or `contact` for which you should leave a `value` of None.
            O (torch.tensor): one-hot sequence tensor of size (batch, num_res, num_alphabet)
            scale (float): scale factor to multiply gradient by
            max_norm (float): if not None, the maximum norm of the gradient (set grad = max_norm * (grad / grad.norm()) if grad.norm() > max_norm)
        """
        self.eval()
        _bak = self.encoder.checkpoint_gradients
        self.encoder.checkpoint_gradients = False

        level = self.class_config[label]["level"]
        head, pool = self.heads[level][label]
        with torch.enable_grad():
            X.requires_grad = True
            node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X, C, O, t)

            if level == "chain":
                node_h, c_mask = pool(node_h, C)
                c_mask = c_mask
            elif level == "first_order":
                c_mask = C > 0
            elif level == "second_order":
                c_mask = (C > 0).unsqueeze(-2) & (C > 0).unsqueeze(-1)

            node_h = head(node_h)

            if mask is not None:
                c_mask = mask & c_mask

            if self.class_config[label]["loss"] == "ce":
                neglogp = node_h.log_softmax(dim=-1).mul(-1)
            else:
                neglogp = node_h.sigmoid().log().mul(-1)

            index = (
                self.class_config[label]["tokenizer"][value] if value is not None else 0
            )
            neglogp = neglogp[..., index][c_mask].sum()
            neglogp.backward()
            grad = scale * X.grad

            if max_norm is not None:
                if grad.norm() > max_norm:
                    grad = max_norm * (grad / grad.norm())

        self.encoder.checkpoint_gradients = _bak
        return grad

    @validate_XC(all_atom=False)
    def forward(self, X, C, O=None, **kwargs):
        # Sample perturbed structure
        X_perturb, t = self.noise_perturb(X, C)
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X_perturb, C, O, t)
        return node_h, edge_h


def load_model(
    weight_file: str,
    device: str = "cpu",
    strict: bool = False,
    strict_unexpected: bool = True,
    verbose: bool = True,
) -> GraphClassifier:
    """Load model `GraphClassifier`

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

    Returns:
        model (GraphClassifier): Instance of `GraphClassifier` with loaded weights.
    """
    return utility_load_model(
        weight_file,
        GraphClassifier,
        device=device,
        strict=strict,
        strict_unexpected=strict_unexpected,
        verbose=verbose,
    )

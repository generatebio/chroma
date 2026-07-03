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
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from chroma import constants
from chroma.data.xcs import validate_XC
from chroma.layers import complexity, graph
from chroma.layers.structure import diffusion, potts, protein_graph, sidechain
from chroma.layers.structure.protein_graph_allatom import (
    EdgeSidechainsDirect,
    NodeChiRBF,
)
from chroma.models.graph_design_parts import (
    BackboneEncoderGNN,
    NodePredictorChi,
    NodePredictorS,
    ProteinTraversalSpatial,
    SidechainDecoderGNN,
)
from chroma.models.model_configs import (
    GraphDesignConfig,
    build_model_config,
    config_to_kwargs,
)
from chroma.utility.model import load_model as utility_load_model


class GraphDesign(nn.Module):
    """Graph-based sequence design and sidechain packing.

    Given a fixed backbone, a GraphDesign model yields probabilities of residue type
    and angles by position. It encodes backbones with a `BackboneEncoderGNN`
    and then autoregressively factorizes the joint distribution of
    sequence and sidechain conformations given these graph embeddings.
    Optional first order marginal and Potts sequence decoders are also available.

    Some `GraphDesign` models are trained in a diffusion-aware mannner
    to model sequence likelihoods given a noised structure and a particular time point
    along a forwards diffusion process.

    Args:
        See documention of `structure.protein_graph.ProteinFeatureGraph`,
        and `graph.GraphNN` for more details.

        dim_nodes (int): Hidden dimension of node tensors of underlying GNNs.
        dim_edges (int): Hidden dimension of edge tensors of underlying GNNs.
        num_neighbors (int): Number of neighbors per node for underlying GNNs.
        node_features (tuple): List of node feature specifications for
            structure encoder. Features can be given as strings or as
            dictionaries.
        edge_features (tuple): List of edge feature specifications for
            structure encoder. Features can be given as strings or as
            dictionaries.
        sequence_embedding (str): How to represent sequence when decoding.
            Currently the only option is `linear`.
        sidechain_embedding (str): How to represent chi angles when decoding.
            Options include `chi_linear` for a simple linear layer, `chi_rbf`
            for a featurization based on smooth binning of chi angles,
            `X_direct` which directly encodes the all-atom coordinates using
            random Fourier features, and `mixed_chi_X` which uses both the
            featurizations of `chi_rbf` and of `X_direct`.
        sidechains (bool): Whether to use a joint sequence/sidechain
            autoregressive model to decode the backbones.
        num_layers (int): Number of layers of underlying GNNs. Can be overridden
            for the structure encoder by `num_layers_encoder`.
        num_layers_encoder (int, optional): Number of layers for structure
            encoder GNN.
        dropout (float): Dropout fraction used for all encoders and decoders
            except for the marginal sequence likelihood decoder in
            `decoder_S_marginals`.
        node_mlp_layers (int): Number of hidden layers for node update function
            of underlying GNNs.
        node_mlp_dim (int, optional): Dimension of hidden layers for node update
            function of underlying GNNs, defaults to match output dimension.
        edge_update (bool): Whether to include an edge update step in the GNNs.
        edge_mlp_layers (int): Number of hidden layers for edge update function
            of underlying GNNs.
        edge_mlp_dim (int, optional): Dimension of hidden layers for edge update
            function of underlying GNNs, defaults to match output dimension.
        skip_connect_input (bool): Whether to include skip connections between
            layers of underlying GNNs.
        mlp_activation (str): MLP nonlinearity function, `relu` or `softplus`
            accepted.
        num_alphabet (int): Number of possible residues for sequence decoder.
        num_chi_bins (int): Number of chi bins for smooth binning of chi angles
            used when `sidechain_embedding` is `chi_rbf` or `mixed_chi_X`.
        decoder_num_hidden (int): Dimension of hidden decoder layers.
        label_smoothing (float): Level of smoothing to apply to sequence and
            sidechain labels.
        separate_packing (bool): If True, then autoregressively factorize
            sequence and sidechains in two stages where the full sequence is predicted
            before all of the chi angles. Otherwise an interleaved factorization
            will be used that autoregressively predicts both the residue identity
            and chi angles in an alternating manner. Default is True.
        graph_criterion (str): Graph criterion for structure encoder, defines
            how neighbors are chosen. See
            `chroma.models.graph_design.BackboneEncoderGNN` for
            allowed values.
        graph_random_min_local (int): Minimum number of neighbors in GNN that
            come from local neighborhood, before random neighbors are chosen.
        graph_attentional (bool): Currently unused, previously used for
            experimental GNN attention mechanism.
        graph_num_attention_heads (int): Currently unused, previously used for
            experimental GNN attention mechanism.
        predict_S_marginals (bool): Whether to train marginal sequence decoder.
        predict_S_potts (bool): Whether to train Potts sequence decoder.
        potts_parameterization (str): How to parametrize Potts sequence decoder,
            see `chroma.layer.structure.potts` for allowed values.
        potts_num_factors (int, optional): Number of factors to use for Potts
            sequence decoder.
        potts_symmetric_J (bool): Whether to force J tensor of Potts model to be
            symmetric.
        noise_schedule (str, optional): Noise schedule for mapping between
            diffusion time and noise level, see
            chroma.layers.structure.diffusion.DiffusionChainCov for allowed
            values. If not set, model should only be provided with denoised
            backbones.
        noise_covariance_model (str): Covariance mode for mapping between
            diffusion time and noise level, see
            chroma.layers.structure.diffusion.DiffusionChainCov for allowed
            values.
        noise_complex_scaling (bool): Whether to scale noise for complexes.
        noise_beta_range (Tuple[float, float]): Minimum and maximum noise levels
            for noise schedule.
        noise_log_snr_range (Tuple[float, float]): Range of log signal-to-noise
            ratio for noising.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_atoms, 3)`.
        C (torch.LongTensor): Chain map with shape `(num_batch, num_residues)`.
        S (torch.LongTensor): Sequence tensor with shape
            `(num_batch, num_residues)`.
        t (torch.Tensor, optional): Diffusion timesteps corresponding to noisy
            input backbones, of shape `(num_batch)`. Use zeros when passing
            structures without noise.
        sample_noise (bool, optional): Whether to apply noise to input
            backbones.
        permute_idx (torch.LongTensor, optional): Permutation tensor for fixing
            the autoregressive decoding order `(num_batch, num_residues)`. If
            `None` (default), a random decoding order will be generated.
        priority (torch.Tensor, optional): Priority values for constraining
            residue orderings with shape `(num_batch, num_residues)`.
            If residues are assigned to integer-valued groups, the sampled
            permutation will be ordered such that all residues within a lower-
            valued priority group will occur before residues with higher-valued
            priority assignments.

    Outputs (dict):
        logp_S (torch.Tensor): Sequence log likelihoods per residue with shape
            `(num_batch, num_residues)`.
        logp_chi (torch.Tensor): Chi angle Log likelihoods per residue with
            shape `(num_batch, num_residues, 4)`.
        logp_S_marginals (torch.Tensor, optional): Sequence log likelihoods
            per residue from marginal decoder with shape
            `(num_batch, num_residues)`.
        logp_S_potts (torch.Tensor, optional): Sequence log likelihoods per
            residue from Potts decoder with shape
            `(num_batch, num_residues)`.
        chi (torch.Tensor): Chi angles with shape
            `(num_batch, num_residues, 4)`.
        mask_chi (torch.Tensor): Chi angle mask with shape
            `(num_batch, num_residues, 4)`.
        node_h_chi (torch.Tensor): Node features used for predicting chi
            angles with shape `(num_batch, num_residues, dim_nodes)`.
        mask_i (torch.Tensor): Node mask with shape
            `(num_batch, num_residues)`.
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.
        mask_ij_causal (torch.Tensor): Causal edge mask for autoregressive
            decoding with shape `(num_batch, num_nodes, num_neighbors)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_residues, num_neighbors)`.
        permute_idx (torch.LongTensor, optional): Permutation tensor that was
            used for the autoregressive decoding order with shape
            `(num_batch, num_residues)`.
        X_noise (torch.Tensor): Noised structure coordinates with shape
            `(num_batch, num_residues, num_atoms, 3)`.
    """

    def __init__(
        self,
        dim_nodes: int = 128,
        dim_edges: int = 128,
        num_neighbors: int = 30,
        node_features: tuple = (("internal_coords", {"log_lengths": True}),),
        edge_features: tuple = (
            "distances_2mer",
            "orientations_2mer",
            "distances_chain",
        ),
        sequence_embedding: str = "linear",
        sidechain_embedding: str = "chi_rbf",
        sidechains: bool = True,
        num_layers: int = 3,
        num_layers_encoder: Optional[int] = None,
        dropout: float = 0.1,
        node_mlp_layers: int = 1,
        node_mlp_dim: Optional[int] = None,
        edge_update: bool = True,
        edge_mlp_layers: int = 1,
        edge_mlp_dim: Optional[int] = None,
        skip_connect_input: bool = False,
        mlp_activation: str = "softplus",
        num_alphabet: int = 20,
        num_chi_bins: int = 20,
        decoder_num_hidden: int = 512,
        label_smoothing: float = 0.1,
        separate_packing: bool = True,
        graph_criterion: str = "knn",
        graph_random_min_local: int = 20,
        graph_attentional: bool = False,
        graph_num_attention_heads: int = 4,
        predict_S_marginals: bool = False,
        predict_S_potts: bool = False,
        potts_parameterization: str = "factor",
        potts_num_factors: Optional[int] = None,
        potts_symmetric_J: bool = True,
        noise_schedule: Optional[str] = None,
        noise_covariance_model: str = "brownian",
        noise_complex_scaling: bool = False,
        noise_beta_range: Tuple[float, float] = (0.2, 70.0),
        noise_log_snr_range: Tuple[float, float] = (-7.0, 13.5),
        checkpoint_gradients: bool = False,
        **kwargs
    ) -> None:
        """Initialize GraphDesign network."""
        super(GraphDesign, self).__init__()

        self.config, self.extra_kwargs = build_model_config(
            GraphDesignConfig,
            {
                "dim_nodes": dim_nodes,
                "dim_edges": dim_edges,
                "num_neighbors": num_neighbors,
                "node_features": node_features,
                "edge_features": edge_features,
                "sequence_embedding": sequence_embedding,
                "sidechain_embedding": sidechain_embedding,
                "sidechains": sidechains,
                "num_layers": num_layers,
                "num_layers_encoder": num_layers_encoder,
                "dropout": dropout,
                "node_mlp_layers": node_mlp_layers,
                "node_mlp_dim": node_mlp_dim,
                "edge_update": edge_update,
                "edge_mlp_layers": edge_mlp_layers,
                "edge_mlp_dim": edge_mlp_dim,
                "skip_connect_input": skip_connect_input,
                "mlp_activation": mlp_activation,
                "num_alphabet": num_alphabet,
                "num_chi_bins": num_chi_bins,
                "decoder_num_hidden": decoder_num_hidden,
                "label_smoothing": label_smoothing,
                "separate_packing": separate_packing,
                "graph_criterion": graph_criterion,
                "graph_random_min_local": graph_random_min_local,
                "graph_attentional": graph_attentional,
                "graph_num_attention_heads": graph_num_attention_heads,
                "predict_S_marginals": predict_S_marginals,
                "predict_S_potts": predict_S_potts,
                "potts_parameterization": potts_parameterization,
                "potts_num_factors": potts_num_factors,
                "potts_symmetric_J": potts_symmetric_J,
                "noise_schedule": noise_schedule,
                "noise_covariance_model": noise_covariance_model,
                "noise_complex_scaling": noise_complex_scaling,
                "noise_beta_range": noise_beta_range,
                "noise_log_snr_range": noise_log_snr_range,
                "checkpoint_gradients": checkpoint_gradients,
                **kwargs,
            },
        )
        self.kwargs = config_to_kwargs(self.config, self.extra_kwargs)
        args = self.config

        # Important global options
        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.num_alphabet = num_alphabet
        self.num_chi_bins = num_chi_bins
        self.separate_packing = separate_packing
        self.sidechains = sidechains
        self.predict_S_potts = predict_S_potts
        self.traversal = ProteinTraversalSpatial()

        # Encoder GNN process backbone
        self.encoder = BackboneEncoderGNN(
            dim_nodes=args.dim_nodes,
            dim_edges=args.dim_edges,
            num_neighbors=args.num_neighbors,
            node_features=args.node_features,
            edge_features=args.edge_features,
            num_layers=(
                args.num_layers
                if args.num_layers_encoder is None
                else args.num_layers_encoder
            ),
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

        # Time features for diffusion
        if args.noise_schedule is not None:
            self.noise_perturb = diffusion.DiffusionChainCov(
                noise_schedule=args.noise_schedule,
                beta_min=args.noise_beta_range[0],
                beta_max=args.noise_beta_range[1],
                log_snr_range=args.noise_log_snr_range,
                covariance_model=args.noise_covariance_model,
                complex_scaling=args.noise_complex_scaling,
            )
            self.time_features = diffusion.NoiseTimeEmbedding(
                dim_embedding=args.dim_nodes,
                noise_schedule=self.noise_perturb.noise_schedule,
            )

        # Decoder GNN process backbone
        if self.sidechains:
            self.decoder = SidechainDecoderGNN(
                dim_nodes=args.dim_nodes,
                dim_edges=args.dim_edges,
                num_neighbors=args.num_neighbors,
                predict_S=True,
                predict_chi=(not args.separate_packing),
                sequence_embedding=args.sequence_embedding,
                sidechain_embedding=args.sidechain_embedding,
                num_layers=args.num_layers,
                node_mlp_layers=args.node_mlp_layers,
                node_mlp_dim=args.node_mlp_dim,
                edge_update=args.edge_update,
                edge_mlp_layers=args.edge_mlp_layers,
                edge_mlp_dim=args.edge_mlp_dim,
                mlp_activation=args.mlp_activation,
                dropout=args.dropout,
                skip_connect_input=args.skip_connect_input,
                num_alphabet=args.num_alphabet,
                num_chi_bins=args.num_chi_bins,
                decoder_num_hidden=args.decoder_num_hidden,
                label_smoothing=args.label_smoothing,
                checkpoint_gradients=checkpoint_gradients,
            )

        if args.predict_S_marginals:
            self.decoder_S_marginals = NodePredictorS(
                num_alphabet=args.num_alphabet,
                dim_nodes=args.dim_nodes,
                dim_hidden=args.decoder_num_hidden,
                label_smoothing=args.label_smoothing,
            )

        if args.predict_S_potts:
            self.decoder_S_potts = potts.GraphPotts(
                dim_nodes=args.dim_nodes,
                dim_edges=args.dim_edges,
                num_states=args.num_alphabet,
                parameterization=args.potts_parameterization,
                num_factors=args.potts_num_factors,
                symmetric_J=args.potts_symmetric_J,
                dropout=args.dropout,
                label_smoothing=args.label_smoothing,
            )

        if args.separate_packing:
            # Optionally do a two-stage autoregressive prediction
            self.embed_S = nn.Embedding(args.num_alphabet, args.dim_nodes)
            self.encoder_S_gnn = graph.GraphNN(
                dim_nodes=args.dim_nodes,
                dim_edges=args.dim_edges,
                num_layers=args.num_layers,
                node_mlp_layers=args.node_mlp_layers,
                node_mlp_dim=args.node_mlp_dim,
                edge_update=args.edge_update,
                edge_mlp_layers=args.edge_mlp_layers,
                edge_mlp_dim=args.edge_mlp_dim,
                mlp_activation=args.mlp_activation,
                dropout=args.dropout,
                norm="transformer",
                scale=args.num_neighbors,
                skip_connect_input=args.skip_connect_input,
                checkpoint_gradients=checkpoint_gradients,
            )
            self.decoder_chi = SidechainDecoderGNN(
                dim_nodes=args.dim_nodes,
                dim_edges=args.dim_edges,
                num_neighbors=args.num_neighbors,
                predict_S=False,
                predict_chi=True,
                sequence_embedding=args.sequence_embedding,
                sidechain_embedding=args.sidechain_embedding,
                num_layers=args.num_layers,
                node_mlp_layers=args.node_mlp_layers,
                node_mlp_dim=args.node_mlp_dim,
                edge_update=args.edge_update,
                edge_mlp_layers=args.edge_mlp_layers,
                edge_mlp_dim=args.edge_mlp_dim,
                mlp_activation=args.mlp_activation,
                dropout=args.dropout,
                skip_connect_input=args.skip_connect_input,
                num_alphabet=args.num_alphabet,
                num_chi_bins=args.num_chi_bins,
                decoder_num_hidden=args.decoder_num_hidden,
                label_smoothing=args.label_smoothing,
                checkpoint_gradients=checkpoint_gradients,
            )

        if sidechains:
            self.chi_to_X = sidechain.SideChainBuilder()
            self.X_to_chi = sidechain.ChiAngles()
            self.loss_rmsd = sidechain.LossSideChainRMSD()
            self.loss_clash = sidechain.LossSidechainClashes()

        self.loss_eps = 1e-5

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        S: torch.LongTensor,
        t: Optional[torch.Tensor] = None,
        sample_noise: bool = False,
        permute_idx: Optional[torch.LongTensor] = None,
        priority: Optional[torch.LongTensor] = None,
    ) -> dict:
        # Sample noisy backbones
        X_noise = X
        if sample_noise and hasattr(self, "noise_perturb"):
            X_bb = X[:, :, :4, :]
            _schedule = self.noise_perturb.noise_schedule
            t = self.noise_perturb.sample_t(C, t)
            X_noise_bb = self.noise_perturb(X_bb, C, t=t)
            if self.sidechains:
                # Rebuild sidechains on noised backbone from native chi angles
                chi, mask_chi = self.X_to_chi(X, C, S)
                X_noise, mask_X = self.chi_to_X(X_noise_bb, C, S, chi)
            else:
                if X.shape[2] > X_noise_bb.shape[2]:
                    X_noise = torch.cat([X_noise_bb, X[:, :, 4:, :]], dim=2)
                else:
                    X_noise = X_noise_bb

        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X_noise, C, t=t)

        logp_S_marginals = None
        if self.kwargs["predict_S_marginals"]:
            logp_S_marginals, _ = self.decoder_S_marginals(S, node_h, mask_i)

        logp_S_potts = None
        if self.kwargs["predict_S_potts"]:
            logp_S_potts = self.decoder_S_potts.loss(
                S, node_h, edge_h, edge_idx, mask_i, mask_ij
            )

        # Sample random permutations and build autoregressive mask
        if permute_idx is None:
            permute_idx = self.traversal(X, C, priority=priority)

        if self.sidechains:
            # In one-stage packing, predict S and chi angles in an interleaved manner
            (
                logp_S,
                logp_chi,
                chi,
                mask_chi,
                node_h_chi,
                _,
                _,
                _,
                mask_ij_causal,
            ) = self.decoder(
                X_noise, C, S, node_h, edge_h, edge_idx, mask_i, mask_ij, permute_idx
            )
        else:
            logp_S = None
            logp_chi = None
            chi = None
            mask_chi = None
            node_h_chi = None
            mask_ij_causal = None

        if self.separate_packing:
            # In two-stage packing, re-process embeddings with sequence
            node_h = node_h + mask_i.unsqueeze(-1) * self.embed_S(S)
            node_h, edge_h = self.encoder_S_gnn(
                node_h, edge_h, edge_idx, mask_i, mask_ij
            )
            _, logp_chi, chi, mask_chi, node_h_chi, _, _, _, _ = self.decoder_chi(
                X_noise, C, S, node_h, edge_h, edge_idx, mask_i, mask_ij, permute_idx
            )
        if t is None:
            t = torch.zeros(C.size(0), device=C.device)
        outputs = {
            "logp_S": logp_S,
            "logp_chi": logp_chi,
            "logp_S_marginals": logp_S_marginals,
            "logp_S_potts": logp_S_potts,
            "chi": chi,
            "mask_chi": mask_chi,
            "node_h_chi": node_h_chi,
            "mask_i": mask_i,
            "mask_ij": mask_ij,
            "mask_ij_causal": mask_ij_causal,
            "edge_idx": edge_idx,
            "permute_idx": permute_idx,
            "X_noise": X_noise,
            "t": t,
        }
        return outputs

    def set_gradient_checkpointing(self, flag: bool):
        """Sets gradient checkpointing to `flag` on all relevant modules"""
        self.encoder.checkpoint_gradients = flag
        self.encoder.gnn.checkpoint_gradients = flag
        if self.sidechains:
            self.decoder.checkpoint_gradients = flag
            self.decoder.gnn.checkpoint_gradients = flag
        if self.separate_packing:
            self.encoder_S_gnn.checkpoint_gradients = flag
            self.decoder_chi.checkpoint_gradients = flag
            self.decoder_chi.gnn.checkpoint_gradients = flag

    @validate_XC()
    def encode(
        self, X: torch.Tensor, C: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the backbone and (optionally) the noise level.

        Args:
            X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
            C (torch.LongTensor): Chain map with shape
                `(num_batch, num_residues)`.
            t (torch.Tensor, optional): Diffusion timesteps corresponding to
                noisy input backbones, of shape `(num_batch)`. Default is no
                noise.

        Returns:
            node_h (torch.Tensor): Node features with shape
                `(num_batch, num_residues, dim_nodes)`.
            edge_h (torch.Tensor): Edge features with shape
                `(num_batch, num_residues, num_neighbors, dim_edges)`.
            edge_idx (torch.LongTensor): Edge indices for neighbors with shape
                `(num_batch, num_residues, num_neighbors)`.
            mask_i (torch.Tensor): Node mask with shape
                `(num_batch, num_residues)`.
            mask_ij (torch.Tensor): Edge mask with shape
                 `(num_batch, num_nodes, num_neighbors)`.
        """

        node_h_aux = None
        if hasattr(self, "time_features"):
            t = 0.0 if t is None else t
            node_h_aux = self.time_features(t)

        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encoder(
            X, C, node_h_aux=node_h_aux
        )
        return node_h, edge_h, edge_idx, mask_i, mask_ij

    @validate_XC()
    def predict_marginals(
        self, X: torch.Tensor, C: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict sequence marginal likelihoods.

        Args:
            X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
            C (torch.LongTensor): Chain map with shape
                `(num_batch, num_residues)`.
            t (torch.Tensor, optional): Diffusion timesteps corresponding to
                noisy input backbones, of shape `(num_batch)`. Default is no
                noise.

        Returns:
            log_probs_S (torch.Tensor): Node-wise sequence log probabilities
                with shape `(num_batch, num_residues, 20)`.
            mask_i (torch.Tensor): Node mask with shape
                `(num_batch, num_residues)`.
        """

        if not self.kwargs["predict_S_marginals"]:
            raise Exception(
                "This version of GraphDesign was not trained with marginal prediction"
            )
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X, C, t)
        log_probs_S = self.decoder_S_marginals.log_probs_S(node_h, mask_i)
        return log_probs_S, mask_i

    @validate_XC()
    def predict_potts(
        self, X: torch.Tensor, C: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Predict sequence Potts model.

        Args:
            X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
            C (torch.LongTensor): Chain map with shape
                `(num_batch, num_residues)`.
            t (torch.Tensor, optional): Diffusion timesteps corresponding to
                noisy input backbones, of shape `(num_batch)`. Default is no
                noise.

        Returns:
            h (torch.Tensor): The h tensor of a Potts model with dimensions
                `(seq_length, n_tokens)`.
            J (torch.Tensor): The J tensor of a Potts model with dimensions
                `(seq_length, seq_length, n_tokens, n_tokens)`.
            edge_idx (torch.LongTensor): Edge indices for neighbors with shape
                `(num_batch, num_residues, num_neighbors)` from GNN encoding.
        """
        if not self.kwargs["predict_S_potts"]:
            raise Exception(
                "This version of GraphDesign was not trained with Potts prediction"
            )
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X, C, t)
        h, J = self.decoder_S_potts(node_h, edge_h, edge_idx, mask_i, mask_ij)
        return h, J, edge_idx

    @validate_XC()
    def loss(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        S: torch.LongTensor,
        t: Optional[torch.Tensor] = None,
        permute_idx: Optional[torch.LongTensor] = None,
        sample_noise: bool = False,
        batched: bool = True,
        **kwargs
    ) -> dict:
        """Compute losses used for training.

        Args:
            X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
            C (torch.LongTensor): Chain map with shape
                `(num_batch, num_residues)`.
            S (torch.LongTensor): Sequence tensor with shape
                `(num_batch, num_residues)`.
            t (torch.Tensor, optional): Diffusion timesteps corresponding to
                noisy input backbones, of shape `(num_batch)`. Default is no
                noise.
            permute_idx (torch.LongTensor, optional): Permutation tensor for
                fixing the autoregressive decoding order
                `(num_batch, num_residues)`. If `None` (default), a random
                decoding order will be generated.
            sample_noise (bool): Whether to apply noise to input backbones.
            batched (bool): Whether to batch average losses.

        Returns (dict):
            neglogp (torch.Tensor): Sum of `neglogp_S` and `neglogp_chi` with
                shape `(num_batch, num_residues)`.
            neglogp_S (torch.Tensor): Average negative log probability per
                residue identity with shape `(num_batch, num_residues)`.
            neglogp_S_marginals (torch.Tensor): Average negative log probability
                per residue identity from marginal decoder with shape
                `(num_batch, num_residues)`.
            neglogp_S_potts (torch.Tensor): Average negative log probability per
                residue identity from Potts decoder with shape
                `(num_batch, num_residues)`.
            neglogp_chi (torch.Tensor): Average negative log probability per chi
                angle with shape `(num_batch, num_residues)`.
            mask_chi (torch.Tensor): Chi angle mask with shape
                `(batch_size, num_residues, 4)`.
            rmsd (torch.Tensor): Average RMSD per side-chain after sampling.
            clash (torch.Tensor): Average number of clashes per side-chain after
                sampling.
            permute_idx (LongTensor, optional): Permutation tensor that was
                used for the autoregressive decoding order with shape
                `(num_batch, num_residues)`.
        """

        o = self.forward(
            X, C, S, t=t, permute_idx=permute_idx, sample_noise=sample_noise
        )

        # Aggregate into per-residue scores for the batch
        if batched:
            _avg = lambda m, l: (m * l).sum() / (m.sum() + self.loss_eps)
        else:
            _avg = lambda m, l: (m * l).sum(dim=tuple(range(1, l.dim()))) / (
                m.sum(dim=tuple(range(1, l.dim()))) + self.loss_eps
            )
        mask_S = o["mask_i"]
        neglogp_chi = -_avg(o["mask_chi"], o["logp_chi"])
        if o["logp_S"] is not None:
            neglogp_S = -_avg(mask_S, o["logp_S"])
            neglogp = neglogp_S + neglogp_chi
        else:
            neglogp_S = None
            neglogp = neglogp_chi
        if o["logp_S_marginals"] is not None:
            neglogp_S_marginals = -_avg(mask_S, o["logp_S_marginals"])
            neglogp = neglogp + neglogp_S_marginals
        else:
            neglogp_S_marginals = None
        if o["logp_S_potts"] is not None:
            neglogp_S_potts = -_avg(mask_S, o["logp_S_potts"])
            neglogp = neglogp + neglogp_S_potts
        else:
            neglogp_S_potts = None

        # Evaluate sampled side chains
        decoder = self.decoder_chi if self.separate_packing else self.decoder
        chi_sample = decoder.decoder_chi.sample(
            S, o["mask_chi"], o["node_h_chi"], o["mask_i"], temperature=0.01
        )
        X_sample, mask_X = self.chi_to_X(o["X_noise"][:, :, :4, :], C, S, chi_sample)

        # RMSD loss
        rmsd_i = self.loss_rmsd(o["X_noise"], X_sample, C, S)
        rmsd = _avg(mask_S, rmsd_i)

        # Clash loss measures clashes generated to the past
        clashes = self.loss_clash(
            X_sample, C, S, edge_idx=o["edge_idx"], mask_ij=o["mask_ij_causal"]
        )
        clash = _avg(mask_S, clashes)

        losses = {
            "neglogp": neglogp,
            "neglogp_S": neglogp_S,
            "neglogp_S_marginals": neglogp_S_marginals,
            "neglogp_S_potts": neglogp_S_potts,
            "neglogp_chi": neglogp_chi,
            "mask_chi": o["mask_chi"],
            "rmsd": rmsd,
            "clash": clash,
            "permute_idx": o["permute_idx"],
            "t": o["t"],
        }
        return losses

    @torch.no_grad()
    @validate_XC()
    def sample(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        S: Optional[torch.LongTensor] = None,
        t: Optional[Union[float, torch.Tensor]] = None,
        t_packing: Optional[Union[float, torch.Tensor]] = None,
        mask_sample: Optional[torch.Tensor] = None,
        permute_idx: Optional[torch.LongTensor] = None,
        temperature_S: float = 0.1,
        temperature_chi: float = 1e-3,
        clamped: bool = False,
        resample_chi: bool = True,
        return_scores: bool = False,
        top_p_S: Optional[float] = None,
        ban_S: Optional[tuple] = None,
        sampling_method: Literal["potts", "autoregressive"] = "autoregressive",
        regularization: Optional[str] = "LCP",
        potts_sweeps: int = 500,
        potts_proposal: Literal["dlmc", "chromatic"] = "dlmc",
        verbose: bool = False,
        symmetry_order: Optional[int] = None,
    ) -> tuple:
        """Sample sequence and side chain conformations given an input structure.

        Args:
            X (torch.Tensor): All atom coordinates with shape
                `(num_batch, num_residues, 14, 3)`.
            C (torch.LongTensor): Chain map with shape
                `(num_batch, num_residues)`.
            S (torch.LongTensor): Sequence tensor with shape
                `(num_batch, num_residues)`.
            t (float or torch.Tensor, optional): Diffusion time for models trained with
                diffusion augmentation of input structures. Setting `t=0` or
                `t=None` will condition the model to treat the structure as
                exact coordinates, while values of `t > 0` will condition
                the model to treat structures as though they were drawn from
                noise-augmented ensembles with that noise level. Default is `None`,
                while for robust design we recommend `t=0.5`. May be a float or
                a tensor of shape `(num_batch)`.
            t_packing (float or torch.Tensor, optional): Potentially separate diffusion
                time for packing.
            mask_sample (torch.Tensor, optional): Binary tensor mask indicating
                positions to be sampled with shape `(num_batch, num_residues)` or
                position-specific valid amino acid choices with shape
                `(num_batch, num_residues, num_alphabet)`. If `None` (default), all
                positions will be sampled.
            permute_idx (LongTensor, optional): Permutation tensor for fixing
                the autoregressive decoding order `(num_batch, num_residues)`.
                If `None` (default), a random decoding order will be generated.
            temperature_S (float): Temperature parameter for sampling sequence
                tokens. A value of `temperature_S=1.0` corresponds to the
                model's unadjusted positions, though because of training such as
                label smoothing values less than 1.0 are recommended. Default is
                `0.1`.
            temperature_chi (float): Temperature parameter for sampling chi
                angles. Even if a high temperature sequence is sampled, this is
                recommended to always be low. Default is `1E-3`.
            clamped (bool): If `True`, no sampling is done and the likelihood
                values will be calculated for the input sequence and structure.
                Used for validating the sequential versus parallel decoding
                modes. Default is `False`.
            resample_chi (bool): If `True`, all chi angles will be resampled,
                even for sequence positions that were not sampled (i.e. the model
                will perform global repacking). Default is `True`.
            return_scores (bool): If `True`, return dictionary containing
                likelihood scores similar to those produced by `forward`.
            top_p_S (float, optional): Option to perform top-p sampling for
                autoregressive sequence decoding. If not `None` it will be the
                top-p value [1].
                [1] Holtzman et al. The Curious Case of Neural Text Degeneration. (2020)
            ban_S (tuple, optional): An optional set of token indices from
                `chroma.constants.AA20` to ban during sampling.
            sampling_method (str): Sampling method for decoding sequence from structure.
                If `autoregressive`, sequences will be designed by ancestral sampling with
                the autoregessive decoder head. If `potts`, sequences will be designed
                via MCMC with the potts decoder head.
            regularization (str, optional): Optional sequence regularization to use
                during decoding. Can be `LCP` for Local Composition Perplexity regularization
                which penalizes local sequence windows from having unnaturally low
                compositional entropies. (Implemented for both `potts` and `autoregressive`)
            potts_sweeps (int): Number of sweeps to perform for MCMC sampling of `potts`
                decoder. A sweep corresponds to a sufficient number of Monte Carlo steps
                such that every position could have changed.
            potts_proposal (str): MCMC proposal for Potts sampling. Currently implemented
                proposals are `dlmc` for Discrete Langevin Monte Carlo [1] or `chromatic`
                for Gibbs sampling with graph coloring.
                [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).
            symmetry_order (int, optional): Optional integer argument to enable
                symmetric sequence decoding under `symmetry_order`-order symmetry.
                The first `(num_nodes // symmetry_order)` states will be free to
                move, and all consecutively tiled sets of states will be locked
                to these during decoding. Internally this is accomplished by
                summing the parameters Potts model under a symmetry constraint
                into this reduced sized system and then back imputing at the end.
                Currently only implemented for Potts models.

        Returns:
            X_sample (torch.Tensor): Sampled all atom coordinates with shape
                `(num_batch, num_residues, 14, 3)`.
            S_sample (torch.LongTensor): Sampled sequence tensor with shape
                `(num_batch, num_residues)`.
            permute_idx (torch.LongTensor): Permutation tensor that was used
                for the autoregressive decoding order with shape
                `(num_batch, num_residues)`.
            scores (dict, optional): Dictionary containing likelihood scores
                similar to those produced by `forward`.
        """
        if X.shape[2] == 4:
            X = F.pad(X, [0, 0, 0, 10])
        alphabet = constants.AA20
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X, C, t=t)

        # Process sampling mask
        logits_init = torch.zeros(
            list(C.shape) + [len(alphabet)], device=C.device
        ).float()
        if ban_S is not None:
            ban_S = [alphabet.index(c) for c in ban_S]
        mask_sample, mask_sample_1D, S_init = potts.init_sampling_masks(
            logits_init, mask_sample, S=S, ban_S=ban_S
        )
        if not clamped:
            S = S_init

        # Sample random permutations and build autoregressive mask
        if permute_idx is None:
            permute_idx = self.traversal(X, C, priority=mask_sample_1D)

        if symmetry_order is not None and not (sampling_method == "potts"):
            raise NotImplementedError(
                "Symmetric decoding is currently only supported for Potts models"
            )

        if sampling_method == "potts":
            if not self.kwargs["predict_S_potts"]:
                raise Exception(
                    "This GraphDesign model was not trained with Potts prediction"
                )

            # Complexity regularization
            penalty_func = None
            mask_ij_coloring = None
            edge_idx_coloring = None
            if regularization == "LCP":
                C_complexity = (
                    C
                    if symmetry_order is None
                    else C[:, : C.shape[1] // symmetry_order]
                )
                penalty_func = lambda _S: complexity.complexity_lcp(_S, C_complexity)
                # edge_idx_coloring, mask_ij_coloring = complexity.graph_lcp(C, edge_idx, mask_ij)

            S_sample, _ = self.decoder_S_potts.sample(
                node_h,
                edge_h,
                edge_idx,
                mask_i,
                mask_ij,
                S=S,
                mask_sample=mask_sample,
                temperature=temperature_S,
                num_sweeps=potts_sweeps,
                penalty_func=penalty_func,
                proposal=potts_proposal,
                rejection_step=(potts_proposal == "chromatic"),
                verbose=verbose,
                edge_idx_coloring=edge_idx_coloring,
                mask_ij_coloring=mask_ij_coloring,
                symmetry_order=symmetry_order,
            )
            chi_sample, logp_S, logp_chi = None, None, None
        else:
            # Sample sequence (and chi angles if one-stage)

            # Complexity regularization
            bias_S_func = None
            if regularization == "LCP":
                bias_S_func = complexity.complexity_scores_lcp_t

            S_sample, chi_sample, logp_S, logp_chi, _ = self.decoder.decode(
                X,
                C,
                S,
                node_h,
                edge_h,
                edge_idx,
                mask_i,
                mask_ij,
                permute_idx,
                temperature_S=temperature_S,
                temperature_chi=temperature_chi,
                sample=not clamped,
                mask_sample=mask_sample,
                resample_chi=resample_chi,
                top_p_S=top_p_S,
                ban_S=ban_S,
                bias_S_func=bias_S_func,
            )

        if self.separate_packing:
            if t != t_packing:
                node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(
                    X, C, t=t_packing
                )

            # In two-stage packing, re-process embeddings with sequence
            node_h = node_h + mask_i.unsqueeze(-1) * self.embed_S(S_sample)
            node_h, edge_h = self.encoder_S_gnn(
                node_h, edge_h, edge_idx, mask_i, mask_ij
            )
            _, chi_sample, _, logp_chi, _ = self.decoder_chi.decode(
                X,
                C,
                S_sample,
                node_h,
                edge_h,
                edge_idx,
                mask_i,
                mask_ij,
                permute_idx,
                temperature_chi=temperature_chi,
                sample=not clamped,
                mask_sample=mask_sample_1D,
                resample_chi=resample_chi,
            )

        # Rebuild side chains
        X_sample, mask_X = self.chi_to_X(X[:, :, :4, :], C, S_sample, chi_sample)

        if return_scores:
            if sampling_method == "potts":
                raise NotImplementedError

            # Summarize
            mask_chi = sidechain.chi_mask(C, S_sample)
            neglogp_S = -(mask_i * logp_S).sum([1]) / (
                (mask_i).sum([1]) + self.loss_eps
            )
            neglogp_chi = -(mask_chi * logp_chi).sum([1, 2]) / (
                mask_chi.sum([1, 2]) + self.loss_eps
            )

            scores = {
                "neglogp_S": neglogp_S,
                "neglogp_chi": neglogp_chi,
                "logp_S": logp_S,
                "logp_chi": logp_chi,
                "mask_i": mask_i,
                "mask_chi": mask_chi,
            }
            return X_sample, S_sample, permute_idx, scores
        else:
            return X_sample, S_sample, permute_idx

    @validate_XC()
    def pack(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        S: torch.LongTensor,
        permute_idx: Optional[torch.LongTensor] = None,
        temperature_chi: float = 1e-3,
        clamped: bool = False,
        resample_chi: bool = True,
        return_scores: bool = False,
    ) -> tuple:
        """Sample side chain conformations given an input structure.

        Args:
            X (torch.Tensor): All atom coordinates with shape
                `(num_batch, num_residues, 14, 3)`.
            C (torch.LongTensor): Chain map with shape
                `(num_batch, num_residues)`.
            S (torch.LongTensor): Sequence tensor with shape
                `(num_batch, num_residues)`.
            permute_idx (LongTensor, optional): Permutation tensor for fixing
                the autoregressive decoding order `(num_batch, num_residues)`.
                If `None` (default), a random decoding order will be generated.
            temperature_chi (float): Temperature parameter for sampling chi
                angles. Even if a high temperature sequence is sampled, this is
                recommended to always be low. Default is `1E-3`.
            clamped (bool): If `True`, no sampling is done and the likelihood
                values will be calculated for the input sequence and structure.
                Used for validating the sequential versus parallel decoding
                modes. Default is `False`
            resample_chi (bool): If `True`, all chi angles will be resampled,
                even for sequence positions that were not sampled (i.e. global
                repacking). Default is `True`.
            return_scores (bool): If `True`, return dictionary containing
                likelihood scores similar to those produced by `forward`.

        Returns:
            X_sample (torch.Tensor): Sampled all atom coordinates with shape
                `(num_batch, num_residues, 14, 3)`.
            neglogp_chi (torch.Tensor, optional): Average negative log
                probability per chi angle.
            permute_idx (torch.LongTensor): Permutation tensor that was used
                for the autoregressive decoding order with shape
                `(num_batch, num_residues)`.
            scores (dict, optional): Dictionary containing likelihood scores
                similar to those produced by `forward`.
        """
        assert self.separate_packing

        with torch.no_grad():
            if X.shape[2] == 4:
                X = F.pad(X, [0, 0, 0, 10])

            node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X, C)

            # Sample random permutations and build autoregressive mask
            if permute_idx is None:
                permute_idx = self.traversal(X, C)

            # In two-stage packing, re-process embeddings with sequence
            node_h = node_h + mask_i.unsqueeze(-1) * self.embed_S(S)
            node_h, edge_h = self.encoder_S_gnn(
                node_h, edge_h, edge_idx, mask_i, mask_ij
            )
            _, chi_sample, _, logp_chi, _ = self.decoder_chi.decode(
                X,
                C,
                S,
                node_h,
                edge_h,
                edge_idx,
                mask_i,
                mask_ij,
                permute_idx,
                temperature_chi=temperature_chi,
                sample=not clamped,
                resample_chi=resample_chi,
            )

            X_sample, mask_X = self.chi_to_X(X[:, :, :4, :], C, S, chi_sample)

            # Summarize
            mask_chi = sidechain.chi_mask(C, S)
            neglogp_chi = -(mask_chi * logp_chi).sum([1, 2]) / (
                mask_chi.sum([1, 2]) + self.loss_eps
            )
        if return_scores:
            scores = {
                "neglogp_chi": neglogp_chi,
                "logp_chi": logp_chi,
                "mask_i": mask_i,
                "mask_chi": mask_chi,
            }
            return X_sample, permute_idx, scores
        else:
            return X_sample, permute_idx

        return X_sample, neglogp_chi, permute_idx
def load_model(
    weight_file: str,
    device: str = "cpu",
    strict: bool = False,
    strict_unexpected: bool = True,
    verbose: bool = True,
) -> GraphDesign:
    """Load model `GraphDesign`

    Args:
        weight_file (str): The destination path of the model weights to load.
            Compatible with files saved by `save_model`.
        device (str, optional): Pytorch device specification, e.g. `'cuda'` or `'mps'` for
        GPU or `'cpu'` in the absence of GPU. Default is `'cpu'`.
        strict (bool): Whether to require that the keys match between the
            input file weights and the model created from the parameters stored
            in the model kwargs.
        strict_unexpected (bool): Whether to require that there are no
            unexpected keys when loading model weights, as distinct from the
            strict option which doesn't allow for missing keys either. By
            default, we use this option rather than strict for ease of
            development when adding model features.

    Returns:
        model (GraphDesign): Instance of `GraphDesign` with loaded weights.
    """
    return utility_load_model(
        weight_file,
        GraphDesign,
        device=device,
        strict=strict,
        strict_unexpected=strict_unexpected,
        verbose=verbose,
    )

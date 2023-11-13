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

"""Layers for conditioning diffusion generative processes.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from torch import nn

import chroma.utility.chroma
from chroma.data.protein import Protein
from chroma.data.xcs import validate_XC
from chroma.layers.structure import backbone, mvn, optimal_transport, symmetry
from chroma.layers.structure.backbone import expand_chain_map
from chroma.models import graph_classifier, procap
from chroma.models.graph_backbone import GraphBackbone
from chroma.models.graph_classifier import GraphClassifier
from chroma.models.graph_design import GraphDesign
from chroma.models.procap import ProteinCaption


class Conditioner(torch.nn.Module):
    """
    A composable function for parameterizing protein design problems.

    Conditioners provide a general framework for expressing complex protein
    design problems in terms of simpler, composable sub-conditions in
    a way that enables automatic sampling. To accomplish this, Conditioners
    parameterize time-dependent transformations to the global coordinate system
    and total energy by mapping from unconstrained coordinates and energy to
    potentially updated coordinates and energy. This convention can subsume
    classifier guidance, bijective change-of-variables constrained MCMC, and
    linear subspace constrained MCMC as special cases.

    A conditioner is implemented as a function which maps from state-energy pairs
    at a time point `t` to updated state-energy pairs which may reflect hard constraints
    (typically updates to coordinates and energy) and restraintes (updates just to
    energy).  Conditioners take in and return 5 arguments `X, C, O, U, t`,
    where `X,C,O` is the protein complex in the `XCS` convention with `S` expressed as a
    one-hot tensor `O`, `U` is the total system energy and `t` is the diffusion time.
    Because conditioners have matched input and output types, they can be composed via
    sequential chaining. Further examples and descriptions of Conditioners can be found
    throughout this module.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Inputs:
        X (torch.Tensor): Coordinates with shape `(batch_size, num_residues, 4, 3)`.
        C (torch.LongTensor): Chain map with shape `(batch_size, num_residues)`.
        O (torch.Tensor): One-hot sequence with shape
            `(batch_size, num_residues, num_alphabet)`.
        U (torch.Tensor): energy tensor with shape `(batch_size,)`.
        t (Union[torch.Tensor, float]): Time tensor with shape `(batch_size,)` or
            a scalar.

    Outputs:
        X_out (torch.Tensor): Updated coordinates with shape
            `(batch_size, num_residues_out, 4, 3)`.
        C_out (torch.LongTensor): Updated chain map with shape
            `(batch_size, num_residues_out)`.
        O (torch.Tensor): Updated one-hot sequences with shape
            `(batch_size, num_residues_out, num_alphabet)`.
        U_out (torch.Tensor): Modified energy tensor with shape `(batch_size,)`.
        t_out (Union[torch.Tensor, float]): Modified time tensor with shape
             `(batch_size,)` or a scalar.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        pass


class Identity(Conditioner):
    def __init__(self):
        super().__init__()

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        return X, C, O, U, t


class ComposedConditioner(Conditioner):
    def __init__(self, conditioners):
        super().__init__()
        self.conditioners = nn.ModuleList(conditioners)

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        for conditioner in self.conditioners:
            X, C, O, U, t = conditioner(X, C, O, U, t)
        return X, C, O, U, t

    def _postprocessing_(
        self, protein: Protein, output_dict: Optional[dict] = None
    ) -> Union[Protein, Tuple[Protein, dict]]:
        for _conditioner in self.conditioners:
            if hasattr(_conditioner, "_postprocessing_"):
                if output_dict is None:
                    protein = _conditioner._postprocessing_(protein, output_dict)
                else:
                    protein, output_dict = _conditioner._postprocessing_(
                        protein, output_dict
                    )

        if output_dict is None:
            return protein
        else:
            return protein, output_dict


class SubsequenceConditioner(Conditioner):
    """
    SequenceConditioner:
    A Chroma Conditioning module which, given a GraphDesign model and a subset of
        residues for which sequence information is known, can add gradients to sampling
        that bias the samples towards increased `log p(sequence | structure)`

    Args:
        design_model (GraphDesign): Trained GraphDesign model.
        S_condition (torch.Tensor): Of shape (1, num_residues) specifying sequence
            information.
        mask_condition (torch.Tensor, optional): Of shape (1, num_residues) specifying
            which residues to include when computing `log p(sequence | structure)`
        weight (float, optional): Overall weight to which the gradient is scaled.
        renormalize_grad (bool, optional): Whether to renormalize gradient to have
            overall variance `weight`.
    """

    def __init__(
        self,
        design_model: GraphDesign,
        protein: Protein,
        selection: str = "all",
        weight: float = 1.0,
        renormalize_grad: Optional[bool] = False,
    ):
        super().__init__()
        self.design_model = design_model

        # Register sequence buffers
        X, C, S = protein.to_XCS()
        mask_condition = protein.get_mask(selection)
        self.register_buffer("S_condition", S)
        self.register_buffer("mask_condition", mask_condition)

        self.weight = weight
        self.renormalize_grad = renormalize_grad

    def _transform_gradient(self, grad, C, t):
        # grad = clip_atomic_magnitudes_percentile(grad)
        scale = self.weight / self.design_model.noise_perturb.noise_schedule.sigma(
            t
        ).to(C.device)
        grad = scale * grad / grad.square().mean().sqrt()
        return grad

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        X_input = X + 0.0

        if self.renormalize_grad:
            X_input.register_hook(lambda _X: self._transform_gradient(_X, C, t))
        if X.shape[2] == 4:
            X_input = F.pad(X_input, [0, 0, 0, 10])

        priority = None
        if self.mask_condition is not None:
            priority = 1.0 - self.mask_condition
        out = self.design_model(X_input, C, self.S_condition, t, priority=priority)
        logp_S = out["logp_S"]

        if self.mask_condition is not None:
            logp_S = self.mask_condition * logp_S
        U = U + self.weight * -logp_S.sum()
        return X, C, O, U, t


class ShapeConditioner(Conditioner):
    """Volumetric potential for optimizing towards arbitrary geometries.

    Args:
        X_target (numpy array): Target point cloud with shape `(num_points, 3)`.
        noise_schedule (GaussianNoiseSchedule): Diffusion time schedule for loss
            scaling.
        autoscale (bool): If True, automatically rescale target point cloud coordinates
            such that they are approximately volume-scaled to a target protein size.
            Volume is roughly estimated by converting the point cloud to a sphere cloud
            with radii large enough to overlap with near neighbors and double counting
            corrections via inclusion-exclusion.
        autoscale_num_residues (int): Target protein size for auto-scaling.
        autoscale_target_ratio (float): Scale factor for adjusting the target protein
            volume.
        scale_invariant (bool): If True, compute the loss in a size invariant manner
            by dynamically renormalizing the point clouds to match Radii of gyration.
            This approach can be more unstable to integrate and require more careful tuning.
        shape_loss_weight (float): Scale factor for the overall restraint.
        shape_loss_cutoff (float): Minimal distance deviation that is penalized in the loss,
            e.g. to treat as a flat-bottom restraint below the cutoff.
        sinkhorn_iterations (int): Number of Sinkhorn iterations for Optimal Transport
            calculations.
        sinkhorn_scale (float): Entropy regularization scaling parameter for Optimal
            Transport calculations.
        sinkhorn_iterations_gw (int): Number of Sinkhorn iterations for Gromov-Wasserstein
            Optimal Transport calculations.
        sinkhorn_scale_gw (float): Entropy regularization scaling parameter for
            Gromov-Wasserstein Optimal Transport calculations.
        gw_layout (bool): If True, use Gromov-Wasserstein Optimal Transport to compute
            a point cloud correspondence assuming ideal protein distance scaling.
        gw_layout_coefficient (float): Scale factor with which to combine average
            inter-point cloud distances according to OT (Wasserstein) versus
            Gromov-Wasserstein couplings.
    """

    def __init__(
        self,
        X_target,
        noise_schedule,
        autoscale: bool = True,
        autoscale_num_residues: int = 500,
        autoscale_target_ratio: float = 0.4,
        scale_invariant: bool = False,
        shape_loss_weight: float = 20.0,
        shape_loss_cutoff: float = 0.0,
        shape_cutoff_D: float = 0.01,
        scale_max_rg_ratio: float = 1.5,
        sinkhorn_iterations: int = 10,
        sinkhorn_scale: float = 1.0,
        sinkhorn_scale_gw: float = 200.0,
        sinkhorn_iterations_gw: int = 30,
        gw_layout: bool = True,
        gw_layout_coefficient: float = 0.4,
        eps: float = 1e-3,
        debug: bool = False,
    ):
        super().__init__()

        self.eps = eps

        self.noise_schedule = noise_schedule

        # Shape control parameters
        self.shape_loss_weight = shape_loss_weight
        self.shape_loss_cutoff = shape_loss_cutoff
        self.scale_invariant = scale_invariant
        self.shape_cutoff_D = shape_cutoff_D
        self.scale_max_rg_ratio = scale_max_rg_ratio

        # Autoscale volumes (in units of cubic angstroms)
        self.autoscale = autoscale
        self.autoscale_num_residues = autoscale_num_residues
        self.autoscale_target_ratio = autoscale_target_ratio

        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_scale = sinkhorn_scale
        self.sinkhorn_iterations_gw = sinkhorn_iterations_gw
        self.sinkhorn_scale_gw = sinkhorn_scale_gw

        self.debug = debug

        if torch.is_tensor(X_target):
            X_target = X_target.cpu().data.numpy()

        if self.autoscale:
            X_target, self.shape_cutoff_D = chroma.utility.chroma.point_cloud_rescale(
                X_target,
                self.autoscale_num_residues,
                scale_ratio=self.autoscale_target_ratio,
            )

        # Map coupling with Gromov Wasserstein optimal transport
        self.gw_layout = gw_layout
        self.gw_layout_coefficient = gw_layout_coefficient
        if self.gw_layout:
            self._map_gw_coupling_ideal_glob(
                X_target, num_residues=autoscale_num_residues
            )

        X_target = torch.Tensor(X_target)
        self.register_buffer("X_target", X_target[None, ...].clone().detach())

    def _distance_knn(self, X, top_k=12, max_scale=10.0):
        """Topology distance."""
        X_np = X.cpu().data.numpy()
        D = np.sqrt(
            ((X_np[:, :, np.newaxis, :] - X_np[:, np.newaxis, :, :]) ** 2).sum(-1)
        )

        # Distance cutoff
        D_cutoff = np.mean(np.sort(D[0, :, :], axis=-1)[:, top_k])
        D[D > D_cutoff] = max_scale * np.max(D)
        D = shortest_path(D[0, :, :])[np.newaxis, :, :]
        D = torch.Tensor(D).float().to(X.device)
        return D

    @torch.no_grad()
    def _map_gw_coupling_ideal_glob(self, X_target, num_residues):
        """Plan a layout using Gromov-Wasserstein Optimal transport"""

        X_target = torch.Tensor(X_target).float().unsqueeze(0)
        if torch.cuda.is_available():
            X_target = X_target.to("cuda")

        chain_ix = torch.arange(4 * num_residues, device=X_target.device) / 4.0
        distance_1D = (chain_ix[None, :, None] - chain_ix[None, None, :]).abs()
        # Scaling fit log-log to large scale single chain 6HYP
        D_model = 7.21 * distance_1D**0.322
        D_model = D_model / D_model.mean([1, 2], keepdims=True)

        D_target = self._distance_knn(X_target)
        D_target = D_target / D_target.mean([1, 2], keepdims=True)

        T_gw, D_gw = optimal_transport.optimize_couplings_gw(
            D_model,
            D_target,
            scale=self.sinkhorn_scale_gw,
            iterations_outer=self.sinkhorn_iterations_gw,
            iterations_inner=self.sinkhorn_iterations,
        )

        self.register_buffer("T_gw", T_gw.clone().detach())
        return

    def _distance(self, X_i, X_j):
        dX = X_i.unsqueeze(2) - X_j.unsqueeze(1)
        D = torch.sqrt((dX**2).sum(-1) + self.eps)
        return D

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        # Distance matrix is
        # [Num_batch, Num_atoms_target, Num_atoms_model]
        X_target = self.X_target
        X_model = X.reshape([X.shape[0], -1, 3])

        # Radius of gyration ceiling
        num_residues = X.shape[1]
        min_rg = 2.0 * num_residues**0.333
        max_rg = self.scale_max_rg_ratio * 2.0 * num_residues**0.4
        shape_cutoff_D = self.shape_cutoff_D

        def _center(_X):
            _X = _X - _X.mean(1, keepdim=True)
            return _X

        def _rg(_X):
            _X = _X - _X.mean(1, keepdim=True)
            rsq = _X.square().sum(2, keepdim=True)
            rg = rsq.mean(1, keepdim=True).sqrt()
            return rg

        X_model = _center(X_model)
        X_target = _center(X_target)

        if self.scale_invariant:

            def _resize(_X, target_rg):
                _X = _X - _X.mean(1, keepdim=True)
                rsq = _X.square().sum(2, keepdim=True)
                rg = rsq.mean(1, keepdim=True).sqrt()
                return _X / rg * target_rg

            X_model = _resize(X_model, _rg(X_target))

        # Compute interatomic distances
        D_inter = self._distance(X_model, X_target)

        # Estimate Wasserstein Distance
        cost = D_inter
        T_w = optimal_transport.optimize_couplings_sinkhorn(
            cost, scale=self.sinkhorn_scale, iterations=self.sinkhorn_iterations
        )
        if self.gw_layout:
            T_w = T_w + self.T_gw * self.gw_layout_coefficient
            T_w = T_w / T_w.sum([-1, -2], keepdims=True)
        D_w = (T_w * D_inter).sum([-1, -2])

        # Scale by sqrt(SNR_t) and constant factor
        scale_t = self.shape_loss_weight * self.noise_schedule.SNR(t).sqrt().clamp(
            min=1e-3, max=3.0
        )
        neglogp = scale_t * F.softplus(D_w - self.shape_loss_cutoff)
        U = U + neglogp
        return X, C, O, U, t


class ProCapConditioner(Conditioner):
    """Natural language conditioning for protein backbones.

    This conditioner uses an underlying `ProteinCaption` model to determine the
    likelihood of a noised structure corresponding to a given caption. Captions
    can be specified as corresopnding to a particular chain of the structure, or
    to the entire complex. The encoded structures and captions are passed to the
    model together, and the output loss that adjusts the energy is the masked
    cross-entropy over the caption tokens.

    Args:
        caption (str): Caption for the conditioner. Currently, a separate
            conditioner should be constructed for each desired caption, even
            with a single `ProteinCaption` model.
        chain_id (int): The 1-indexed chain to which the caption corresponds, or
            -1 for captions corresponding to the entire structure. The provided
            checkpoints are trained with UniProt captions for chain_id > 0 and
            PDB caption for chain_id = -1. Regardless of whether the caption is
            specific to one chain, the conditioner acts on the entire structure.
        weight (float): Overall factor by which the caption gradient is scaled.
        model (generate.models.procap.ProteinCaption, optional): The
            input model whose likelihoods are used. If not given, defaults to
            the checkpoint used for the paper.
        use_sequence (bool): Whether to use input sequence, default False.
        device (str, optional): Device on which to store model. If not given,
            GPU will be used when available.
    Inputs:
        X (torch.Tensor): Structure tensor with shape
            `(batch_size, num_residues, 4, 3)`.
        C (torch.LongTensor): Chain map tensor with shape
            `(batch_size, num_residues)`
        O (torch.Tensor, optional): One-hot tensor allowing the input of
            sequence information, of shape (1, num_residues, num_alphabet).
        U (torch.Tensor): Energy tensor with shape `(batch_size,)`.
        t (Union[torch.Tensor, float]): Time tensor with shape `(batch_size,)`
            or a scalar.

    Outputs:
        X_out (torch.Tensor): Unchanged structure tensor with shape
            `(batch_size, num_residues, 4, 3)`.
        C_out (torch.LongTensor): Unchanged chain map tensor with shape
            `(batch_size, num_residues)`.
        O_out (torch.Tensor, optional): One-hot tensor allowing the output of
            sequence information, of shape (1, num_residues, num_alphabet).
        U_out (torch.Tensor): Modified energy tensor with shape `(batch_size,)`.
        t_out (Union[torch.Tensor, float]): Modified time tensor with shape
            `(batch_size,)` or a scalar.
    """

    def __init__(
        self,
        caption: str,
        chain_id: int,
        weight: float = 10,
        model: Union[ProteinCaption, str] = "named:public",
        use_sequence: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        if isinstance(model, ProteinCaption):
            self.model = model
        elif isinstance(model, str):
            self.model = procap.load_model(
                model, device=device, strict_unexpected=False
            )
        self.model.eval()
        if device is None:
            if torch.cuda.is_available():
                self.model.to("cuda")
        else:
            self.model.to(device)
        self.caption = caption
        self.register_buffer("chain_id", torch.Tensor([int(chain_id)]))
        self.weight = weight
        self.use_sequence = use_sequence

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        loss = self.model(
            X,
            C,
            [self.caption] * X.shape[0],
            self.chain_id.to(X.device).expand(X.shape[0], 1),
            O=O if self.use_sequence else None,
            add_noise=False,
            t=t,
        ).loss
        U = U + self.weight * loss
        return X, C, O, U, t


class ProClassConditioner(Conditioner):
    """
    ProClassConditioner:
    A Chroma Conditioning module which can specify chain level annotations for fold,
    function, and organism. The current labels that can be conditioned on are:

    * cath: protein domain annotations from <https://www.cathdb.info/>. Annotation
        examples include `2`, `2.40`, `2.40.155`.
    * funfam: domain level functional annotations.
    * organism: the organism of origin of a protein. Annotation examples include `Homo
        sapiens (Human)`, `Escherichia coli`,  `Pseudomonas putida (Arthrobacter
        siderocapsulatus)`, `Rattus norvegicus (Rat)`
    * pfam: protein family annotations which represent domain level structural
        characteristics.

    For a complete list of valid value label pairs import the value dictionary from the
     `GraphClassifierLoader` in the zoo.

    Note:
        This conditioner is a research preview. Conditioning with it can be inconsistent
        and depends on the relative prevalence of a given label in the dataset.
        With repeated tries it will produce successful results for more abundant labels.
        Please see the supplement to the paper for details. This is currently not
        recommended for production use. The most reproducible labels are C level
        annotations in cath, (e.g. `1`,`2`,`3`).

    Args:
        label (str): The annotation to condition on in the set [cath, funfam, pfam,
            organism, secondary_structure].
        value (str, optional): The particular annotation string to use. For a complete
             list of values for a given label use the static method
                :meth:`possible_conditions`. Defaults to None.
        model (GraphClassifier, optional): A ProClass instance to use for conditioning.
             if None is provided the recommended model is automatically loaded. Defaults
             to None.
        weight (float, optional): The weighting of the conditioner relative to the
             backbone model. Defaults to 1.
        max_norm (float, optional): The maximum magnitude of the gradient, above which
             the magnitude is clipped. Defaults to None.
        renormalize_grad (bool, optional): Whether to renormalize gradient to have
            overall variance `weight`.
        use_sequence (bool, optional): Whether to use input sequence, default False.
        device (str, optional): the device to put the conditioner on, accepts `cpu`
            and `cuda`. If None is provided it will automatically try to put it on the
             GPU if possible. Defaults to None.
        debug (bool, optional): provides gradient values during optimization for
            setting weights and debugging.
    """

    def __init__(
        self,
        label: str,
        value: Union[Optional[str], torch.Tensor] = None,
        model: Union[GraphClassifier, str] = "named:public",
        weight: float = 5,
        max_norm: Optional[float] = 20,
        renormalize_grad: Optional[bool] = False,
        use_sequence: bool = False,
        device: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.label = label
        self.value = value
        self.max_norm = max_norm
        self.renormalize_grad = renormalize_grad
        self.weight = weight
        self.use_sequence = use_sequence
        self.debug = debug

        if isinstance(model, str):
            self.proclass_model = graph_classifier.load_model(model, device=device)
        elif isinstance(model, GraphClassifier):
            self.proclass_model = model
        self.proclass_model.eval()

        # Move Model to the indicated device
        if device is None:
            if torch.cuda.is_available():
                self.proclass_model.to("cuda")
        else:
            self.proclass_model.to(device)

        self._transform_inputs()
        self._validate_inputs()

    def _transform_inputs(self):
        # Automatically handle heirarchical inputs in the format X.Y.Z.W
        if self.label.lower() in ["cath", "funfam"]:
            self.label = self.label.lower()
            self.label += "_" + str(self.value.count("."))

        # Correct Capitalization
        if self.label.lower() == "organism":
            self.label = "Organism"

        # Support Normative PFam IDs
        if self.label.lower() == "pfam":
            valid_values = self.proclass_model.class_config["pfam"]["tokens"]
            if self.value.count(".") == 0:
                valid_ids = [s for s in valid_values if self.value in s]
                if len(valid_ids) == 1:
                    self.value = valid_ids[0]
                else:
                    raise Exception(f"Invalid Value {self.value} for {self.label}.")

    def _validate_inputs(self):
        # Check Labels
        valid_labels = list(self.proclass_model.heads["chain"].keys())
        valid_labels += list(self.proclass_model.heads["first_order"])
        if self.label not in valid_labels:
            valid_label_str = ", ".join(valid_labels)
            raise Exception(f"Invalid Label. Label must be one of: {valid_label_str}.")

        # Check Values
        if self.label in list(self.proclass_model.heads["chain"].keys()):
            valid_values = self.proclass_model.class_config[self.label]["tokens"]
            if self.value not in valid_values:
                raise Exception(f"Invalid Value {self.value} for {self.label}.")

    def _proclass_neglogp(self, X, C, t, label, value=None, O=None, mask=None):
        """
        Args:
            X (torch.tensor): (batch, num_res, 4, 3) or (batch, num_res, 14, 3)
            C (torch.tensor): (batch, num_res)
            t (float): 0 < t <= 1
            label (string): class label to condition on, chosen from
                `self.class_config.keys()`
            mask (torch.tensor): (optional) bool tensor of shape (batch, num_res) for
                first order scores, (batch, num_chains) for hain-based scores, and (
                batch, num_res, num_res) for second order scores. The order of your
                score can be determined by inspecting self.class_config[label]['level']
            value (string): (optional) the token-based representation of the value you
                 would like to condition `label` on, you can select options from
                `self.class_config[label]['tokens']` for all scores except `is_interface`
                or `contact` for which you should leave a `value` of None.
            O (torch.tensor): one-hot sequence tensor of size (batch, num_res, num_alphabet)
        """
        self.proclass_model.eval()
        _bak = self.proclass_model.encoder.checkpoint_gradients
        self.proclass_model.encoder.checkpoint_gradients = False

        level = self.proclass_model.class_config[label]["level"]
        head, pool = self.proclass_model.heads[level][label]

        node_h, edge_h, edge_idx, mask_i, mask_ij = self.proclass_model.encode(
            X, C, O if self.use_sequence else None, t
        )

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

        if self.proclass_model.class_config[label]["loss"] == "ce":
            neglogp = node_h.log_softmax(dim=-1).mul(-1)
        else:
            neglogp = node_h.sigmoid().log().mul(-1)

        if level == "chain":
            index = (
                self.proclass_model.class_config[label]["tokenizer"][value]
                if value is not None
                else 0
            )
            neglogp = neglogp[..., index][c_mask].sum()
        elif level == "first_order":
            if isinstance(value, str):
                index = torch.LongTensor(
                    [
                        self.proclass_model.class_config[label]["tokenizer"][v]
                        for v in value
                    ]
                ).to(neglogp.device)
                neglogp = torch.gather(
                    neglogp, 2, index.unsqueeze(0).unsqueeze(2)
                ).sum()
            elif isinstance(
                value, torch.Tensor
            ):  # Mask Tensor is Passed for SS Conditioning
                logp = -1 * neglogp
                masked_log_probs = torch.where(
                    value > 0, logp, -float("inf") * torch.ones_like(logp)
                )
                log_probs_sum = torch.logsumexp(masked_log_probs, dim=-1)
                log_probs_sum = torch.where(
                    value.sum(-1) > 0, log_probs_sum, torch.zeros_like(log_probs_sum)
                )
                neglogp = -1 * log_probs_sum.sum()

        self.proclass_model.encoder.checkpoint_gradients = _bak
        return neglogp

    def _transform_gradient(self, grad, C, t):
        if self.debug:
            print("conditioning grad norm:", grad.norm().item())
        if grad.norm() > 1e-8:  # Don't rescale zero gradients!
            # grad = clip_atomic_magnitudes_percentile(grad,percentile=0.95)
            if self.renormalize_grad:
                scale = (
                    self.weight
                    / self.proclass_model.noise_perturb.noise_schedule.sigma(t).to(
                        C.device
                    )
                )
                grad = scale * (grad / grad.norm())
            else:
                grad = self.weight * grad

            if self.max_norm is not None:
                if grad.norm() > self.max_norm:
                    grad = self.max_norm * (grad / grad.norm())
        if self.debug:
            print("output_grad_norm", grad.norm().item())
        return grad

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        X_input = X + 0.0
        X_input.register_hook(lambda _X: self._transform_gradient(_X, C, t))

        neglogp = self._proclass_neglogp(
            X_input, C, t, self.label, value=self.value, O=O
        )
        if self.debug:
            print("time", t.item(), "neglogp:", neglogp.item())
        return X, C, O, neglogp + U, t


class SubstructureConditioner(Conditioner):
    """
    SubstructureConditioner:
    A Chroma Conditioning module which can specifiy a subset of residues for which to
    condition on absolute atomic coordinates, see supplementary section M for more
    details.

    Args:
        protein (generate.data.protein.Protein): Object containing structural
            information to condition on.
        backbone_model (generate.models.GraphBackbone): The `GraphBackbone`
            object one is conditioning
        selection (str): A string specifying the selection to condition on, will be
            retrieved by `protein.get_mask(selection)` . The selection can be defined
            from a set of residue indices `indices` by
            `protein.sys.setSelection(indices, selection)`.
        rg (bool, optional): Whether or not to add reconstruction guidance gradients,
             see supplementary section M for a discussion. This can reduce incidence of
            clashes / bond violations / discontinuities at the cost of inference time
            and some stability.
        weight (float, optional): Overall weight of the reconstruction guidance term
            (untransformed).
        tspan (Tuple[float, float], optional): Time interval over which to appl
            y reconstruction guidance, can be helpful to turn off at times close to
            zero. tspan[0] should be < tspan[1].
        weight_max (float, optional): Final rg gradient is rescaled to have `scale`
            variance, where `scale` is clamped to have a maximum value of `max_weight`.
        gamma (Optional[float]): Gamma inflates the translational degree of freedom
             of the underlying conditional multivariate normal, making it easier for
            Chroma to move the center of mass of the infilled samples.
            Setting to [0.01, 0.1, 1.0] is a a plausible place to start to increase
            sample Rg.
        center_init (Optional[bool]): Whether to center the input structural data
    """

    def __init__(
        self,
        protein: Protein,
        backbone_model: GraphBackbone,
        selection: str,
        rg: bool = False,
        weight: float = 1.0,
        tspan: Tuple[float, float] = (1e-1, 1),
        weight_max: float = 3.0,
        gamma: Optional[float] = None,
        center_init: Optional[bool] = True,
    ):
        super().__init__()
        self.protein = protein
        self.backbone_model = backbone_model
        X, C, S = protein.to_XCS()
        X = X[:, :, :4, :]
        if center_init:
            X = backbone.center_X(X, C)
        D = protein.get_mask(selection).bool()
        self.base_distribution = self.backbone_model.noise_perturb.base_gaussian
        self.noise_schedule = self.backbone_model.noise_perturb.noise_schedule
        self.conditional_distribution = mvn.ConditionalBackboneMVNGlobular(
            covariance_model=self.base_distribution.covariance_model,
            complex_scaling=self.base_distribution.complex_scaling,
            X=X,
            C=C,
            D=D,
            gamma=gamma,
        )
        X = self.conditional_distribution.sample(1)
        self.tspan = tspan
        self.weight = weight
        self.weight_max = weight_max
        self.rg = rg
        self.register_buffer("X", X)
        self.register_buffer("C", C)
        self.register_buffer("S", S)
        self.register_buffer("D", D)

    def _transform_gradient(self, grad, C, t):
        mask = (t > self.tspan[0]) & (t < self.tspan[1])
        scale = (
            (self.weight / self.noise_schedule.sigma(t).to(C.device))
            .clamp(max=self.weight_max)
            .masked_fill(~mask, 0.0)
        )
        grad = scale * grad / grad.square().mean(dim=[1, 2, 3], keepdim=True).sqrt()
        return grad

    def _rg_loss(self, X0, C):
        C_clamp = torch.where(self.D, C, -C.abs())
        X0 = backbone.impute_masked_X(X0, C_clamp)
        X_target = backbone.impute_masked_X(self.X.repeat(X0.size(0), 1, 1, 1), C_clamp)
        loss = (
            self.base_distribution._multiply_R_inverse(X_target - X0, C).square().sum()
        )
        return loss

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        loss = 0.0
        Z = self.base_distribution._multiply_R_inverse(X, C)
        X = self.conditional_distribution.sample(Z=Z)

        # Reconstruction guidance
        if self.rg:
            X_input = X + 0.0
            X_input.register_hook(lambda _X: self._transform_gradient(_X, C, t))
            X0 = self.backbone_model.denoise(X_input, C, t)
            loss = self._rg_loss(X0, C)
        U = U + loss
        return X, C, O, U, t


class SymmetryConditioner(Conditioner):
    """A class that implements a symmetry conditioner for a protein structure.

    A symmetry conditioner applies a set of symmetry operations to a protein structure
    and enforces constraints on the resulting conformations. It can be used to model
    symmetric complexes or assemblies of proteins.

    Args:
        G (torch.Tensor or str): A tensor of shape (n_sym, 3, 3) representing the symmetry
            operations as rotation matrices.
        num_chain_neighbors (int): The number of neighbors to consider for each chain in
            the complex.
        freeze_com (bool): Whether to freeze the center of mass of the complex during
            optimization.
        grad_com_surgery (bool): Whether to apply gradient surgery to remove the center
            of mass component from the gradient.
        interface_restraint (bool): Whether to apply a flat-bottom potential to restrain
            the distance between neighboring chains in the complex.
        restraint_grad (bool): Whether to include the restraint gradient in the total
            gradient.
        enable_rigid_drift (bool): Whether to enable rigid body drift correction for
            the complex.
        canonicalize (bool): Whether to canonicalize the chain order and orientation
            of the complex.

    Inputs:
        X (torch.Tensor): Data tensor with shape `(batch_size, num_residues, 4, 3)`.
        C (torch.LongTensor): Conditioning tensor with shape `(batch_size,
             num_residues)`.
        O (torch.Tensor): One-hot sequence with shape
            `(batch_size, num_residues, num_alphabet)`.
        U (torch.Tensor): Energy tensor with shape `(batch_size,)`.
        t (Union[torch.Tensor, float]): Time tensor with shape `(batch_size,)` or a
            scalar.

    Outputs:
        X_out (torch.Tensor): Modified data tensor with shape `(batch_size, num_residues
            , 4, 3)`.
        C_out (torch.LongTensor): Modified conditioning tensor with shape `(batch_size,
             num_residues)`.
        O_out (torch.Tensor, optional): Modified one-hot tensor with sequence
            of shape `(batch_size, num_residues, num_alphabet)`.
        U_out (torch.Tensor): Modified Energy tensor with shape `(batch_size,)`.
        t_out (Union[torch.Tensor, float]): Modified time tensor with shape `(batch_size
            ,)` or a scalar.
    """

    def __init__(
        self,
        G,
        num_chain_neighbors,
        freeze_com=False,
        grad_com_surgery=False,
        interface_restraint=False,
        restraint_grad=False,
        enable_rigid_drift=True,
        canonicalize=True,
        seed_idx=None,
    ):
        super().__init__()

        if type(G) == str:
            self.G = symmetry.get_point_group(G)
        else:
            self.G = G

        self.num_chain_neighbors = num_chain_neighbors
        self.freeze_com = freeze_com
        self.grad_com_surgery = grad_com_surgery
        self.interface_restraint = interface_restraint
        self.restraint_grad = restraint_grad
        self.enable_rigid_drift = enable_rigid_drift
        self.canonicalize = canonicalize
        self.seed_idx = seed_idx

        if num_chain_neighbors > self.G.shape[0] - 1:
            self.num_chain_neighbors = self.G.shape[0] - 1

        self.potts_symmetry_order = self.num_chain_neighbors + 1

    def flat_bottom_potential(self, r, r0, k, d):
        condition = torch.abs(r - r0) < d
        return torch.where(
            condition, torch.zeros_like(r), k * (torch.abs(r - r0) - d) ** 2
        )

    def translational_scaling(self, C):
        """Compute parameters for enforcing Rg scaling"""

        # Build expanded map per chain
        C_expand = C.unsqueeze(-1).expand(-1, -1, 4)
        C_atomic = C_expand.reshape(C.shape[0], -1)
        C_mask_all = backbone.expand_chain_map(torch.abs(C_atomic))[..., None]

        a = 1.5587407701549267  # TODO: this can change if our prior changed
        nu = 2.0 / 5.0
        r = 2.0 / 3.0

        # C_mask_all is ()
        # Monomer and complex sizes (batch, {chains})
        C_mask = C_mask_all.squeeze(-1)
        N_per_chain = C_mask.sum(1)
        N_per_complex = C_mask.sum([1, 2])

        # Compute expected Rg^2 values per complex
        Rg2_complex = (r**2) * N_per_complex ** (2.0 * nu)
        Rg2_chain = (r**2) * N_per_chain ** (2.0 * nu)

        # Compute OU process parameters
        N_per_chain = torch.clip(N_per_chain, 1, 1e6)

        # Compute size-weighted average Rg^2 per chain
        Rg2_chain_avg = (N_per_chain * Rg2_chain).sum(1) / (N_per_chain.sum(1) + 1e-5)
        Rg2_centers_of_mass = torch.clip(Rg2_complex - Rg2_chain_avg, min=1)
        Rg_centers_of_mass = torch.sqrt(Rg2_centers_of_mass)

        N_chains_per_complex = (C_mask.sum(1) > 0).sum(1)
        # Correct for the fact that we are sampling chains IID (not
        # centered) but want to control centered Rg
        std_correction = torch.sqrt(
            N_chains_per_complex / (N_chains_per_complex - 1).clamp(min=1)
        )
        marginal_COM_std = std_correction * Rg_centers_of_mass

        return marginal_COM_std

    def expand_C(self, C, k):
        missing = C < 0
        Cs = []
        for i in range(k):
            newC = C.abs() + C.unique().max() * i
            newC[missing] = -newC[missing]
            Cs += [newC]
        C = torch.cat(Cs, dim=1)
        return C

    def expand_S(self, S, k):
        S = torch.cat([S] * k, dim=1)
        return S

    def expand_au(self, X, C, G, scale=True):
        n_atoms_per_res = X.shape[-2]

        C_au = C
        # compute new chain mask
        C = self.expand_C(C, G.shape[0])

        #  compute COM inflation due to tesselate
        if scale:
            if self.enable_rigid_drift:
                translate_ratio = self.translational_scaling(C) / (
                    self.translational_scaling(C_au)
                    * (self.num_chain_neighbors + 1) ** 0.5
                )

            else:
                translate_ratio = 1.0

            mask_expand = (
                (C_au > 0)
                .float()
                .reshape(list(C_au.shape) + [1, 1])
                .expand([-1, -1, n_atoms_per_res, -1])
            )
            X_com = (mask_expand * X).sum([1, 2], keepdims=True) / (
                mask_expand.sum([1, 2], keepdims=True)
            )

            X_shifted_mean = X_com * translate_ratio
            X = (X - X_com) + X_shifted_mean

        X = torch.einsum("gij,braj->bgrai", G, X).reshape(1, -1, n_atoms_per_res, 3)

        return X, C

    def _postprocessing_(
        self, protein: Protein, output_dict: Optional[dict] = None
    ) -> Union[Protein, Tuple[Protein, dict]]:
        X, C, S = protein.to_XCS(all_atom=True)
        X_sym, C_sym, S_sym = self.symmetrize_output(X, C, S)
        protein_sym = Protein.from_XCS(X_sym, C_sym, S_sym)

        if output_dict is None:
            return protein_sym
        else:
            trajectory = output_dict["trajectory"]
            traj_sym, C_sym, S_sym = self.symmetrize_output(
                trajectory.to_XCS_trajectory()[0], C, S
            )
            trajectory_sym = Protein.from_XCS_trajectory(traj_sym, C_sym, S_sym)
            output_dict["trajectory"] = trajectory_sym

            return protein_sym, output_dict

    def center_X(self, X, C):
        mask_expand = (
            (C > 0).float().reshape(list(C.shape) + [1, 1]).expand([-1, -1, 4, -1])
        )

        # compute mean based on backbone coordinates
        X_mean = (mask_expand * X[:, :, :4, :]).sum([1, 2], keepdims=True) / (
            mask_expand.sum([1, 2], keepdims=True)
        )
        X_centered = X - X_mean

        return X_centered

    def symmetrize_output(self, X, C, S):
        if type(X) == torch.Tensor:
            assert len(X.shape) == 4
            X = [X]

        n_chains = (
            self.num_chain_neighbors + 1
            if self.num_chain_neighbors + 1 < self.G.shape[0]
            else self.G.shape[0]
        )

        C_seed = C.reshape(1, n_chains, -1)[:, 0]
        S_seed = S.reshape(1, n_chains, -1)[:, 0]

        traj = []
        for each in X:
            n_atoms_per_res = each.shape[-2]
            X_seed = each.reshape(1, n_chains, -1, n_atoms_per_res, 3)[:, 0]
            X_tess, C_tess = self.expand_au(X_seed, C_seed, self.G, scale=False)
            S_tess = self.expand_S(S_seed, k=self.G.shape[0])
            X_tess = self.center_X(X_tess, C_tess)
            traj.append(X_tess)

        if len(traj) == 1:
            traj = traj[0]

        return traj, C_tess, S_tess

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        self.G = self.G.to(X.device)

        if self.grad_com_surgery or self.freeze_com:
            X_tess, C_tess = self.expand_au(X, C, self.G, scale=False)
        else:
            X_tess, C_tess = self.expand_au(X, C, self.G, scale=True)

        X_subdomain, C_subdomain, subdomain_idx, seed_idx = symmetry.subsample(
            X_tess, C_tess, self.G, self.num_chain_neighbors, seed_idx=self.seed_idx
        )

        if self.canonicalize:
            X_canonical = torch.einsum(
                "ij,barj->bari", self.G[seed_idx].inverse(), X_subdomain
            )
        else:
            X_canonical = X_subdomain

        def grad_surgery(dx):
            if self.grad_com_surgery:
                # inflate COM signal
                translate_ratio = self.translational_scaling(C_tess) / (
                    self.translational_scaling(C)
                )

                dx_com = dx.mean([0, 1, 2])
                dx_com_scale = dx_com * translate_ratio
                dx = (dx - dx_com) + dx_com_scale

            if self.freeze_com:
                dx = backbone.center_X(dx, C_subdomain)

            # averaging grad
            dx = dx / (self.num_chain_neighbors + 1)
            return dx

        X_canonical.register_hook(grad_surgery)

        # Tesselate sequence
        symmetry_order = C_subdomain.shape[1] // C.shape[1]
        O_subdomain = (
            O[:, None, :, :]
            .expand([-1, symmetry_order, -1, -1])
            .reshape(list(C_subdomain.shape) + [O.shape[-1]])
        )
        return X_canonical, C_subdomain, O_subdomain, U, t


class ScrewConditioner(Conditioner):
    """A class that implements a screw conditioner for a protein structure.

    A screw conditioner applies a screw transformation to a protein structure
    and repeats it for a given number of times. It can be used to model
    helical or cyclic symmetry of proteins.

    Attributes:
        theta (float): The angle of rotation about the z-axis in radians.
        tz (float): The translation along the z-axis.
        M (int): The number of repetitions of the screw transformation.

    Methods:
        prepare_transforms(N_repeat): Compute the rotation matrices and translation
            vectors for the screw transformation.
        expand_C(C, k): Expand a chain tensor C by duplicating each chain k times with
             different labels.
        rebuild(X, C, M): Rebuild a protein structure with the screw transformation.
        forward(X, C, U, t): Apply the screw transformation to a protein structure and
            return modified tensors.

    Inputs:
        X (torch.Tensor): Data tensor with shape `(batch_size, num_residues, 4, 3)`.
        C (torch.LongTensor): Chain tensor with shape `(batch_size, num_residues)`.
        O (torch.Tensor): One-hot sequence with shape
            `(batch_size, num_residues, num_alphabet)`.
        U (torch.Tensor): Energy tensor with shape `(batch_size,)`.
        t (Union[torch.Tensor, float]): Time tensor with shape `(batch_size,)` or a s
            calar.

    Outputs:
        X_out (torch.Tensor): Modified data tensor with shape `(batch_size,
             num_residues * M, 4, 3)`.
        C_out (torch.LongTensor): Modified chain tensor with shape `(batch_size,
            num_residues * M)`.
        O_out (torch.Tensor, optional): Modified one-hot tensor with sequence
            of shape `(batch_size, num_residues, num_alphabet)`.
        U_out (torch.Tensor): Modified energy tensor with shape `(batch_size,)`.
        t_out (Union[torch.Tensor, float]): Modified time tensor with shape
            `(batch_size,)` or a scalar.
    """

    def __init__(self, theta, tz, M):
        super().__init__()
        self.theta = torch.Tensor([theta]).squeeze()
        self.tz = tz
        self.M = M
        self.Rs, self.ts = self.prepare_transforms(M)

    def prepare_transforms(self, N_repeat):
        # Rotation matrix for rotation about the z-axis
        R_base = torch.tensor(
            [
                [torch.cos(self.theta), -torch.sin(self.theta), 0],
                [torch.sin(self.theta), torch.cos(self.theta), 0],
                [0, 0, 1],
            ]
        )

        t_base = torch.tensor([0, 0, self.tz])

        Rs = []
        ts = []

        R = R_base
        t = t_base
        for _ in range(N_repeat):
            R = R @ R_base
            t = t + t_base

            Rs.append(R[None])
            ts.append(t[None])

        Rs = torch.cat(Rs, dim=0)
        ts = torch.cat(ts, dim=0)

        return Rs, ts

    def expand_C(self, C, k):
        Cs = []
        for i in range(k):
            newC = C + C.unique().max() * i
            Cs += [newC]
        C = torch.cat(Cs, dim=1)
        return C

    def rebuild(self, X, C, M, au_len):
        Rs, ts = self.prepare_transforms(M)
        X = torch.einsum("mji,bari->bmarj", Rs.to(X.device), X[:, :au_len])
        X_screw = X + ts.to(X.device)[None][:, :, None, None, :]
        C_screw = self.expand_C(C[:, :au_len], Rs.shape[0])
        X_screw = X_screw.reshape(1, -1, 4, 3)
        return X_screw, C_screw

    @validate_XC()
    def forward(self, X, C, O, U, t):
        X.requires_grad = True
        X = torch.einsum("mji,bari->bmarj", self.Rs.to(X.device), X)
        X_screw = X + self.ts.to(X.device)[None][:, :, None, None, :]
        C_screw = self.expand_C(C, self.M)

        def grad_surgery(dx):
            dx = dx / (self.M)
            return dx

        X.register_hook(grad_surgery)
        X_screw = X_screw.reshape(1, -1, 4, 3)

        # Tesselate sequence
        symmetry_order = C_screw.shape[1] // C.shape[1]
        O_screw = (
            O[:, None, :, :]
            .expand([-1, symmetry_order, -1, -1])
            .reshape(list(C_screw.shape) + [O.shape[-1]])
        )
        return X_screw, C_screw, O_screw, U, t


class InflateConditioner(Conditioner):
    """Inflate conditioner

    This class inherits from the Conditioner class and defines a specific conditioner
      that inflates shift the COM of X based on a vector v and a scalar.

    Args:
        v (torch.Tensor): Vector to add to X with shape `(num_residues, 4, 3)`.
        scale (float): Scale factor for v.

    Inputs:
        X (torch.Tensor): Data tensor with shape `(batch_size, num_residues, 4, 3)`.
        C (torch.LongTensor): Conditioning tensor with shape `(batch_size,
            num_residues)`.
        O (torch.Tensor): One-hot sequence with shape
            `(batch_size, num_residues, num_alphabet)`.
        U (torch.Tensor): Noise tensor with shape `(batch_size,)`.
        t (Union[torch.Tensor, float]): Time tensor with shape `(batch_size,)` or a
            scalar.

    Outputs:
        X_out (torch.Tensor): Modified data tensor with shape `(batch_size, num_residues,
            4, 3)`.
        C_out (torch.LongTensor): Modified conditioning tensor with shape `(batch_size,
             num_residues)`.
        O_out (torch.Tensor, optional): Modified one-hot tensor with sequence
            of shape `(batch_size, num_residues, num_alphabet)`.
        U_out (torch.Tensor): Modified noise tensor with shape `(batch_size,)`.
        t_out (Union[torch.Tensor, float]): Modified time tensor with shape
          `(batch_size,)` or a scalar.
    """

    def __init__(self, v: torch.Tensor, scale: float):
        super().__init__()
        self.v = v / v.norm()
        self.scale = scale

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        X.requires_grad = True
        X = X + self.v.to(X.device) * self.scale
        return X, C, O, U, t


class RgConditioner(Conditioner):
    """Conditioners that penalized backbones for having Rg deviated from the expected Rg
      Scaling. The penalty function takes the form of a flat bottom potential

        penalty = || ReLU( || Rg(X, C) - Rg_ceiling_scale * expected_Rg(C) || ) ||^2

    Args:
        scale (float): Scale factor for the penalty
        Rg_ceiling_scale (float): the flat bottom potentialy width, needs to be larger
            than 1.
        complex_rg (bool): whether compute expected Rg based on the complex Rg scaling.
            If True, expected Rg will be computed by treating the entire complex as if
            it is a single cahin. If False, expected Rg will be computed for individual
            chains

    Inputs:
        X (torch.Tensor): Data tensor with shape `(batch_size, num_residues, 4, 3)`.
        C (torch.LongTensor): Conditioning tensor with shape `(batch_size,
            num_residues)`.
        O (torch.Tensor): One-hot sequence with shape
            `(batch_size, num_residues, num_alphabet)`.
        U (torch.Tensor): Noise tensor with shape `(batch_size,)`.
        t (Union[torch.Tensor, float]): Time tensor with shape `(batch_size,)` or a
            scalar.

    Outputs:
        X_out (torch.Tensor): Modified data tensor with shape `(batch_size, num_residues,
            4, 3)`.
        C_out (torch.LongTensor): Modified conditioning tensor with shape `(batch_size,
             num_residues)`.
        O_out (torch.Tensor, optional): Modified one-hot tensor with sequence
            of shape `(batch_size, num_residues, num_alphabet)`.
        U_out (torch.Tensor): Modified noise tensor with shape `(batch_size,)`.
        t_out (Union[torch.Tensor, float]): Modified time tensor with shape
          `(batch_size,)` or a scalar.
    """

    def __init__(
        self,
        scale=1.0,
        Rg_ceiling_scale=1.5,
        complex_rg=False,
    ):
        super().__init__()
        self.eps = 1e-5
        self.scale = scale
        self.Rg_ceiling_scale = Rg_ceiling_scale
        self.complex_rg = complex_rg

    def means_per_chain(self, _X, _C, eps=1e-5):
        """Compute center of mass for each chain in a complex"""
        # (B,N) => (B,N,C) => (B,N,C,A,X)
        mask_chains = (expand_chain_map(_C) > 0).float()
        mask_chains_expand = mask_chains[..., None, None]
        X_masked = mask_chains_expand * _X.unsqueeze(2)
        # Compute per chain means
        X_mean_chains = X_masked.sum([1, 3], keepdims=True) / (
            4 * mask_chains_expand.sum([1, 3], keepdims=True) + eps
        )
        # Compute per complex mean
        X_mean_complex = X_masked.sum([1, 2, 3], keepdims=True) / (
            4 * mask_chains_expand.sum([1, 2, 3], keepdims=True) + eps
        )
        return X_masked, X_mean_chains, X_mean_complex, mask_chains

    def expected_Rg(self, N):
        """compute expected Rg"""
        nu = 2.0 / 5.0
        r = 2.0

        return ((r**2) * N ** (2.0 * nu)) ** 0.5

    def compute_Rg(
        self,
        X,
        C,
    ):
        """compute Rg with X and C"""
        X.requires_grad = True
        X_masked, X_mean_chains, X_mean_complex, mask_chains = self.means_per_chain(
            X, C
        )

        mask_chains_expand = mask_chains[..., None]
        r2_i = mask_chains_expand * (X_masked - X_mean_chains).square().sum(-1)

        r2_i_mean = (r2_i + self.eps).mean(-1).sum(1) / (mask_chains.sum(1) + self.eps)

        r_i_rms = torch.sqrt(r2_i_mean + self.eps)

        return r_i_rms

    @validate_XC()
    def forward(self, X, C, O, U, t):
        if self.complex_rg:
            C_tmp = torch.ones_like(C)
        else:
            C_tmp = C

        # Compute expected Rg
        N_chain = expand_chain_map(torch.abs(C_tmp)).sum(1)
        r_i_rms_expected = self.expected_Rg(N_chain)

        true_rg = self.compute_Rg(X, C_tmp)
        U_Rg = F.relu(true_rg - self.Rg_ceiling_scale * r_i_rms_expected).square()

        U = U + self.scale * U_Rg.sum()
        return X, C, O, U, t


def clip_atomic_magnitudes_percentile(dX, percentile=0.9):
    D = dX.square().sum(-1, keepdims=True).add(1e-5).sqrt()
    D_max = D.quantile(percentile)
    dX_adjust = dX * D.clamp(max=D_max) / D
    return dX_adjust

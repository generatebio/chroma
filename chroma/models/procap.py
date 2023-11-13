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

from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import pad

from chroma.data.xcs import validate_XC
from chroma.layers.basic import FourierFeaturization
from chroma.layers.structure import diffusion
from chroma.models import graph_classifier
from chroma.models.graph_classifier import GraphClassifier
from chroma.models.graph_design import BackboneEncoderGNN
from chroma.utility.model import load_model as utility_load_model
from chroma.utility.model import save_model as utility_save_model


class ProteinCaption(nn.Module):
    """ProCap model for caption likelihood given a noised structure.

    Provides an architecture to model the likelihood of a caption representing a
    protein backbone at an arbitrary diffusion time. For caption processing, it
    uses a pretrained language model from Hugging Face which can be
    user-specified and fine-tuned. For structures, ProteinCaption uses a
    `BackboneEncoderGNN` that encodes a structure and its noise level in the
    embedding space of the language model. There are several options for
    interfacing between the representations of the backbone residues and those
    of the caption.

    A `ProteinCaption` model can be used to conditionally generate backbones
    given a natural language caption, through the creation of a
    `ProCapConditioner` using the model. In this case, the noising parameters
    used for the `ProteinCaption` model should be identical to those that were
    used to train the underlying backbone diffusion model.

    Args:
        lm_id (str): Base language model to pull from Hugging Face.
        gnn_dim_edges (int): Number of edges for structure encoder.
        context_size (int): When encoding structures by chains, specifies the
            maximum number of chains to be used for the encodings. Not used when
            `direct_gnn` is specified.
        context_per_chain (int): When encoding structures by chain, the number
            of context tokens to use per chain. Not used when `direct_gnn` is
            specified.
        gnn_num_neighbors (int): Number of neighbors per node for structure
            encoder.
        gnn_num_layers (int): Number of layers for structure encoder.
        only_encode_caption_chain (bool): Whether to pass structure of only
            chain whose caption is being predicted, as opposed to entire
            structure.
        gnn_embed_ratio (int): Number of context tokens to extract from GNN per
            chain, stacks with gnn_embed_ratio.
        graph_criterion (str): Graph criterion for structure encoder, defines
            how neighbors are chosen. See
            `chroma.models.graph_design.BackboneEncoderGNN` for
            allowed values.
        node_mlp_layers (int): Number of hidden layers for node update function
            of structure encoder.
        node_mlp_dim (int, optional): Dimension of hidden layers for node update
            function of structure encoder, defaults to match output dimension.
        noise_schedule (str): Noise schedule for mapping between diffusion time
            and noise level, see
            chroma.layers.structure.diffusion.DiffusionChainCov for allowed
            values.
        covariance_model (str): Covariance mode for mapping between diffusion
            time and noise level, see
            chroma.layers.structure.diffusion.DiffusionChainCov for allowed
            values.
        noise_complex_scaling (bool): Whether to scale noise for complexes.
        noiseless (bool): Whether to train with denoised structures only, useful
            for debugging but resulting model cannot be used for classifier
            guidance.
        normalize_context_embeddings (bool): Whether to normalize context
            embeddings to an overall length of 1.
        standardize_context_embeddings (bool): Whether to standardize context
            embeddings to have mean 0 and variance 1.
        time_feature_type (str): Method of encoding diffusion timestep.
        time_log_feature_scaling (float): Scaling of diffusion timestep in
            preprocessing when encoding with `time_feature_type = "log_snr"`.
        use_transformer (bool): Whether to use transformer to embed context from
            residue-level GNN outputs.
        classifier_checkpoint (str, optional): Path to pre-trained graph
            classifier checkpoint, whose encoder head will be used for structure
            encoding.
        direct_gnn (bool): Whether to pass in GNN encodings for chains/complexes
            directly to the language model, without any pooling or transformer
            layers.
        classifier_kwargs (dict, optional): Dictionary of parameters to create
            classifier network for encoding. Will override classifier_checkpoint
            if given.


    Inputs:
        X (torch.Tensor): Backbone tensor of shape `(num_batch, num_residues,
            4, 3)`.
        C (torch.Tensor): Chain map of shape `(num_batch, num_residues)`.
            Positions with 0 are masked, positive integers are used for chain
            indices, and negative integers are used for missing residues of the
            chains with indices equal to the corresponding positive integers.
        caption (List[str]): List of captions with length `num_batch`.
        chain_id (torch.Tensor): Chain indices for given captions of shape
            `(num_batch)`. For a caption corresponding to an entire complex, use
            -1.
        O (torch.Tensor, optional): One-hot sequence tensor of shape
            `(num_batch, num_residues, num_alphabet)`. If not given, the loss is
            computed without sequence information.
        add_noise (bool): Whether to randomly add noise to the input backbones.
            If structures are already noised, use `t` instead.
        t (torch.Tensor, optional): Diffusion timesteps corresponding to noisy
            input backbones, of shape `(num_batch)`. Use zeros when passing
            structures without noise.
        by_sample (bool): Whether to return loss per sample, as opposed to
            overall batch loss.

    Outputs:
        loss (Union[transformers.modeling_outputs.CausalLMOutputWithCrossAttentions,
            torch.Tensor]): Loss containing average -log(p) of caption tokens
            given output structures. If `by_sample` is specified, loss is output
            as a tensor of length `(num_batch)`.
    """

    def __init__(
        self,
        lm_id: str = "EleutherAI/gpt-neo-125m",
        gnn_dim_edges: int = 128,
        context_size: int = 16,
        context_per_chain: int = 1,
        gnn_num_neighbors: int = 30,
        gnn_num_layers: int = 3,
        only_encode_caption_chain: bool = False,
        gnn_embed_ratio: int = 1,
        graph_criterion: str = "knn",
        node_mlp_layers: int = 1,
        node_mlp_dim: Optional[int] = None,
        noise_schedule: str = "log_snr",
        covariance_model: str = "globular",
        noise_complex_scaling: bool = False,
        noiseless: bool = False,
        normalize_context_embeddings: bool = False,
        standardize_context_embeddings: bool = False,
        time_feature_type: str = "t",
        time_log_feature_scaling: float = 0.05,
        use_transformer: bool = False,
        classifier_checkpoint: Optional[str] = None,
        direct_gnn: bool = False,
        classifier_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        # Save configuration in kwargs
        self.kwargs = locals()
        self.kwargs.pop("self")
        for key in list(self.kwargs.keys()):
            if key.startswith("__") and key.endswith("__"):
                self.kwargs.pop(key)
        args = SimpleNamespace(**self.kwargs)

        try:
            import transformers
        except ImportError:
            print("Install the hugging face package `transformers` to use ProCap")

        self.context_size = context_size
        self.context_per_chain = context_per_chain
        self.only_encode_caption_chain = only_encode_caption_chain
        self.gnn_embed_ratio = gnn_embed_ratio
        self.normalize_context_embeddings = normalize_context_embeddings
        self.standardize_context_embeddings = standardize_context_embeddings
        self.time_feature_type = time_feature_type
        self.time_log_feature_scaling = time_log_feature_scaling
        self.use_transformer = use_transformer
        self.classifier_checkpoint = classifier_checkpoint
        self.direct_gnn = direct_gnn
        self.classifier_kwargs = classifier_kwargs

        if self.normalize_context_embeddings and self.standardize_context_embeddings:
            print(
                "Warning: both normalization and standardization of context embeddings"
                " are selected, choosing only standardization"
            )
            self.normalize_context_embeddings = False

        # Use Pretrained Tokenizer From Hugging Face
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            lm_id,
            additional_special_tokens=["<|pdb|>", "<|unconditioned|>"],
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )

        # Use Pretrained Language Model From Hugging Face
        self.language_model = transformers.AutoModelForCausalLM.from_pretrained(lm_id)

        # Embedding
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.embedder = self.language_model.get_input_embeddings()
        self.d_model = self.embedder.embedding_dim

        # Standardization for context embeddings
        if self.standardize_context_embeddings:
            self.context_normalization = nn.LayerNorm(
                self.d_model, elementwise_affine=False
            )

        # Transformer for context embeddings
        if self.use_transformer:
            self.transformer = nn.Transformer(
                nhead=8,
                d_model=self.d_model,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=2048,
                batch_first=True,
            )
            if gnn_embed_ratio != 1:
                print(
                    "Warning: both use_transformer and gnn_embed_ratio are set, setting"
                    " gnn_embed_ratio to 1"
                )
                self.gnn_embed_ratio = 1
            if context_per_chain != 1:
                print(
                    "Warning: both use_transformer and context_per_chain are set,"
                    " setting context_per_chain to 1"
                )
                self.context_per_chain = 1
            if not self.only_encode_caption_chain:
                print(
                    "Warning: use_transformer is set but only_encode_caption_chain is"
                    " not, this is unsupported! Setting only_encode_caption_chain to"
                    " True"
                )
                self.only_encode_caption_chain = True

        # Pass in GNN encodings without averaging or transformer
        if self.direct_gnn:
            if gnn_embed_ratio != 1:
                print(
                    "Warning: both direct_gnn and gnn_embed_ratio are set, setting"
                    " gnn_embed_ratio to 1"
                )
                self.gnn_embed_ratio = 1
            if context_per_chain != 1:
                print(
                    "Warning: both direct_gnn and context_per_chain are set, setting"
                    " context_per_chain to 1"
                )
                self.context_per_chain = 1
            if not self.only_encode_caption_chain:
                print(
                    "Warning: direct_gnn is set but only_encode_caption_chain is not,"
                    " this is unsupported! Setting only_encode_caption_chain to True"
                )
                self.only_encode_caption_chain = True
            if self.use_transformer:
                print(
                    "Warning: direct_gnn and use_transformer are both set, turning off"
                    " use_transformer"
                )
                self.use_transformer = False
            if self.context_size is not None:
                print(
                    "Warning: context_size given but not used for direct_gnn, setting"
                    " context_size to None"
                )
                self.context_size = None

        # Use Standard Protein Encoder
        if self.classifier_checkpoint is not None or self.classifier_kwargs is not None:
            if self.classifier_kwargs is not None:
                self.protein_encoder = GraphClassifier(**classifier_kwargs)
            else:
                self.protein_encoder = graph_classifier.load_model(
                    classifier_checkpoint
                )
                self.classifier_kwargs = self.protein_encoder.kwargs
                self.kwargs["classifier_kwargs"] = self.classifier_kwargs
            self.protein_encoder_linear = nn.Sequential(
                nn.Linear(
                    self.protein_encoder.dim_nodes, self.d_model * self.gnn_embed_ratio
                ),
                nn.ReLU(),
            )
        else:
            self.protein_encoder = BackboneEncoderGNN(
                dim_nodes=self.d_model * self.gnn_embed_ratio,
                dim_edges=gnn_dim_edges,
                num_neighbors=gnn_num_neighbors,
                num_layers=gnn_num_layers,
                node_mlp_layers=node_mlp_layers,
                node_mlp_dim=node_mlp_dim,
                graph_criterion=graph_criterion,
            )

        # Use same Noise Layer as in Graph Energy model
        if not noiseless:
            self.noise_generator = diffusion.DiffusionChainCov(
                log_snr_range=(-7.0, 13.5),
                noise_schedule=noise_schedule,
                covariance_model=covariance_model,
                complex_scaling=noise_complex_scaling,
            )
        else:
            self.noise_generator = None
        self.time_features = FourierFeaturization(
            d_input=1,
            d_model=self.d_model * self.gnn_embed_ratio,
            trainable=False,
            scale=16.0,
        )

        # Embed Tokens for 21 Residue Possibilities
        self.sequence_embedding = nn.Embedding(22, self.d_model * self.gnn_embed_ratio)

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.Tensor,
        caption: List[str],
        chain_id: torch.Tensor,
        O: Optional[torch.Tensor] = None,
        add_noise: bool = True,
        t: Optional[Union[torch.Tensor, float]] = None,
        by_sample: bool = False,
    ) -> Union[
        "transformers.modeling_outputs.CausalLMOutputWithCrossAttentions", torch.Tensor
    ]:
        if self.noise_generator is None:
            t = torch.zeros(X.shape[0]).to(X.device)

        if isinstance(t, float):
            t = torch.Tensor([t]).to(X.device)

        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.unsqueeze(0)

        if add_noise and self.noise_generator is not None:
            # Add Chain Noise
            X, t = self._noise(X, C)
            assert all(t <= 1) and all(t >= 0), (
                "Noise Temperatures must be between 0 and 1, but got values"
                f" {t[(t > 1) | (t < 0)]}"
            )
        else:
            assert t is not None, "Must pass diffusion timestep if not adding noise!"

        # Encode Protein Context

        if self.classifier_kwargs is None:
            # Aux feature encoding
            node_h = self._time_features(t)
            if O is not None:
                # pad one-hot tensor by two to account for special tokens used
                node_h = node_h + pad(O, (0, 2)) @ self.sequence_embedding.weight.to(
                    X.device
                )
            Xe, _, _, Me, _ = self.protein_encoder.to(X.device)(X, C, node_h_aux=node_h)
        else:
            # TODO: is there a better way to deal with sequence padding tokens when batch size > 1?
            if O is not None and O[:, :, -1].any():
                O = None
            Xe0, _, _, Me, _ = self.protein_encoder.to(X.device).encode(X, C, O, t)
            Xe = self.protein_encoder_linear.to(X.device)(Xe0)

        context_embedding, attention_mask_context = self._encode_context(
            Xe, C, Me, chain_id
        )
        if self.standardize_context_embeddings:
            context_embedding = self.context_normalization.to(Xe.device)(
                context_embedding
            )
        elif self.normalize_context_embeddings:
            context_embedding = torch.nn.functional.normalize(context_embedding, dim=-1)

        # Encode Text Input
        if self.direct_gnn:
            max_caption_tokens = (
                self.tokenizer.model_max_length - context_embedding.shape[1]
            )
        else:
            max_caption_tokens = (
                self.tokenizer.model_max_length
                - (self.context_size - 1)
                * self.gnn_embed_ratio
                * self.context_per_chain
                - 1
            )
        Y, attention_mask_caption = self._tokenize(
            caption, add_stop=True, max_length=max_caption_tokens
        )
        Y = Y.to(X.device)
        attention_mask_caption = attention_mask_caption.to(X.device)
        caption_embedding = self._embed_text(Y)

        # Caption
        inputs_embeds = torch.cat([context_embedding, caption_embedding], dim=1)
        attention_mask = torch.cat(
            [attention_mask_context, attention_mask_caption], dim=1
        )
        labels = torch.cat(
            [
                torch.tensor(-100, device=X.device).expand(
                    attention_mask_context.shape
                ),
                Y * attention_mask_caption + (-100) * (1 - attention_mask_caption),
            ],
            dim=1,
        )

        # returns a transformers.modeling_outputs.CausalLMOutputWithCrossAttentions object
        # can get logits with output.logits
        output = self.language_model.to(X.device).forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )
        if not by_sample:
            return output
        else:  # below code adapted from transformers/modeling_gpt2.py
            shift_logits = output.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ).reshape(X.shape[0], -1)
            return torch.Tensor(
                (loss * (shift_labels != -100).int()).sum(dim=-1)
                / (shift_labels != -100).int().sum(dim=-1)
            )

        return output

    @validate_XC(all_atom=False)
    def _noise(
        self, X: torch.Tensor, C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes in a Structure Tensor X and Chain Tensor C, adds chain noise with quasi-uniformly sampled temperature.
        Returns the noised X and the time steps used."""
        assert self.noise_generator is not None, "Model does not have noising!"
        return [x.to(X.device) for x in self.noise_generator.to(X.device)(X, C)]

    # Taken from graph classifier model
    def _time_features(self, t: torch.Tensor) -> torch.Tensor:
        h = {
            "t": lambda: t,
            "log_snr": lambda: self.noise_generator.noise_schedule.log_SNR(t),
        }[self.time_feature_type]()

        if "log" in self.time_feature_type:
            h = self.time_log_feature_scaling * h

        time_h = self.time_features.to(t.device)(h[:, None, None])
        return time_h

    def _encode_context(
        self, Xe: torch.Tensor, C: torch.Tensor, M: torch.Tensor, polymer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average Pool over Chains after accounting for masking
        input:
            Xe (torch.Tensor): embedding tensor of shape [batch, residue, d_model]
            C (torch.Tensor): chain tensor indexing which chain each residue belongs to [batch, residue]
            M (torch.Tensor): mask tensor of shape [batch, residue]
            polymer_id (int): index in C of chain, or -1 for entire structure, or 0 to apply no conditioning
        """

        Cm = C * M  # Mask Chain Map
        Cm[Cm < 0] = 0  # Remove Negatives from Chain Map

        B, R, Dm = Xe.shape
        pooled_encoding = []
        for x, c, pid in zip(Xe, Cm, polymer_id):
            batch_encoding = []

            # The predict whole complex token is added under this syntax
            if pid == -1:
                pdb_embedding = self._embed_text(
                    self._tokenize(["<|pdb|>"], add_stop=False)[0].to(Xe.device)
                ).squeeze(0)
                batch_encoding.append(pdb_embedding)

            if pid == 0:
                pdb_embedding = (
                    self._embed_text(
                        self._tokenize(["<|unconditioned|>"], add_stop=False)[0]
                    )
                    .squeeze(0)
                    .to(Xe.device)
                )
                batch_encoding.append(pdb_embedding)

            # Power Average Pool By Chain
            if pid != 0:
                if self.only_encode_caption_chain and (pid != -1):
                    cid = self._pid_2_cid(pid, c)
                    residue_mask = c == cid
                    n_residues = residue_mask.sum(-1)
                    if self.use_transformer:
                        encodings = [
                            self.transformer.to(Xe.device)(
                                x[residue_mask].unsqueeze(0),
                                torch.zeros(1, self.context_size, self.d_model).to(
                                    Xe.device
                                ),
                            ).squeeze(0)
                        ]
                    elif self.direct_gnn:
                        encodings = x[residue_mask].unsqueeze(0)
                    else:
                        encodings = [
                            (x[residue_mask].pow(p).sum(0).unsqueeze(0) / n_residues)
                            .abs()
                            .pow(1 / p)
                            * (
                                x[residue_mask].pow(p).sum(0).unsqueeze(0).sign()
                                if p % 2 == 1
                                else 1
                            )
                            for p in range(1, self.context_per_chain + 1)
                        ]
                        encodings = [
                            enc.reshape(self.gnn_embed_ratio, -1) for enc in encodings
                        ]
                    batch_encoding.extend(encodings)
                else:
                    if self.use_transformer or self.direct_gnn:
                        residue_mask = (
                            c > 0
                        )  # just use all embeddings, no chain structure
                        if self.use_transformer:
                            # should have pid == -1 to get here, so need encoding of size context_size - 1 because of <|pdb|> token
                            assert self.only_encode_caption_chain, (
                                "only_encode_caption chain = False not supported when"
                                " use_transformer = True!"
                            )
                            batch_encoding.append(
                                self.transformer.to(Xe.device)(
                                    x[residue_mask].unsqueeze(0),
                                    torch.zeros(
                                        1, self.context_size - 1, self.d_model
                                    ).to(Xe.device),
                                ).squeeze(0)
                            )
                        else:  # direct_gnn
                            batch_encoding.extend(x[residue_mask].unsqueeze(0))
                    else:
                        for cid in torch.unique(c):
                            if cid == 0:
                                continue
                            residue_mask = c == cid
                            n_residues = residue_mask.sum(-1)
                            encodings = [
                                (
                                    x[residue_mask].pow(p).sum(0).unsqueeze(0)
                                    / n_residues
                                )
                                .abs()
                                .pow(1 / p)
                                * (
                                    x[residue_mask].pow(p).sum(0).unsqueeze(0).sign()
                                    if p % 2 == 1
                                    else 1
                                )
                                for p in range(1, self.context_per_chain + 1)
                            ]
                            batch_encoding.extend(
                                [
                                    enc.reshape(self.gnn_embed_ratio, -1)
                                    for enc in encodings
                                ]
                            )

                    # Reorder the chain embedding to caption to be first
                    if pid != -1:
                        first_cid = self._pid_2_cid(pid, c)
                        try:
                            if first_cid != 0:
                                (
                                    batch_encoding[
                                        (first_cid - 1)
                                        * self.gnn_embed_ratio
                                        * self.context_per_chain : (first_cid)
                                        * self.gnn_embed_ratio
                                        * self.context_per_chain
                                    ],
                                    batch_encoding[
                                        0 : self.gnn_embed_ratio
                                        * self.context_per_chain
                                    ],
                                ) = (
                                    batch_encoding[
                                        0 : self.gnn_embed_ratio
                                        * self.context_per_chain
                                    ],
                                    batch_encoding[
                                        (first_cid - 1)
                                        * self.gnn_embed_ratio
                                        * self.context_per_chain : (first_cid)
                                        * self.gnn_embed_ratio
                                        * self.context_per_chain
                                    ],
                                )
                        except IndexError:
                            print(
                                "Problem: tried to switch encodings at positions 0 and"
                                f" {first_cid}, but failed!"
                            )
                            # raise

            pooled_encoding.append(torch.cat(batch_encoding))

        # Pad with Zero Tensor
        X_pooled = torch.nn.utils.rnn.pad_sequence(pooled_encoding, batch_first=True)

        if self.context_size is not None:
            if (
                X_pooled.shape[1]
                > (self.context_size - 1)
                * self.gnn_embed_ratio
                * self.context_per_chain
                + 1
            ):
                print([x.shape for x in pooled_encoding])
                print(polymer_id)
            assert (
                X_pooled.shape[1]
                <= (self.context_size - 1)
                * self.gnn_embed_ratio
                * self.context_per_chain
                + 1
            ), (
                f"Context is of length {X_pooled.shape[1]}, which is larger than the"
                " allowed number of tokens"
                f" {(self.context_size - 1) * self.gnn_embed_ratio * self.context_per_chain + 1};"
                " this will cause the model to behave poorly!"
            )
            if (
                X_pooled.shape[1]
                < (self.context_size - 1)
                * self.gnn_embed_ratio
                * self.context_per_chain
                + 1
                and not self.direct_gnn
            ):
                pad_shape = (
                    (self.context_size - 1)
                    * self.gnn_embed_ratio
                    * self.context_per_chain
                    + 1
                    - X_pooled.shape[1]
                )
                zero_pad = torch.zeros(
                    [B, pad_shape, int(Dm / self.gnn_embed_ratio)], device=Xe.device
                )
                X_pooled = torch.cat([X_pooled, zero_pad], dim=1)

        M_pooled = (X_pooled != 0)[
            :, :, 0
        ]  # This is a bit dangerous because very rarely X_pooled could contain zeros in masked regions...
        return X_pooled, M_pooled

    def _pid_2_cid(self, pid: int, c: int) -> int:
        """This function converts the polymer_entity_id in the pdb to the chain_id in the XCS format of generate."""
        assert pid in c, f"pid value {pid} must be in the chain map!"
        chain_values = torch.unique(c)
        nonzero_chain_values = chain_values[chain_values != 0]
        cid = (nonzero_chain_values == pid).nonzero(as_tuple=True)[0].item() + 1
        return cid

    def _tokenize(
        self, text: list, add_stop: bool = True, max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts list of strings into a padded tensor, returning the tokenized strings as well as the associated masks."""
        if add_stop:
            text = [x + self.tokenizer.eos_token for x in text]
        # Note that there are no stop tokens in truncated sequences
        tokenized_dict = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return tokenized_dict["input_ids"], tokenized_dict["attention_mask"]

    def _embed_text(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        """Embeds tokenized text."""

        return self.embedder.to(tokenized_text.device)(tokenized_text)


def load_model(
    weight_file: str,
    device: str = "cpu",
    strict: bool = False,
    strict_unexpected: bool = True,
) -> ProteinCaption:
    """Loads a ProCap model.

    Args:
        weight_file (str): Path to the saved model weights.
        device (str): Device on which to load the model.
        strict (bool): Whether to require that the keys match between the
            input file weights and the model created from the parameters stored
            in the model kwargs.
        strict_unexpected (bool): Whether to require that there are no
            unexpected keys when loading model weights, as distinct from the
            strict option which doesn't allow for missing keys either. By
            default, we use this option rather than strict for ease of
            development when adding model features.

    Returns:
        model (ProteinCaption): Instance of `ProteinCaption` with loaded
            weights. For inference the returned model should be set to eval mode
            with `model.eval()`.
    """
    return utility_load_model(
        weight_file,
        ProteinCaption,
        device=device,
        strict=strict,
        strict_unexpected=strict_unexpected,
    )


def save_model(
    model: ProteinCaption, weight_file: str, metadata: Optional[dict] = None
) -> None:
    """Save model, including optional metadata.

    Args:
        model (ProteinCaption): An instance of `ProteinCaption`.
        weight_file (str): The destination path for saving model weights.
        metadata (dict): A dictionary of additional metadata to add to the model
            weights. For example, when saving models during training it can be
            useful to store `args` representing the CLI args, the date and time
            of training, etc.
    """
    utility_save_model(model, weight_file, metadata=metadata)

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

"""
Utilities to save and load models with metadata.
"""

import os
import os.path as osp
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import torch

import chroma.utility.api as api
from chroma.constants.named_models import NAMED_MODELS


def save_model(model, weight_file, metadata=None):
    """Save model, including optional metadata.

    Args:
        model (nn.Module): The model to save. Details about the model needed
            for initialization, such as layer sizes, should be in model.kwargs.
        weight_file (str): The destination path for saving model weights.
        metadata (dict): A dictionary of additional metadata to add to the model
            weights. For example, when saving models during training it can be
            useful to store `args` representing the CLI args, the date and time
            of training, etc.
    """
    save_dict = {"init_kwargs": model.kwargs, "model_state_dict": model.state_dict()}
    if metadata is not None:
        save_dict.update(metadata)
    local_path = str(
        Path(tempfile.gettempdir(), str(uuid4())[:8])
        if weight_file.startswith("s3:")
        else weight_file
    )
    torch.save(save_dict, local_path)
    if weight_file.startswith("s3:"):
        raise NotImplementedError("Uploading to an s3 link not supported.")


def load_model(
    weights,
    model_class,
    device="cpu",
    strict=False,
    strict_unexpected=True,
    verbose=True,
):
    """Load model saved with save_model.

    Args:
        weights (str): The destination path of the model weights to load.
            Compatible with files saved by `save_model`.
        model_class: Name of model class.
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
        verbose (bool, optional): Show outputs from download and loading. Default True.

    Returns:
        model (nn.Module): Torch model with loaded weights.
    """

    # Process weights path
    if str(weights).startswith("named:"):
        weights = weights.split("named:")[1]
        if weights not in NAMED_MODELS[model_class.__name__]:
            raise Exception(f"Unknown {model_class.__name__} model name: {weights},")
        weights = NAMED_MODELS[model_class.__name__][weights]["s3_uri"]

    # resolve s3 paths
    if str(weights).startswith("s3:"):
        raise NotImplementedError("Loading Models from an S3 link not supported.")

    # download public models from generate
    if str(weights).startswith("https:"):
        # Decompose into arguments
        parsed_url = urlparse(weights)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        model_name = parse_qs(parsed_url.query).get("weights", [None])[0]
        weights = api.download_from_generate(
            base_url, model_name, force=False, exist_ok=True
        )

    # load model weights
    params = torch.load(weights, map_location="cpu")
    model = model_class(**params["init_kwargs"]).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(
        params["model_state_dict"], strict=strict
    )
    if strict_unexpected and len(unexpected_keys) > 0:
        raise Exception(
            f"Error loading model from checkpoint file: {weights} contains {len(unexpected_keys)} unexpected keys: {unexpected_keys}"
        )
    return model

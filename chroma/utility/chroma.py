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

import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from torch import nn

import chroma
from chroma.constants import AA20
from chroma.data import Protein
from chroma.layers.structure import backbone


def letter_to_point_cloud(
    letter="G",
    width_pixels=35,
    font=os.path.join(
        os.path.dirname(chroma.__path__[0]), "assets/LiberationSans-Regular.ttf"
    ),
    depth_ratio=0.15,
    fontsize_ratio=1.2,
    stroke_width=1,
    margin=0.5,
    max_points=2000,
):
    """Build a point cloud from a letter"""
    depth = int(depth_ratio * width_pixels)
    fontsize = int(fontsize_ratio * width_pixels)

    font = ImageFont.truetype(font, fontsize)
    ascent, descent = font.getmetrics()
    text_width = font.getmask(letter).getbbox()[2]
    text_height = font.getmask(letter).getbbox()[3] + descent

    margin_width = int(text_width * margin)
    margin_height = int(text_height * margin)
    image_size = [text_width + margin_width, text_height + margin_height]

    image = Image.new("RGBA", image_size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text(
        (margin_width // 2, margin_height // 2),
        letter,
        (0, 0, 0),
        font=font,
        stroke_width=stroke_width,
        stroke_fill="black",
    )

    A = np.asarray(image).mean(-1)
    A = A < 100.0
    V = np.ones(list(A.shape[:2]) + [depth]) * A[:, :, None]
    X_point_cloud = np.stack(np.nonzero(V), 1)
    # Uniform dequantization
    X_point_cloud = X_point_cloud + np.random.rand(*X_point_cloud.shape)

    if max_points is not None and X_point_cloud.shape[0] > max_points:
        np.random.shuffle(X_point_cloud)
        X_point_cloud = X_point_cloud[:max_points, :]

    return X_point_cloud


def point_cloud_rescale(
    X, num_residues, neighbor_k=8, volume_per_residue=128.57, scale_ratio=0.4
):
    """Rescale target coordinates to occupy protein-sized volume"""

    # Use heuristic for radius value from the average m-th nearest neighbor
    # This was tuned empirically for target problems (could be optimized on the fly as LB estimate)
    D = np.sqrt(np.square(X[None, :] - X[:, None]).sum(-1))
    radius = 0.5 * np.sort(D, axis=1)[:, neighbor_k].mean()
    diameter = D.max()

    # Estimate initial volume with 2nd order inclusion exclusion
    V = point_cloud_volume(X, radius)

    # Compute target volume, which scales linearly with number of residues
    V_target = num_residues * volume_per_residue
    scale_factor = (scale_ratio * V_target / V) ** (1.0 / 3.0)
    X_rescale = scale_factor * X
    cutoff_D = scale_factor * radius
    return X_rescale, cutoff_D


def point_cloud_volume(X, radius):
    """Estimate the volume of a point cloud given sphere radii"""
    N = X.shape[0]

    # Volume estimation - One body volumes
    V_1 = N * (4.0 / 3.0) * np.pi * radius ** 3

    # Volume estimation - 2nd order overlaps
    D = np.sqrt(np.square(X[None, :] - X[:, None]).sum(-1))
    overlap_ij = (
        (D < 2.0 * radius)
        * (np.pi / 12.0)
        * (4.0 * radius + D)
        * (2.0 * radius - D) ** 2
    )
    V_2 = np.tril(overlap_ij, k=-1).sum()

    # Inclusion-Exclusion Principle
    volume = V_1 - V_2
    return volume


def plane_split_protein(X=None, C=None, protein=None, mask_percent=0.5):
    if protein is None:
        assert X is not None and C is not None
    else:
        X, C, _ = protein.to_XCS()
        X = X[:, :, :4, :]

    X = backbone.center_X(X, C)
    points = X[C > 0].reshape(-1, 3)
    pca = PCA(n_components=1)
    normal = torch.from_numpy(
        pca.fit_transform(points.detach().cpu().numpy().transpose(1, 0))
    ).to(X.device)
    c_alphas = X[:, :, 1, :]

    c = 0
    tries = 0

    def percent_masked(c):
        C_mask = ((c_alphas @ normal) > c).squeeze(-1) & (C > 0)
        return (~C_mask).float().sum().item() / (C > 0).sum().item()

    # In the first stage we find the minimum C such that all of the residues
    # lie on one side of the plane (c_alphas @ normal = c)
    while (percent_masked(c) < 1.0) and (tries < 300000):
        tries += 1
        c += 100

    # Now we drag the plane back until percent_masked - masked_percent is small.
    size = X.size(1)
    threshold = 0.1 if size < 100 else 0.05 if size < 500 else 0.01
    tries = 0
    while (np.abs(percent_masked(c) - mask_percent) > threshold) and (tries < 300000):
        c -= 100
        tries += 1

    if tries >= 300000:
        print(
            "Tried and failed to split protein by plane to grab"
            f" {mask_percent} residues."
        )
        c = 0
        C_mask = ((c_alphas @ normal) > c).squeeze(-1) & (C > 0)
        print(
            f"Returning {100 * percent_masked(0.0):.2f} percent residues masked"
            " instead."
        )

    else:
        C_mask = ((c_alphas @ normal) > c).squeeze(-1) & (C > 0)
        print(
            f"Split protein by plane, masking {100 * percent_masked(c):.2f} percent of"
            " residues."
        )

    return C_mask


def export_potts_evzoom(
    outfile: str,
    chroma: nn.Module,
    protein: Protein,
    t: float = 0.5,
    alphabet_reorder="DEKRHQNSTPGAVILMCFWY",
    element_cutoff_percentile: float = 0.95,
    norm_cutoff_fraction: float = 0.98,
    logo_cutoff_fraction: float = 0.9,
):
    """Build EVzoom visualization given Chroma and a protein.

    EVzoom files can be dragged and dropped onto a viewer at EVzoom.org

    Args:
        outfile (str): File destination for outputting EVzoom json data. Should
            end in `.json`.
        chroma (nn.Module): Chroma model instance.
        protein (Protein): Protein
        t (float): Diffusion time for evaluating ChromaDesign Potts models.
        alphabet_reorder (str): Amino acid ordering for EVzoom output.
        element_cutoff_percentile (float): Percentile on (0,1) at which to clip
            `J_ij` element-wise values for visualization.
        norm_cutoff_fraction (float): Only visualize the top `i,j` pairs whose
            norms account for this fraction of the sum of all norm values.
        logo_cutoff_fraction (float): Only visualize the amino acids whose
            cumulative marginal probabilities account for this total fraction
            per position.
    """
    if not outfile.endswith(".json"):
        raise Exception("Destination should be a json file")

    alphabet = AA20
    X, C, S = protein.to_XCS()
    h, J, edge_idx = chroma.design_network.predict_potts(X, C, t=t)
    log_probs = chroma.design_network.predict_marginals(X, C, t=t)[0]

    sequence = "".join([alphabet[ax] for ax in S.cpu().data.numpy().flatten().tolist()])
    permute_ix = np.array([alphabet.index(c) for c in alphabet_reorder])

    evz_data = {
        "map": {"letters": sequence, "indices": list(range(len(sequence)))},
        "logo": [],
        "couplings": [],
    }

    h = -h[0, :, :].data.cpu().numpy()
    J = -J[0, :, :, :, :].data.cpu().numpy()
    edge_idx = edge_idx[0, :, :].data.cpu().numpy()
    log_probs = log_probs[0, :, :].data.cpu().numpy()
    N = J.shape[0]
    k = J.shape[1]

    # Compute entropy
    P = np.exp(log_probs)
    P = P / np.sum(P, 1, keepdims=True)
    H = -np.sum(P * np.log2(P), 1)

    # Determine top residues per position
    P_sort = np.sort(P, axis=-1)[:, ::-1]
    CDF = P_sort.cumsum(axis=-1)
    ix_valid = CDF <= logo_cutoff_fraction
    ix = ix_valid.sum(axis=1)
    p_cutoff_logo = P_sort[np.arange(N), ix]
    P_mask = P >= p_cutoff_logo[:, np.newaxis]

    # Export logo
    for i in range(N):
        bits = (np.log2(20.0) - H[i]) * P[i]
        aa_idx = np.argsort(bits)[::-1]
        data_i = []
        for a_i in aa_idx:
            if P_mask[i, a_i]:
                data_i.append({"code": alphabet[a_i], "bits": str(bits[a_i])})
        evz_data["logo"].append(data_i)

    # Compute cutoff automatically
    norms = (J ** 2).sum(axis=(2, 3))
    norms_descend = np.sort(norms.flatten())[::-1]
    norms_cumsum = np.cumsum(norms_descend)
    valid_norms = norms_cumsum / norms_cumsum[-1] < norm_cutoff_fraction
    cutoff = norms_descend[valid_norms].min()

    # Export couplings
    for i in range(N):
        for j in range(k):
            j_idx = edge_idx[i, j]
            score = np.sqrt(np.mean(J[i, j, :, :] ** 2))
            if score > cutoff:
                ii = [ix for ix in permute_ix if P_mask[i, ix]]
                jj = [ix for ix in permute_ix if P_mask[j_idx, ix]]
                J_ij = J[i, j, :, :]
                J_ij = J_ij[ii][:, jj]
                J_ij = np.clip(
                    J_ij,
                    np.percentile(J_ij, 100 * (1 - element_cutoff_percentile)),
                    np.percentile(J_ij, element_cutoff_percentile * 100),
                )
                J_ij = np.round(J_ij, 3)
                data_ij = {
                    "i": i + 1,
                    "j": str(j_idx + 1),
                    "score": str(score),
                    "iC": [alphabet[ix] for ix in ii],
                    "jC": [alphabet[ix] for ix in jj],
                    "matrix": J_ij.tolist(),
                }
                evz_data["couplings"].append(data_ij)

    with open(outfile, "w") as file:
        file.write(json.dumps(evz_data))

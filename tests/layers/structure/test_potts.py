from collections import Counter
from itertools import product

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from chroma.layers.structure.potts import (
    GraphPotts,
    compute_potts_energy,
    fold_symmetry,
    sample_potts,
)


def test_graphpotts():
    # Testing symmetry
    # Create non-symmetric Potts model and symmetrize using serial or not
    potts = GraphPotts(128, 128, 20, symmetric_J=False)

    node_h = torch.rand(1, 3, 128)
    edge_h = torch.rand(1, 3, 2, 128)
    edge_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
    mask_i = torch.ones(1, 3)
    mask_ij = torch.ones(1, 3, 2)

    h, J = potts(node_h, edge_h, edge_idx, mask_i, mask_ij)

    assert (
        potts._symmetrize_J(J, edge_idx, mask_ij)
        != potts._symmetrize_J_serial(J, edge_idx, mask_ij)
    ).sum().detach().numpy() == 0

    mask_ij = torch.tensor([[[1, 1], [1, 0], [1, 0]]])
    h, J = potts(node_h, edge_h, edge_idx, mask_i, mask_ij)

    assert (
        potts._symmetrize_J(J, edge_idx, mask_ij)
        != potts._symmetrize_J_serial(J, edge_idx, mask_ij)
    ).sum().detach().numpy() == 0


def test_symmetry_folding():
    N, Q = 12, 3
    symmetry_order = 3
    N_au = N // symmetry_order

    # Testing symmetry
    mask_i = torch.ones(1, N)
    mask_ij = (1.0 - torch.eye(N))[None, ...]
    h = torch.randn([1, N, Q])
    J = torch.randn([1, N, N, Q, Q])
    J = J + J.permute([0, 2, 1, 4, 3])
    # J = torch.eye(Q)[None,None,None,...].expand([1, N, N, Q, Q])
    J = J * mask_ij[..., None, None]
    edge_idx = torch.arange(N).long()[None, None, :].expand([1, N, N])

    h_fold, J_fold, edge_idx_fold, mask_i_fold, mask_ij_fold = fold_symmetry(
        symmetry_order, h, J, edge_idx, mask_i, mask_ij, normalize=False
    )
    # Validate dimensions
    assert tuple(h_fold.shape) == (1, N_au, Q)
    assert tuple(J_fold.shape) == (1, N_au, N_au, Q, Q)
    assert tuple(edge_idx_fold.shape) == (1, N_au, N_au)
    assert tuple(mask_i_fold.shape) == (1, N_au)
    assert tuple(mask_ij_fold.shape) == (1, N_au, N_au)

    # Does the folded Potts model return same energies as full?
    S_test_fold = torch.randint(high=Q, size=[1, N_au])
    S_test = S_test_fold[:, None, :].expand([1, symmetry_order, N_au]).reshape([1, N])

    U, U_i = compute_potts_energy(S_test, h, J, edge_idx)
    U_fold, U_i_fold = compute_potts_energy(S_test_fold, h_fold, J_fold, edge_idx_fold)

    assert torch.allclose(U, U_fold)


@pytest.mark.parametrize("proposal", ["dlmc", "chromatic"])
def test_potts_mcmc(proposal, debug=False):
    """MCMC test for Chromatic Gibbs sampling."""
    # Build a test, fully connected Potts model
    if debug:
        # Heavy duty sampling with large state space
        N = 5
        q = 4
        num_sweeps = 1000
        num_chains = 1000
        rtol = 0.05
    else:
        # Quick and dirty small state space
        N = 3
        q = 3
        num_sweeps = 200
        num_chains = 1000
        rtol = 0.1

    beta = 0.1
    warmup_fraction = 0.1

    torch.manual_seed(1)
    mask_i = torch.ones([1, N]).float()
    mask_ij = (1 - torch.eye(N))[None, ...].float()
    edge_idx = torch.arange(N)[None, None, :].expand([1, N, N])

    h = beta * torch.randn([1, N, q])
    J = beta * torch.randn([1, N, N, q, q])
    J = mask_ij[..., None, None] * (J + J.permute([0, 2, 1, 4, 3])) / np.sqrt(2)

    # Enumerate all of sequence space
    alphabet = "ABCDEFGHIJK"[:q]
    sequences = ["".join(x) for x in product(alphabet, repeat=N)]
    S_exact = torch.Tensor(
        [[alphabet.index(s) for s in seq] for seq in sequences]
    ).long()
    print(f"Enumerated {len(sequences)} sequences")

    if torch.cuda.is_available():
        device = "cuda"
        h = h.to(device)
        J = J.to(device)
        edge_idx = edge_idx.to(device)
        mask_i = mask_i.to(device)
        mask_ij = mask_ij.to(device)
        S_exact = S_exact.to(device)
    if torch.backends.mps.is_available():
        device = "mps"
        h = h.to(device)
        J = J.to(device)
        edge_idx = edge_idx.to(device)
        mask_i = mask_i.to(device)
        mask_ij = mask_ij.to(device)
        S_exact = S_exact.to(device)

    # Compute exact distribution over sequence space
    B = S_exact.shape[0]
    h_expand = h.expand([B, -1, -1])
    J_expand = J.expand([B, -1, -1, -1, -1])
    edge_idx_expand = edge_idx.expand([B, -1, -1])
    mask_i_expand = mask_i.expand([B, -1])
    mask_ij_expand = mask_ij.expand([B, -1, -1])
    U, _ = compute_potts_energy(S_exact, h_expand, J_expand, edge_idx_expand)
    p_exact = F.softmax(-U, -1).tolist()

    # Estimate distribution from sampled sequences
    h_expand = h.expand([num_chains, -1, -1])
    J_expand = J.expand([num_chains, -1, -1, -1, -1])
    edge_idx_expand = edge_idx.expand([num_chains, -1, -1])
    mask_i_expand = mask_i.expand([num_chains, -1])
    mask_ij_expand = mask_ij.expand([num_chains, -1, -1])

    S, U, S_trajectory, U_trajectory = sample_potts(
        h_expand,
        J_expand,
        edge_idx_expand,
        mask_i_expand,
        mask_ij_expand,
        num_sweeps=num_sweeps,
        proposal=proposal,
        rejection_step=True,
        verbose=True,
        return_trajectory=True,
    )
    if warmup_fraction is not None:
        S_trajectory = S_trajectory[int(warmup_fraction * len(S_trajectory)) :]

    S_samples = torch.cat(S_trajectory, 0)
    U_trajectory = torch.stack(U_trajectory, 1).cpu().data.numpy()
    S_samples = S_samples.cpu().data.numpy()
    sample_counts = Counter(["".join([alphabet[c] for c in s]) for s in S_samples])
    p_sample = [sample_counts[seq] / S_samples.shape[0] for seq in sequences]

    if debug:
        from matplotlib import pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.plot(p_exact, p_sample, "k.")
        plt.grid()
        plt.axis("square")
        plt.xlabel("Probability, exact enumeration")
        plt.ylabel("Sampling frequencey (MCMC)")
        plt.title(f"Random Potts model over {q}^{N} sequences")
        plt.subplot(1, 2, 2)
        plt.plot(U_trajectory[0, :])
        plt.xlabel("Iterations")
        plt.ylabel("Energy")
        plt.tight_layout()
        plt.show()

    # The frequencies of states visited via MCMC should reproduce their
    # exact probabilities (via enumeration) within rtol percent error
    assert np.allclose(p_sample, p_exact, rtol=rtol)


def debug_potts_2D():
    """Debug test for Potts model"""
    N = 100
    q = 4

    num_sites = N * N
    mask_i = torch.ones([1, N]).float()
    ix = torch.arange(num_sites).long()

    # Build 2D lattice topology
    edge_idx = torch.stack([ix + 1, ix - 1, ix + N, ix - N], -1)
    mask_ij = torch.ones_like(edge_idx).float()[None, :, :]
    edge_idx = torch.remainder(edge_idx, num_sites)[None, :, :].long()

    # Ferromagnetic parameters
    h = torch.zeros([1, num_sites, q])
    h[:, :, 0] = h[:, :, 0]
    mask_J = mask_ij[:, :, :, None, None] * torch.eye(q)[None, None, None, :, :]

    if torch.cuda.is_available():
        device = "cuda"
        h = h.to(device)
        edge_idx = edge_idx.to(device)
        mask_J = mask_J.to(device)
        mask_ij = mask_ij.to(device)

    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    temp_range = (1.2, 0.8)
    plt.figure(figsize=(5, 5), dpi=600)
    _, _, S_trajectory, U_trajectory = sample_potts(
        h,
        -mask_J,
        edge_idx,
        mask_i,
        mask_ij,
        num_sweeps=10000,
        verbose=True,
        return_trajectory=True,
        S=None,
        annealing_fraction=1.0,
        temperature_init=1.2,
        temperature=0.8,
    )

    # Define a function to update the plot for each frame
    num_frames = len(S_trajectory)
    temps = np.linspace(temp_range[0], temp_range[1], len(S_trajectory))
    betas = 1.0 / temps

    def update(frame):
        plt.clf()  # Clear the previous frame
        plt.pcolor(S_trajectory[frame].cpu().data.numpy().reshape([N, N]), cmap="tab10")
        plt.clim([0, 10])
        plt.axis("square")
        plt.axis("off")
        plt.title(f"Beta = {betas[frame]:0.2f}")
        print(frame)

    # Create a figure and set the number of frames
    fig = plt.figure(figsize=(4, 4), dpi=300)
    animation = FuncAnimation(fig, update, frames=num_frames, interval=1000 / 60)
    animation.save("potts.mp4", writer="ffmpeg")
    return

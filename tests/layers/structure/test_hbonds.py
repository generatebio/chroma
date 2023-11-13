from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import chroma
from chroma.data import Protein
from chroma.layers import graph
from chroma.layers.structure import hbonds, protein_graph


@pytest.fixture(scope="session")
def XCS():
    repo = Path(chroma.__file__).parent.parent
    pdb_id = "6wgl"
    test_cif = str(Path(repo, "tests", "resources", "6wgl.cif"))
    X, C, S = Protein(test_cif).to_XCS()
    return X, C, S, pdb_id


def test_backbone_hbonds(XCS, debug_plot=False):
    X, C, S, pdb_id = XCS

    bb_hbonds = hbonds.BackboneHBonds()

    # Build Graph
    graph_builder = protein_graph.ProteinGraph()
    edge_idx, mask_ij = graph_builder(X, C)
    hb, mask_hb, H_i = bb_hbonds(X, C, edge_idx, mask_ij)
    hb_dense = graph.scatter_edges(hb[..., None], edge_idx)[..., 0]

    if debug_plot:
        if False:
            H = hb_dense[0, :, :].data.numpy()
            from matplotlib import pyplot as plt

            plt.matshow(H)
            plt.show()

        # Build
        rgb = (0.3, 0.7, 0.1)
        with open(f"viz_hbonds_{pdb_id}.pml", "w") as f:
            f.write(
                "delete all\n"
                f"fetch {pdb_id}\n"
                f"hide everything, {pdb_id}\n"
                "show sticks, bb.\n"
                "color white, all\n"
                "color atomic, (not elem C)\n"
                "h_add bb.\n"
                "distance hbonds_pymol, don. and bb., acc. and bb., 3.6, mode=2\n"
                "hide labels\n"
            )
            cgo_list = [protein_graph._cgo_color(rgb)]
            for i in range(edge_idx.size(1)):
                for j_idx in range(edge_idx.size(2)):
                    if hb[0, i, j_idx] > 0:
                        j = edge_idx[0, i, j_idx]
                        cgo_list.append(
                            protein_graph._cgo_cylinder(
                                H_i[0, i, :], X[0, j, 3, :], radius=0.08, rgb=rgb
                            )
                        )
                        cgo_list.append(
                            protein_graph._cgo_sphere(H_i[0, i, :], radius=0.3)
                        )
            cgo_str = " + ".join(cgo_list)
            f.write(f'cmd.load_cgo({cgo_str}, "hbonds_pytorch", 1)\n')

    # These hydrogen bonds were manually spot checked for 6wgl
    # in Pymol using the above script. We don't count i-i+2 and
    # there appear to be subtle orientation dependent, but
    # SS-dependent calls agree well
    assert hb_dense.sum().item() == 303


def test_loss_hbb(XCS, debug=False):
    X, C, S, pdb_id = XCS
    loss_hbb = hbonds.LossBackboneHBonds()

    torch.manual_seed(1.0)
    X_noise = X + torch.randn_like(X)
    recovery_local, recovery_nonlocal, error_co = loss_hbb(X_noise, X, C)
    assert recovery_local.mean().item() < 1.0
    assert recovery_nonlocal.mean().item() < 1.0
    assert error_co > 0.0

    recovery_local, recovery_nonlocal, error_co = loss_hbb(X, X, C)
    assert recovery_local.mean().item() == pytest.approx(1.0, 1e-2)
    assert recovery_nonlocal.mean().item() == pytest.approx(1.0, 1e-2)
    assert error_co.mean().item() == pytest.approx(0.0, 1e-2)

    if debug:
        # This
        from chroma.layers.structure import diffusion

        noise = diffusion.DiffusionChainCov(complex_scaling=True)

        T = np.linspace(0, 1, 100)
        R_local = []
        R_nonlocal = []
        for t in T:
            X_noise = noise(X, C, t=t)
            recovery_local, recovery_nonlocal, error_co = loss_hbb(X_noise, X, C)
            R_local.append(recovery_local.mean().item())
            R_nonlocal.append(recovery_nonlocal.mean().item())
        A = noise.noise_schedule.alpha(T.tolist()).data.numpy().flatten()

        from matplotlib import pyplot as plt

        plt.subplot(1, 2, 1)
        plt.plot(T, R_local, label="Local H-Bonds")
        plt.plot(T, R_nonlocal, label="Nonlocal H-Bonds")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("t")
        plt.ylabel("Recovery")
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(A, R_local, label="Local H-Bonds")
        plt.plot(A, R_nonlocal, label="Nonlocal H-Bonds")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("alpha")
        plt.ylabel("Recovery")
        plt.legend()
        plt.grid()
        plt.show()
    return

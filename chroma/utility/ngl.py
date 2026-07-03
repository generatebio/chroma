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
Utilities for rendering protein structures in Jupyter notebooks.

This provides convenience functions for rendering our common structure
datatypes, such as `mst.System` and `XCS` tensors, with nglview.
"""

import tempfile
import uuid
from functools import lru_cache


def _load_nglview():
    try:
        import nglview as nv
    except Exception as exc:
        raise ImportError(
            "Protein display requires a working `nglview` installation."
        ) from exc
    return nv


def has_nglview() -> bool:
    try:
        _load_nglview()
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _system_trajectory_type():
    nv = _load_nglview()

    class _SystemTrajectory(nv.base_adaptor.Trajectory, nv.base_adaptor.Structure):
        """NGLView adaptor for multi-model `Protein` objects."""

        def __init__(self, protein):
            self.protein = protein
            self.ext = "pdb"
            self.params = {}
            self.id = str(uuid.uuid4())

        def get_coordinates(self, index):
            self.protein.sys.swap_model(index)
            X, _, _ = self.protein.sys.to_XCS()
            self.protein.sys.swap_model(index)
            return X.view(-1, 3).numpy()

        @property
        def n_frames(self):
            return self.protein.sys.num_models()

        def get_structure_string(self):
            return self.protein.sys.to_PDB_string()

    return _SystemTrajectory


def SystemTrajectory(protein):
    return _system_trajectory_type()(protein)


def view_gsystem(system, **kwargs):
    """Return an NGL Viewer Widget for an generate System.

    Args:
        system (System): Structure to view.

    Returns:
        view: NGL Viewer widget instance that. In a Jupyter notebook
            returning this to the notebook will trigger display of a
            widget.
    """
    nv = _load_nglview()
    temp = tempfile.NamedTemporaryFile(suffix=".pdb")
    filename = temp.name
    system.to_PDB(filename)
    view = nv.show_file(filename)
    view.clear_representations()
    view.add_representation("cartoon")
    view.add_representation("licorice", selection="(sidechain or .CA) and not hydrogen")
    view.add_representation("contact")
    view.center()
    return view


def view_protein(protein, representations=None):
    nv = _load_nglview()
    representations = representations or []

    if protein.sys.num_models() == 1:
        viewer = view_gsystem(protein.sys)
    else:
        viewer = nv.NGLWidget(SystemTrajectory(protein))

    for representation in representations:
        viewer.add_representation(representation)
    return viewer

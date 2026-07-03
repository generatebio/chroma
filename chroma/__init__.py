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

from importlib import import_module

__version__ = "1.0.0"
__all__ = ["__version__", "Protein", "conditioners", "Chroma", "api"]


def __getattr__(name):
    if name == "Protein":
        return import_module("chroma.data.protein").Protein
    if name == "conditioners":
        return import_module("chroma.layers.structure.conditioners")
    if name == "Chroma":
        return import_module("chroma.models.chroma").Chroma
    if name == "api":
        return import_module("chroma.utility.api")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

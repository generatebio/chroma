from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SystemAssemblyInfo:
    """A class for representing the assembly information for System objects."""

    assemblies: dict
    operations: dict

    def __init__(self, assemblies: Optional[dict] = None, operations: Optional[dict] = None):
        self.assemblies = {} if assemblies is None else assemblies
        self.operations = {} if operations is None else operations

    @staticmethod
    def make_operation(type: str, name: str, matrix: list, vector: list):
        op = {
            "type": type,
            "name": name,
            "matrix": np.zeros([3, 3]),
            "vector": np.zeros(3),
        }
        assert len(matrix) == 9, "expected 9 elements in rotation matrix"
        assert len(vector) == 3, "expected 3 elements in translation vector"
        for i in range(3):
            op["vector"][i] = float(vector[i])
            for j in range(3):
                op["matrix"][i][j] = float(matrix[i * 3 + j])
        return op

    def delete_chain(self, cid: str):
        for _, assembly in self.assemblies.items():
            for ins in assembly["instructions"]:
                ins["chains"] = [_id for _id in ins["chains"] if _id != cid]

    def rename_chain(self, old_cid: str, new_cid: str):
        for _, assembly in self.assemblies.items():
            for ins in assembly["instructions"]:
                ins["chains"] = [
                    new_cid if cid == old_cid else cid for cid in ins["chains"]
                ]


class StringList:
    """Memory-efficient list of strings with constant-time access."""

    def __init__(self, init_list: Optional[List[str]] = None):
        init_list = [] if init_list is None else init_list
        self.string = ""
        self.rng = ArrayList(2, dtype=int)
        for item in init_list:
            self.append(item)

    def __getitem__(self, i: int):
        beg, length = self.rng[i]
        return self.string[beg : beg + length]

    def __setitem__(self, i: int, new_string: str):
        beg, length = self.rng[i]
        self.string = self.string[:beg] + new_string + self.string[beg + length :]
        if len(new_string) != length:
            self.rng[i, 1] = len(new_string)
            self.rng[i + 1 :, 0] = self.rng[i + 1 :, 0] + len(new_string) - length

    def __str__(self):
        return self.string

    def __len__(self):
        return len(self.rng)

    def copy(self):
        new_list = StringList()
        new_list.string = self.string
        new_list.rng = self.rng.copy()
        return new_list

    def append(self, new_string: str):
        self.rng.append([len(self.string), len(new_string)])
        self.string = self.string + new_string

    def insert(self, i: int, new_string: str):
        if i < len(self):
            ix, _ = self.rng[i]
        elif i == len(self):
            ix = 0 if len(self) == 0 else self.rng[i - 1].sum()
        else:
            raise Exception(
                f"cannot insert in position {i} for stringList of length {len(self)}"
            )
        self.string = self.string[0:ix] + new_string + self.string[ix:]
        self.rng.insert(i, [ix, len(new_string)])
        if len(new_string) > 0:
            self.rng[i + 1 :, 0] = self.rng[i + 1 :, 0] + len(new_string)

    def pop(self, i: int):
        beg, length = self.rng[i]
        val = self.string[beg : beg + length]
        self.string = self.string[0:beg] + self.string[beg + length :]
        self.rng[i + 1 :, 0] = self.rng[i + 1 :, 0] - len(val)
        self.rng.pop(i)
        return val

    def delete_range(self, rng: range):
        rng = sorted(rng)
        [i, j] = [rng[0], rng[-1]]
        beg, _ = self.rng[i]
        end = self.rng[j].sum()
        self.string = self.string[0:beg] + self.string[end:]
        self.rng[j + 1 :, 0] = self.rng[j + 1 :, 0] - (end - beg + 1)
        self.rng.delete_range(rng)


class NameList:
    """A list of repeated names backed by an index table."""

    def __init__(self, init_list: Optional[List[str]] = None):
        init_list = [] if init_list is None else init_list
        self._reindex(init_list)

    def _reindex(self, init_list: List[str]):
        self.unique_names = []
        self.name_indicies = dict()
        self.index_use = dict()
        self.indices = ArrayList(1, dtype=int)
        for name in init_list:
            self.append(name)

    def copy(self):
        new_list = NameList()
        new_list.unique_names = self.unique_names.copy()
        new_list.name_indicies = self.name_indicies.copy()
        new_list.index_use = self.index_use.copy()
        new_list.indices = self.indices.copy()
        return new_list

    def _check_index(self):
        if (len(self.unique_names) > 2 * len(self.index_use)) and (
            len(self.unique_names) - len(self.index_use) > 10
        ):
            self._reindex([self[i] for i in range(len(self))])

    def __getitem__(self, i: int):
        try:
            idx = self.indices[i].item()
        except IndexError as e:
            raise IndexError(f"index {i} out of range for nameList\n" + str(e))
        return self.unique_names[idx]

    def __setitem__(self, i: int, new_name: str):
        try:
            idx = self.indices[i]
        except IndexError as e:
            raise IndexError(f"index {i} out of range for nameList\n" + str(e))
        self.index_use[idx] = self.index_use[idx] - 1
        if self.index_use[idx] == 0:
            del self.index_use[idx]
        idx = self._get_name_index(new_name)
        self.indices[i] = idx
        self._update_use(idx, 1)
        self._check_index()

    def __str__(self):
        return str([self[i] for i in range(len(self))])

    def __len__(self):
        return len(self.indices)

    def _update_use(self, idx, delta):
        self.index_use[idx] = self.index_use.get(idx, 0) + delta
        if self.index_use[idx] <= 0:
            del self.index_use[idx]

    def _get_name_index(self, name: str):
        if name not in self.name_indicies:
            idx = len(self.name_indicies)
            self.name_indicies[name] = idx
            self.unique_names.append(name)
        else:
            idx = self.name_indicies[name]
        return idx

    def append(self, name: str):
        idx = self._get_name_index(name)
        self.indices.append(idx)
        self.index_use[idx] = self.index_use.get(idx, 0) + 1

    def insert(self, i: int, new_string: str):
        idx = self._get_name_index(new_string)
        self.indices.insert(i, idx)
        self.index_use[idx] = self.index_use.get(idx, 0) + 1

    def pop(self, i: int):
        idx = self.indices.pop(i).item()
        val = self.unique_names[idx]
        self._update_use(idx, -1)
        self._check_index()
        return val

    def delete_range(self, rng: range):
        for i in reversed(sorted(rng)):
            self.pop(i)


class ArrayList:
    def __init__(self, ndims: int, dtype: type, length: int = 0, val=0):
        if ndims == 1:
            self._array = np.ndarray(shape=(max(length, 2)), dtype=dtype)
        else:
            self._array = np.ndarray(shape=(max(length, 2), ndims), dtype=dtype)
        self.ndims = ndims
        self._array[:] = val
        self.length = length
        self.array = self._array[: self.length]

    def convert_negative_slice(self, slice_obj):
        start = slice_obj.start if slice_obj.start is not None else 0
        stop = slice_obj.stop if slice_obj.stop is not None else self.length
        if start < 0:
            start = self.length + start
        if stop < 0:
            stop = self.length + stop
        return slice(start, stop, slice_obj.step)

    def copy(self):
        new_list = ArrayList(ndims=self.ndims, dtype=self.array.dtype, length=len(self))
        new_list[:] = self[:]
        return new_list

    def __len__(self):
        return self.length

    def capacity(self):
        return self._array.shape[0]

    def __getitem__(self, i: int):
        return self.array[i]

    def __setitem__(self, i: int, row: list):
        self.array[i] = row

    def resize(self, delta):
        new_length = self.length + delta
        cap = self._array.shape[0]
        if (new_length > cap) or (new_length < cap / 3):
            self._resize(2 * new_length)
        self.length = new_length
        self.array = self._array[: self.length]

    def _resize(self, new_size):
        if self.ndims == 1:
            self._array.resize((new_size), refcheck=False)
        else:
            self._array.resize((new_size, self.ndims), refcheck=False)

    def items(self):
        for i in range(self.length):
            yield self.array[i, :]

    def append(self, row: list):
        self.resize(1)
        self.array[-1] = row

    def insert(self, i: int, row: list):
        self.resize(1)
        self.array[i + 1 :] = self.array[i:-1]
        self.array[i] = row

    def pop(self, i: int):
        row = self.array[i].copy()
        self.array[i:-1] = self.array[i + 1 :]
        self.resize(-1)
        return row

    def delete_range(self, rng: range):
        i, j = min(rng), max(rng)
        cut_length = j - i + 1
        new_length = len(self) - cut_length
        self.array[i:new_length] = self.array[j + 1 :]
        self.resize(-cut_length)

    def __str__(self):
        return str([self[i] for i in range(len(self))])


@dataclass
class HierarchicList:
    """Utility class that represents a hierarchy of lists."""

    _properties: dict
    _parent_list: "HierarchicList"
    _child_list: "HierarchicList"
    _num_children: ArrayList
    _child_offset: ArrayList

    def __init__(
        self,
        properties: dict,
        parent_list: "HierarchicList" = None,
        num_children: ArrayList = ArrayList(1, dtype=int),
    ):
        self._properties = {key: value.copy() for key, value in properties.items()}
        self._parent_list = parent_list
        if self._parent_list is not None:
            self._parent_list._child_list = self
        self._child_list = None
        self._num_children = num_children.copy() if num_children is not None else None
        self._child_offset = None

    def copy(self):
        new_list = HierarchicList(
            self._properties, self._parent_list, self._num_children
        )
        new_list._child_list = self._child_list
        new_list._child_offset = (
            None if self._child_offset is None else self._child_offset.copy()
        )
        return new_list

    def set_parent(self, parent_list: "HierarchicList"):
        self._parent_list = parent_list

    def child_index(self, i: int, at: int):
        if self._child_offset is not None:
            return self._child_offset[i] + at
        return self._num_children[0:i].sum() + at

    def reindex(self):
        if self._num_children is not None:
            self._child_offset = ArrayList(
                1, dtype=int, length=len(self._num_children), val=0
            )
            for i in range(1, len(self)):
                self._child_offset[i] = (
                    self._child_offset[i - 1] + self._num_children[i - 1]
                )

    def append_child(self, properties):
        self._num_children[len(self._num_children) - 1] += 1
        self._child_list.append(properties)

    def insert_child(self, i: int, at: int, properties):
        idx = self.child_index(i, at)
        self._num_children[i] += 1
        self._child_offset = None
        self._child_list.insert(idx, properties)
        return idx

    def delete_child(self, i: int, at: int):
        idx = self.child_index(i, at)
        self._num_children[i] -= 1
        self._child_offset = None
        self._child_list.delete(idx)

    def append(self, properties):
        if set(properties.keys()) != set(self._properties.keys()):
            raise Exception(f"unexpected set of attributes '{list(properties.keys())}")
        for key, value in properties.items():
            self._properties[key].append(value)
        if self._child_offset is not None:
            self._child_offset.append(
                self._child_offset[-1:].sum() + self._num_children[-1:].sum()
            )
        if self._num_children is not None:
            self._num_children.append(0)

    def insert(self, i: int, properties):
        if set(properties.keys()) != set(self._properties.keys()):
            raise Exception(f"unexpected set of attributes '{list(properties.keys())}")
        for key, value in properties.items():
            self._properties[key].insert(i, value)
        if self._child_offset is not None:
            off = (
                self._child_offset[-1:].sum() + self._num_children[-1:].sum()
                if i >= len(self._child_offset)
                else self._child_offset[i]
            )
            self._child_offset.insert(i, off)
        if self._num_children is not None:
            self._num_children.insert(i, 0)

    def delete(self, i: int):
        for key in self._properties:
            self._properties[key].pop(i)
        if self._num_children is not None and self._num_children[i] != 0:
            for at in range(self._num_children[i] - 1, -1, -1):
                self.delete_child(i, at)
            self._num_children.pop(i)
        self._child_offset = None

    def delete_range(self, rng: range):
        for key in self._properties:
            self._properties[key].delete_range(rng)
        for i in reversed(sorted(rng)):
            if self._num_children is not None and self._num_children[i] != 0:
                idx = self.child_index(i, 0)
                self._child_list.delete_range(self, range(idx, idx + self._num_children[i]))
                self._num_children[i] = 0
        self._child_offset = None

    def __len__(self):
        for key in self._properties:
            return len(self._properties[key])
        return None

    def __getitem__(self, i: str):
        return self._properties[i]

    def num_children(self, i: int):
        return self._num_children[i]

    def has_children(self, i: int):
        return self._num_children is not None and self._num_children[i]

    def __str__(self):
        string = "Properties:\n"
        for key in self._properties:
            string += f"{key}: {str(self._properties[key])}\n"
        string += f"num_children: {str(self._num_children)}\n"
        string += f"child_offset: {str(self._child_offset)}\n"
        string += "----\n"
        string += str(self._child_list)
        return string

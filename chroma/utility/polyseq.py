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

"""Standard residue names for polymers of different types (e.g., L- or D-amino acid proteins,
mixed-chirality proteins, DNA/RNA, etc.)
"""

from enum import Enum


class polymerType(Enum):
    LPROT = 0
    DPROT = 1
    LDPROT = 2
    DNA = 3
    RNA = 4


def polymer_type_name(ptype: polymerType):
    if ptype == polymerType.LPROT:
        return "polypeptide(L)"
    elif ptype == polymerType.DPROT:
        return "polypeptide(D)"
    elif ptype == polymerType.LDPROT:
        return "polypeptide(L,D)"
    elif ptype == polymerType.DNA:
        return "polydeoxyribonucleotide"
    elif ptype == polymerType.RNA:
        return "polyribonucleotide"
    else:
        raise Exception(f"unknown polymer type {ptype}")


_res3 = [[] for _ in range(len(polymerType))]

_res1 = [[] for _ in range(len(polymerType))]

_res_to_idx = [dict() for _ in range(len(polymerType))]

_unk_idx = [set() for _ in range(len(polymerType))]

_gap_idx = [set() for _ in range(len(polymerType))]

_stp_idx = [set() for _ in range(len(polymerType))]


def _add_residue(ptype: polymerType, res3, res1):
    if isinstance(ptype, list):
        for pt, r3, r1 in zip(ptype, res3, res1):
            _add_residue(pt, r3, r1)
    else:
        _res_to_idx[ptype.value][res3] = len(_res3[ptype.value])
        # single-letter code is ambiguous, so take the first residue when going from single-letter code to index
        if res1 not in _res_to_idx[ptype.value]:
            _res_to_idx[ptype.value][res1] = _res_to_idx[ptype.value][res3]
        _res3[ptype.value].append(res3)
        _res1[ptype.value].append(res1)
        if res3 == "---":
            _gap_idx[ptype.value].add(_res_to_idx[ptype.value][res3])
        elif res3 == "UNK":
            _unk_idx[ptype.value].add(_res_to_idx[ptype.value][res3])
        elif res3 == "STP":
            _stp_idx[ptype.value].add(_res_to_idx[ptype.value][res3])


def num_tokens(ptype=polymerType.LPROT):
    return len(_res3[ptype.value])


def num_known_molecular_tokens(ptype=polymerType.LPROT):
    return sum(
        [
            not is_punctuation_index(idx) and not is_unknown(idx)
            for idx in range(len(_res3[ptype.value]))
        ]
    )


def res_to_index(res: str, ptype=polymerType.LPROT):
    return _res_to_idx[ptype.value].get(res, next(iter(_unk_idx[ptype.value])))


def index_to_single(idx: int, ptype=polymerType.LPROT):
    return _res1[ptype.value][idx]


def index_to_triple(idx: int, ptype=polymerType.LPROT):
    return _res3[ptype.value][idx]


def to_single(res: str, ptype=polymerType.LPROT):
    return index_to_single(res_to_index(res, ptype))


def to_triple(res: str, ptype=polymerType.LPROT):
    return index_to_triple(res_to_index(res, ptype))


def is_gap_index(idx: int, ptype=polymerType.LPROT):
    return idx in _gap_idx[ptype.value]


def is_stop_index(idx: int, ptype=polymerType.LPROT):
    return idx in _stp_idx[ptype.value]


def is_unknown(res: str, ptype=polymerType.LPROT):
    return is_unknown_index(res_to_index(res, ptype), ptype)


def is_unknown_index(idx: int, ptype=polymerType.LPROT):
    return idx in _unk_idx[ptype.value]


def is_polymer_residue(res: str, ptype: polymerType):
    if ptype is None:
        # determine if this is a polymer residue for any known polymer
        for ptype in polymerType:
            if res in _res_to_idx[ptype.value]:
                return True
        return False
    return res in _res_to_idx[ptype.value]


def is_punctuation_index(idx: int, ptype=polymerType.LPROT):
    return is_gap_index(idx, ptype) or is_stop_index(idx, ptype)


def is_canonical(res: str, ptype=polymerType.LPROT):
    if ptype == polymerType.LPROT or ptype == polymerType.DPROT:
        idx = res_to_index(res, ptype)
        return (idx < 20) and (idx >= 0)
    elif ptype == polymerType.LDPROT:
        return is_canonical(res, polymerType.LPROT) or is_canonical(
            mirror_amino_acid(res), polymerType.DPROT
        )
    raise Exception(f"do not known how to deal with polymer type {ptype}")


def canonical_amino_acids(ptype=polymerType.LPROT):
    canonicals = []
    for aa in _res3[ptype.value]:
        if is_canonical(aa, ptype):
            canonicals.append(aa)
    return canonicals


_add_residue([polymerType.LPROT, polymerType.DPROT], ["ALA", "DAL"], ["A", "a"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["CYS", "DCY"], ["C", "c"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["ASP", "DAS"], ["D", "d"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["GLU", "DGL"], ["E", "e"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["PHE", "DPN"], ["F", "f"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["GLY", "GLY"], ["G", "G"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["HIS", "DHI"], ["H", "h"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["ILE", "DIL"], ["I", "i"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["LYS", "DLY"], ["K", "k"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["LEU", "DLE"], ["L", "l"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["MET", "MED"], ["M", "m"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["ASN", "DSG"], ["N", "n"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["PRO", "DPR"], ["P", "p"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["GLN", "DGN"], ["Q", "q"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["ARG", "DAR"], ["R", "r"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["SER", "DSN"], ["S", "s"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["THR", "DTH"], ["T", "t"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["VAL", "DVA"], ["V", "v"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["TRP", "DTR"], ["W", "w"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["TYR", "DTY"], ["Y", "y"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["HSD", "DSD"], ["H", "h"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["HSE", "DSE"], ["H", "h"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["HSC", "DSC"], ["H", "h"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["HSP", "DSP"], ["H", "h"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["MSE", "DMS"], ["M", "m"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["CSO", "DCS"], ["C", "c"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["SEC", "DEC"], ["C", "c"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["CSX", "DCX"], ["C", "c"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["HIP", "DHP"], ["H", "h"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["SEP", "DEP"], ["S", "s"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["TPO", "DTP"], ["T", "t"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["PTR", "DPT"], ["Y", "y"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["UNK", "UNK"], ["X", "X"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["STP", "STP"], ["*", "*"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["---", "---"], ["-", "-"])

_add_residue([polymerType.LPROT, polymerType.DPROT], ["---", "---"], [".", "."])

for grp in [1, 2, 3]:
    for tp in [polymerType.LPROT, polymerType.DPROT]:
        for idx in range(num_tokens(tp)):
            if grp == 1:
                if not is_punctuation_index(idx, tp) and (
                    not is_unknown_index(idx, tp)
                ):
                    if _res3[tp.value][idx] not in _res3[polymerType.LDPROT.value]:
                        _add_residue(
                            polymerType.LDPROT,
                            _res3[tp.value][idx],
                            _res1[tp.value][idx],
                        )
            elif grp == 2:
                if is_unknown_index(idx, tp):
                    if _res3[tp.value][idx] not in _res3[polymerType.LDPROT.value]:
                        _add_residue(
                            polymerType.LDPROT,
                            _res3[tp.value][idx],
                            _res1[tp.value][idx],
                        )
            elif grp == 3:
                if is_punctuation_index(idx, tp):
                    if _res3[tp.value][idx] not in _res3[polymerType.LDPROT.value]:
                        _add_residue(
                            polymerType.LDPROT,
                            _res3[tp.value][idx],
                            _res1[tp.value][idx],
                        )


def mirror_amino_acid(res: str):
    idx = mirror_amino_acid_index(res_to_index(res, polymerType.LDPROT))
    if len(res) == 1:
        return index_to_single(idx)
    return index_to_triple(idx)


def mirror_amino_acid_index(idx: int):
    N = num_known_molecular_tokens(polymerType.LDPROT)

    # if this is an unknown residue or a punctuation mark, return as is
    if idx >= N:
        return idx

    # otherwise, flip chirality
    return (idx + N // 2) % N

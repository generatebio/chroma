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

import shlex
from dataclasses import dataclass


@dataclass
class PeekedLine:
    line: str
    next_position: int


def peek_line(f, peeked: PeekedLine, rewind=True):
    ret = True
    pos = f.tell()
    line = f.readline()
    if line == "":  # at EOF
        ret = False
    elif line[-1] == "\n":
        line = line[:-1]
    peeked.line = line
    if rewind:
        peeked.next_position = f.tell()
        f.seek(pos)
    else:
        peeked.next_position = pos
    return ret


def advance(f, peeked: PeekedLine):
    f.seek(peeked.next_position)


def star_item_parse(line: str):
    parts = line.split(".")
    if len(parts) < 2:
        raise Exception(f"expected at least two parts in the STAR data line {line}")
    cat = parts[0]
    name_parts = parts[1].split()
    name = name_parts[0]
    if len(name_parts) >= 2:
        val = name_parts[1]
    else:
        val = ""
    return (cat, name, val)


def star_read_data(f, names: list, in_loop: bool, cols=False, has_blocks=True):
    tab = []
    line = ""
    if cols:
        tab = [[] for _ in range(len(names))]
    peeked = PeekedLine("", 0)
    if in_loop:
        heads = []
        while peek_line(f, peeked):
            if not peeked.line.startswith("_"):
                break
            parts = peeked.line.split(".")
            if len(parts) != 2:
                raise Exception(f"expected two parts in the STAR data line {line}")
            heads.append(parts[1].strip())
            advance(f, peeked)

        # figure out which columns we want
        indices = [-1] * len(names)
        for i, name in enumerate(names):
            if name in heads:
                indices[i] = heads.index(name)

        # read each row and get the corresponding columns
        row = [None] * len(heads)
        ma = max(indices)
        while star_read_data_row(f, row, in_loop, has_blocks):
            if (ma >= 0) and (len(row) <= ma):
                raise Exception(f"loop row has insufficient elements: {line}")
            if not cols:
                tab.append([""] * len(names))
                for i, index in enumerate(indices):
                    if cols:
                        tab[i].append(row[index] if index >= 0 else "")
                    else:
                        tab[-1][i] = row[index] if index >= 0 else ""
    else:
        if not cols:
            tab = [[""] * len(names)]
        category, cat, name = "", "", ""

        row = ["", ""]
        while star_read_data_row(f, row, in_loop, has_blocks, peeked):
            cat, name, _ = star_item_parse(row[0])
            if category == "":
                category = cat
            elif category != cat:
                advance(f, peeked)
                break

            if name not in names:
                continue
            idx = names.index(name)
            if cols:
                tab[idx].push_back(row[1])
            else:
                tab[0][idx] = row[1]

    return tab


def star_read_data_row(
    f, row: list, in_loop: bool, has_blocks: bool, peeked: PeekedLine = None
):
    i = 0
    ret = True
    if peeked is None:
        peeked = PeekedLine("", 0)
    while i < len(row):
        if not peek_line(f, peeked, rewind=False):
            if peeked.line == "" and i == 0:
                return False
            raise Exception(f"read {i} tokens when {len(row)} were requested: {row}")
        if (
            peeked.line.startswith("loop_")
            or peeked.line.startswith("data_")
            or (in_loop and peeked.line.startswith("_"))
        ):
            if i == 0:
                advance(f, peeked)
                return False
            raise Exception(
                f"data block ended while reading requested number of tokens: {len(row)}"
            )

        if peeked.line.startswith(";"):
            row[i] = peeked.line[1:]
            while peek_line(f, peeked, rewind=False):
                if peeked.line.startswith(";"):
                    break
                row[i] += peeked.line
            i = i + 1
        elif peeked.line.startswith("#"):
            pass
        else:
            elems = (
                [part for part in shlex.split(peeked.line.strip())]
                if has_blocks
                else peeked.line.strip().split()
            )
            if i + len(elems) > len(row):
                raise Exception(
                    f"too many elements when trying to read {len(row)} tokens; last read: {elems}, row was: {row}, i = {i}"
                )
            for elem in elems:
                row[i] = elem
                i = i + 1

    return ret


def star_string_escape(text):
    # NOTE: has_space designates whether the string really should be quoted, not
    # based on having quote characters within it, but just because of some other
    # reason (e.g., it has spaces or is empty or starts with underscore, which can
    # have special meaning in CIF).
    has_space = (" " in text) or (text == "") or ((len(text) > 0) and (text[0] == "_"))
    has_single = "'" in text
    has_double = '"' in text

    if not has_single and not has_double:
        if not has_space:
            return text
        else:
            return f"'{text}'"
    elif not has_single:
        return f"'{text}'"
    elif not has_double:
        return '"' + text + '"'
    return "\n;" + str + "\n;"


def star_loop_header_write(f, category, names):
    f.write("loop_\n")
    for name in names:
        f.write(f"{category}.{name} \n")


def star_value_defined(val):
    return (val != ".") and (val != "?")


def star_value(val, default):
    if star_value_defined(val):
        return val
    return default


def atom_site_token(value):
    return "." if value == " " else value

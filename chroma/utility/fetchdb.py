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

"""Functions to retrieve information from external databases via their API; Uniprot and RCSB are the primary databases included here.

"""


import requests


def _download_file(url, out_file):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except requests.HTTPError:
        return False


def RCSB_file_download(pdb_id, ext, local_filename):
    """Downloads a file from the RCSB files section.

    Args:
        pdb_id (str) : 4-letter pdb id, case-insensitive
        ext (str) : Extension of file. E.g. ".pdb" or ".pdb1"
        local_filename (str) : Name for downloaded file.
    Returns:
        None
    """
    url = f"https://files.rcsb.org/view/{pdb_id.upper()}{ext}"
    return _download_file(url, local_filename)

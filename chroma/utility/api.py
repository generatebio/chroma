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

import hashlib
import json
import os
import tempfile
from urllib.parse import urlparse

import requests

import chroma

ROOT_DIR = os.path.dirname(os.path.dirname(chroma.__file__))

# Hosts the registered access token is allowed to be sent to. Weight paths are
# often passed as free-form strings (e.g. via model constructor arguments), so
# this prevents the token from being exfiltrated to an arbitrary URL.
ALLOWED_TOKEN_HOSTS = {"chroma-weights.generatebiomedicines.com"}


def download_cache_path(base_url: str, weights_name: str) -> str:
    """Return the local cache path for a named weight download."""
    url_hash = hashlib.md5((base_url + weights_name).encode()).hexdigest()
    temp_dir = os.path.join(tempfile.gettempdir(), "chroma_weights", url_hash)
    return os.path.join(temp_dir, "weights.pt")


def register_key(key: str, key_directory=ROOT_DIR) -> None:
    """
    Registers the provided key by saving it to a JSON file.

    Args:
        key (str): The access token to be registered.
        key_directory (str, optional): The directory where the access key is registered.

    Returns:
        None
    """
    config_path = os.path.join(key_directory, "config.json")
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"access_token": key}, f)


def read_key(key_directory=ROOT_DIR) -> str:
    """
    Reads the registered key from the JSON file. If no key has been registered,
    it informs the user and raises a FileNotFoundError.

    Args:
        key_directory (str, optional): The directory where the access key is registered.

    Returns:
        str: The registered access token.

    Raises:
        FileNotFoundError: If no key has been registered.
    """
    config_path = os.path.join(key_directory, "config.json")

    if not os.path.exists(config_path):
        print("No access token has been registered.")
        print(
            "To obtain an access token, go to https://chroma-weights.generatebiomedicines.com/ and agree to the license."
        )
        raise FileNotFoundError("No token has been registered.")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config["access_token"]


def download_from_generate(
    base_url: str,
    weights_name: str,
    force: bool = False,
    exist_ok: bool = False,
    key_directory=ROOT_DIR,
) -> str:
    """
    Downloads data from the provided URL using the registered access token.
    Provides caching behavior based on force and exist_ok flags.

    Args:
        base_url (str): The base URL from which data should be fetched.
        force (bool): If True, always fetches data from the URL regardless of cache existence.
        exist_ok (bool): If True and cache exists (and force is False), uses the cached data.
        key_directory (str, optional): The directory where the access key is registered.

    Returns:
        str: Path to the downloaded (or cached) file.
    """

    host = urlparse(base_url).netloc
    if host not in ALLOWED_TOKEN_HOSTS:
        raise ValueError(
            f"Refusing to send the registered access token to '{host}': only"
            f" {sorted(ALLOWED_TOKEN_HOSTS)} are allowed. Check that the weights"
            " path/URL is correct."
        )

    destination = download_cache_path(base_url, weights_name)
    temp_dir = os.path.dirname(destination)

    # Ensure the directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # Check if cache exists
    cache_exists = os.path.exists(destination)

    # Determine if we should use the cache or not
    use_cache = cache_exists and exist_ok and not force

    if use_cache:
        print(f"Using cached data from {destination}")
        return destination

    # If not using cache, proceed with download

    # Define the query parameters
    params = {"token": read_key(key_directory), "weights": weights_name}

    # Perform the GET request with the token as a query parameter
    response = requests.get(base_url, params=params, stream=True, timeout=60)
    response.raise_for_status()  # Raise an error for HTTP errors

    # Stream to a per-process temp file and atomically rename it into place so
    # a killed/interrupted download never leaves a truncated file at the
    # canonical cache path for a future call to trust.
    fd, tmp_destination = tempfile.mkstemp(dir=temp_dir)
    try:
        with os.fdopen(fd, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)
        os.replace(tmp_destination, destination)
    except BaseException:
        os.remove(tmp_destination)
        raise

    print(f"Data saved to {destination}")
    return destination

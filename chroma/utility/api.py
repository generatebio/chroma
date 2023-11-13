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

import requests

import chroma

ROOT_DIR = os.path.dirname(os.path.dirname(chroma.__file__))


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
    with open(config_path, "w") as f:
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

    # Create a hash of the URL + weight name to determine the path for the cached/temporary file
    url_hash = hashlib.md5((base_url + weights_name).encode()).hexdigest()
    temp_dir = os.path.join(tempfile.gettempdir(), "chroma_weights", url_hash)
    destination = os.path.join(temp_dir, "weights.pt")

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
    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raise an error for HTTP errors

    with open(destination, "wb") as file:
        file.write(response.content)

    print(f"Data saved to {destination}")
    return destination

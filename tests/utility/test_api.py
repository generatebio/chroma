import os
import socket
import tempfile

import pytest

import chroma
import chroma.utility.api as api
from chroma.models.graph_backbone import GraphBackbone
from chroma.utility.model import load_model

pytestmark = pytest.mark.integration

KEY_PATH = os.path.dirname(os.path.dirname(chroma.__file__))
KEY_PATH = os.path.join(KEY_PATH, "config.json")


@pytest.mark.skipif(not os.path.exists(KEY_PATH), reason="requires config.json")
def test_api():

    # Test Key Registration
    with tempfile.TemporaryDirectory() as key_directory:
        api.register_key("my_key", key_directory)

    # Test Reading
    api.read_key()

    # Test Download
    try:
        socket.getaddrinfo("chroma-weights.generatebiomedicines.com", 443)
    except socket.gaierror:
        pytest.skip("requires network access")

    api.download_from_generate(
        "https://chroma-weights.generatebiomedicines.com/", "chroma_backbone_v1.0.pt"
    )

    # Test Public Loading of BB Model (load a specific model using this requests pull)
    model = load_model(
        "named:public", GraphBackbone, device="cpu", strict_unexpected=False,
    )

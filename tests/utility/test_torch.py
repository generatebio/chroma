import warnings

import torch

from chroma.utility import torch as torch_utils


def test_call_with_mps_cpu_fallback_passthrough_tensor_output():
    torch_utils._MPS_OP_SUPPORT_CACHE.clear()
    x = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    result = torch_utils.call_with_mps_cpu_fallback(torch.linalg.cholesky, x)
    expected = torch.linalg.cholesky(x)

    assert torch.allclose(result, expected)
    assert result.device == x.device


def test_call_with_mps_cpu_fallback_passthrough_tuple_output():
    torch_utils._MPS_OP_SUPPORT_CACHE.clear()
    x = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    values, vectors = torch_utils.call_with_mps_cpu_fallback(torch.linalg.eigh, x)
    expected_values, expected_vectors = torch.linalg.eigh(x)

    assert torch.allclose(values, expected_values)
    assert torch.allclose(vectors.abs(), expected_vectors.abs())
    assert values.device == x.device
    assert vectors.device == x.device


def test_call_with_mps_cpu_fallback_retries_after_mps_error(monkeypatch):
    torch_utils._MPS_OP_SUPPORT_CACHE.clear()
    calls = {"count": 0}
    x = torch.tensor([1.0, 2.0, 3.0])

    def flaky_op(tensor):
        calls["count"] += 1
        raise RuntimeError("MPS backend: operation not implemented")

    monkeypatch.setattr(
        torch_utils, "_find_mps_device", lambda _: torch.device("cpu")
    )
    monkeypatch.setattr(
        torch_utils, "_call_on_cpu", lambda op, args, kwargs, mps_device: x + 1
    )

    result = torch_utils.call_with_mps_cpu_fallback(flaky_op, x)
    result_cached = torch_utils.call_with_mps_cpu_fallback(flaky_op, x)

    assert calls["count"] == 1
    assert torch.equal(result, x + 1)
    assert torch.equal(result_cached, x + 1)
    assert result.device == x.device


def test_call_with_explicit_mps_cpu_fallback_passthrough():
    torch_utils._MPS_OP_SUPPORT_CACHE.clear()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch_utils.call_with_explicit_mps_cpu_fallback(torch.linalg.det, x)

    assert torch.allclose(result, torch.linalg.det(x))


def test_call_with_explicit_mps_cpu_fallback_uses_cpu_path_once(monkeypatch):
    torch_utils._MPS_OP_SUPPORT_CACHE.clear()
    calls = {"count": 0}
    x = torch.tensor([[2.0, 0.0], [0.0, 5.0]])

    def op(tensor):
        calls["count"] += 1
        warnings.warn(
            "The operator 'aten::linalg_svd' is not currently supported on the MPS backend and will fall back to run on the CPU.",
            UserWarning,
        )
        return torch.linalg.det(tensor)

    monkeypatch.setattr(
        torch_utils, "_find_mps_device", lambda _: torch.device("cpu")
    )
    monkeypatch.setattr(
        torch_utils,
        "_call_on_cpu",
        lambda op, args, kwargs, mps_device: torch.linalg.det(x),
    )

    result = torch_utils.call_with_explicit_mps_cpu_fallback(op, x)
    result_cached = torch_utils.call_with_explicit_mps_cpu_fallback(op, x)

    assert calls["count"] == 1
    assert torch.allclose(result, torch.linalg.det(x))
    assert torch.allclose(result_cached, torch.linalg.det(x))
    assert result.device == x.device

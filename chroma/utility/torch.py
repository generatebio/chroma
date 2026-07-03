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

"""Utilities for device-aware torch execution."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import torch

_MPS_OP_SUPPORT_CACHE = {}


# Fix #4: single recursive traversal used by all helpers below.
def _iter_tensors(obj: Any):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item)


def _op_cache_key(op: Callable[..., Any], args: Any, kwargs: Any):
    # Fix #5: include shape so different-sized tensors don't share a cache entry.
    tensor_signature = tuple(
        (str(tensor.dtype), tensor.ndim, tuple(tensor.shape))
        for tensor in _iter_tensors((args, kwargs))
        if tensor.device.type == "mps"
    )
    return (getattr(op, "__module__", ""), getattr(op, "__qualname__", repr(op)), tensor_signature)


# Fix #4: implemented via _iter_tensors instead of a separate recursive walk.
def _find_mps_device(obj: Any) -> torch.device | None:
    return next((t.device for t in _iter_tensors(obj) if t.device.type == "mps"), None)


# Fix #4: unified mover; pass pred=lambda t: t.device.type == "mps" to get the
# old _move_mps_tensors behaviour.
def _move_tensors(obj: Any, device: torch.device, pred=None) -> Any:
    if torch.is_tensor(obj):
        if pred is None or pred(obj):
            return obj.to(device)
        return obj
    if isinstance(obj, tuple):
        return tuple(_move_tensors(item, device, pred) for item in obj)
    if isinstance(obj, list):
        return [_move_tensors(item, device, pred) for item in obj]
    if isinstance(obj, dict):
        return {key: _move_tensors(value, device, pred) for key, value in obj.items()}
    return obj


def _move_mps_tensors(obj: Any, device: torch.device) -> Any:
    return _move_tensors(obj, device, pred=lambda t: t.device.type == "mps")


def _is_mps_not_implemented_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "mps" in message and (
        "not implemented" in message
        or "not currently implemented" in message
        or "not supported" in message
        or "unsupported" in message
    )


def _is_mps_cpu_fallback_warning(warning: Warning) -> bool:
    message = str(warning).lower()
    return "mps backend" in message and "fall back to run on the cpu" in message


def _call_on_cpu(
    op: Callable[..., Any],
    args: Any,
    kwargs: Any,
    mps_device: torch.device,
) -> Any:
    cpu_args = _move_mps_tensors(args, torch.device("cpu"))
    cpu_kwargs = _move_mps_tensors(kwargs, torch.device("cpu"))
    result = op(*cpu_args, **cpu_kwargs)
    return _move_tensors(result, mps_device)


# Fix #7: shared cache-check and CPU-dispatch used by both public helpers.
def _with_mps_fallback_cache(
    op: Callable[..., Any],
    args: Any,
    kwargs: Any,
    mps_device: torch.device,
    try_fn: Callable[[], tuple[bool, Any]],
) -> Any:
    key = _op_cache_key(op, args, kwargs)
    if _MPS_OP_SUPPORT_CACHE.get(key) is False:
        return _call_on_cpu(op, args, kwargs, mps_device)
    if _MPS_OP_SUPPORT_CACHE.get(key) is True:
        return op(*args, **kwargs)
    supported, result = try_fn()
    if not supported:
        _MPS_OP_SUPPORT_CACHE[key] = False
        return _call_on_cpu(op, args, kwargs, mps_device)
    _MPS_OP_SUPPORT_CACHE[key] = True
    return result


def get_default_device() -> torch.device:
    """Return the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def call_with_explicit_mps_cpu_fallback(
    op: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run an op on CPU when given MPS tensors, then move the result back.

    Detects both PyTorch's silent CPU-fallback UserWarning and explicit
    RuntimeError for unsupported MPS ops (fix #3).
    """
    mps_device = _find_mps_device((args, kwargs))
    if mps_device is None:
        return op(*args, **kwargs)

    def _try():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=".*MPS backend.*fall back to run on the CPU.*",
                category=UserWarning,
            )
            try:
                return True, op(*args, **kwargs)
            except UserWarning as exc:
                if not _is_mps_cpu_fallback_warning(exc):
                    raise
                return False, None
            except RuntimeError as exc:
                if not _is_mps_not_implemented_error(exc):
                    raise
                return False, None

    return _with_mps_fallback_cache(op, args, kwargs, mps_device, _try)


def call_with_mps_cpu_fallback(op: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run an op on the current device and retry on CPU if MPS rejects it."""
    mps_device = _find_mps_device((args, kwargs))
    if mps_device is None:
        return op(*args, **kwargs)

    def _try():
        try:
            return True, op(*args, **kwargs)
        except Exception as exc:
            if not _is_mps_not_implemented_error(exc):
                raise
            return False, None

    return _with_mps_fallback_cache(op, args, kwargs, mps_device, _try)

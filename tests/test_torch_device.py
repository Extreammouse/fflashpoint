import pytest

from src.torch_device import resolve_torch_device


def test_resolve_torch_device_accepts_cpu():
    assert resolve_torch_device("cpu") == "cpu"


def test_resolve_torch_device_auto_returns_concrete_device():
    assert resolve_torch_device("auto") in {"cpu", "cuda:0"}


def test_resolve_torch_device_rejects_cuda_when_unavailable():
    if resolve_torch_device("auto") == "cuda:0":
        pytest.skip("CUDA is available in this environment.")

    with pytest.raises(RuntimeError):
        resolve_torch_device("cuda")

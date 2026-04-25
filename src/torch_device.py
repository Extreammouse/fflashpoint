"""Small helpers for choosing a PyTorch device."""

from __future__ import annotations


def resolve_torch_device(device_setting: str | None) -> str:
    """Return a concrete torch device string.

    `auto` chooses CUDA when the installed PyTorch build can actually use it.
    A visible NVIDIA GPU alone is not enough; the venv needs a CUDA-enabled
    PyTorch wheel.
    """
    requested = (device_setting or "auto").strip().lower()
    if requested in {"", "auto"}:
        return "cuda:0" if cuda_is_available() else "cpu"

    if requested.startswith("cuda") and not cuda_is_available():
        raise RuntimeError(
            f"Requested device '{device_setting}', but this PyTorch install cannot use CUDA. "
            "Install a CUDA-enabled torch/torchvision build or set the device to 'cpu'."
        )

    return requested


def cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())

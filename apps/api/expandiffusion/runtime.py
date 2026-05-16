"""Runtime hardware inspection for model loading."""

from __future__ import annotations

from typing import Any

from .schemas import RuntimeDeviceInfo, RuntimeInfo


def inspect_runtime() -> RuntimeInfo:
    """Return the PyTorch runtime visible to the backend process."""
    try:
        import torch
    except ImportError:
        return RuntimeInfo(
            torch_version=None,
            torchvision_version=None,
            cuda_available=False,
            cuda_version=None,
            devices=[],
            preferred_device="cpu",
            preferred_dtype="float32",
            note="PyTorch is not installed.",
        )

    torchvision_version = None
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception as exc:
        torchvision_version = f"unavailable: {exc}"

    devices = _cuda_devices(torch)
    preferred_device = devices[0].id if devices else "cpu"
    preferred_dtype = "float16" if devices else "float32"
    note = (
        f"Using {preferred_device} ({devices[0].name}) with {preferred_dtype}."
        if devices
        else "CUDA is not available; generation will run on CPU with float32."
    )
    return RuntimeInfo(
        torch_version=torch.__version__,
        torchvision_version=torchvision_version,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        devices=devices,
        preferred_device=preferred_device,
        preferred_dtype=preferred_dtype,
        note=note,
    )


def resolve_device_and_dtype(
    torch_module: Any,
    requested_device: str,
    requested_dtype: str,
) -> tuple[str, Any, str]:
    """Resolve one execution path: CUDA when available, otherwise explicit CPU."""
    device = _resolve_device(torch_module, requested_device)
    dtype_name = _resolve_dtype_name(device, requested_dtype)
    return device, _torch_dtype(torch_module, dtype_name), dtype_name


def _cuda_devices(torch_module: Any) -> list[RuntimeDeviceInfo]:
    if not torch_module.cuda.is_available():
        return []
    devices: list[RuntimeDeviceInfo] = []
    for index in range(torch_module.cuda.device_count()):
        name = torch_module.cuda.get_device_name(index)
        total_memory = None
        free_memory = None
        try:
            with torch_module.cuda.device(index):
                free_memory, total_memory = torch_module.cuda.mem_get_info()
        except Exception:
            props = torch_module.cuda.get_device_properties(index)
            total_memory = int(props.total_memory)
        devices.append(
            RuntimeDeviceInfo(
                id=f"cuda:{index}",
                name=name,
                total_memory=total_memory,
                free_memory=free_memory,
            )
        )
    return sorted(
        devices,
        key=lambda item: item.free_memory or item.total_memory or 0,
        reverse=True,
    )


def _resolve_device(torch_module: Any, requested_device: str) -> str:
    normalized = requested_device.strip().lower()
    if normalized in {"", "auto", "cuda"}:
        devices = _cuda_devices(torch_module)
        return devices[0].id if devices else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda:"):
        if not torch_module.cuda.is_available():
            return "cpu"
        index = int(normalized.split(":", 1)[1])
        if index >= torch_module.cuda.device_count():
            return "cpu"
        return normalized
    return "cpu"


def _resolve_dtype_name(device: str, requested_dtype: str) -> str:
    normalized = requested_dtype.strip().lower()
    if device == "cpu":
        return "float32"
    if normalized in {"", "auto"}:
        return "float16"
    if normalized in {"float16", "bfloat16", "float32"}:
        return normalized
    return "float16"


def _torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    return torch_module.float32

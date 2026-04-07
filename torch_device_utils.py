from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None


T = TypeVar("T")


@dataclass(frozen=True)
class TorchExecutionDevice:
    requested: str
    runtime_device: str
    torch_device: str
    label: str
    reason: str | None = None

    @property
    def is_gpu(self) -> bool:
        return self.runtime_device != "cpu"


def _safe_call(func: Callable[[], T], default: T) -> T:
    try:
        return func()
    except Exception:
        return default


def _rocm_available() -> bool:
    if torch is None:
        return False
    return bool(_safe_call(lambda: getattr(torch.version, "hip", None), None))


def _cuda_available() -> bool:
    if torch is None:
        return False
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is None:
        return False
    return bool(_safe_call(cuda_module.is_available, False))


def _xpu_available() -> bool:
    if torch is None:
        return False
    xpu_module = getattr(torch, "xpu", None)
    if xpu_module is None:
        return False
    return bool(_safe_call(xpu_module.is_available, False))


def _cuda_label() -> str:
    return "ROCm" if _rocm_available() else "CUDA"


def available_device_choices() -> tuple[str, ...]:
    return ("auto", "cuda", "xpu", "rocm", "cpu")


def detect_preferred_torch_device() -> TorchExecutionDevice:
    if _cuda_available():
        return TorchExecutionDevice(
            requested="auto",
            runtime_device="cuda",
            torch_device="cuda",
            label=_cuda_label(),
        )
    if _xpu_available():
        return TorchExecutionDevice(
            requested="auto",
            runtime_device="xpu",
            torch_device="xpu",
            label="XPU",
        )
    return TorchExecutionDevice(
        requested="auto",
        runtime_device="cpu",
        torch_device="cpu",
        label="CPU",
    )


def resolve_torch_device(requested: str | None = None) -> TorchExecutionDevice:
    normalized = (requested or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return detect_preferred_torch_device()
    if normalized == "cpu":
        return TorchExecutionDevice(
            requested="cpu",
            runtime_device="cpu",
            torch_device="cpu",
            label="CPU",
        )
    if normalized == "cuda":
        if _cuda_available():
            return TorchExecutionDevice(
                requested="cuda",
                runtime_device="cuda",
                torch_device="cuda",
                label=_cuda_label(),
            )
        return TorchExecutionDevice(
            requested="cuda",
            runtime_device="cpu",
            torch_device="cpu",
            label="CPU",
            reason="CUDA was requested, but PyTorch did not report a usable CUDA backend.",
        )
    if normalized == "rocm":
        if _cuda_available() and _rocm_available():
            return TorchExecutionDevice(
                requested="rocm",
                runtime_device="cuda",
                torch_device="cuda",
                label="ROCm",
            )
        return TorchExecutionDevice(
            requested="rocm",
            runtime_device="cpu",
            torch_device="cpu",
            label="CPU",
            reason="ROCm was requested, but this PyTorch build did not report a usable ROCm backend.",
        )
    if normalized == "xpu":
        if _xpu_available():
            return TorchExecutionDevice(
                requested="xpu",
                runtime_device="xpu",
                torch_device="xpu",
                label="XPU",
            )
        return TorchExecutionDevice(
            requested="xpu",
            runtime_device="cpu",
            torch_device="cpu",
            label="CPU",
            reason="XPU was requested, but PyTorch did not report a usable XPU backend.",
        )
    return TorchExecutionDevice(
        requested=normalized,
        runtime_device="cpu",
        torch_device="cpu",
        label="CPU",
        reason=f"Unsupported device selection '{normalized}'. Falling back to CPU.",
    )

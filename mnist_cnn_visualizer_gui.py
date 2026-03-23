from __future__ import annotations

import argparse
import contextlib
import os
import queue
import shutil
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None
    DataLoader = None
    datasets = None
    transforms = None


INPUT_IMAGE_SIZE = 28
OUTPUT_CLASSES = 10
DIGIT_LABELS = tuple(str(index) for index in range(OUTPUT_CLASSES))
SUPPORTED_ACTIVATIONS = ("relu", "tanh", "sigmoid", "gelu")

SETTING_TOOLTIPS = {
    "data_dir": (
        "Folder used for the MNIST dataset.\n\n"
        "If MNIST is not already inside this folder, the GUI will ask whether it should download it."
    ),
    "epochs": (
        "How many full passes through the MNIST training set to run.\n\n"
        "Higher values usually improve accuracy, but they also take longer."
    ),
    "batch_size": (
        "How many images are processed before one optimizer step.\n\n"
        "Larger batches make training smoother but can feel a little slower per update."
    ),
    "learning_rate": (
        "The AdamW step size.\n\n"
        "Higher values learn faster but can overshoot. Lower values are steadier but slower."
    ),
    "weight_decay": (
        "AdamW weight decay.\n\n"
        "This gently discourages very large weights and can improve generalization."
    ),
    "activation": (
        "Activation function used after convolution and dense layers.\n\n"
        "ReLU is usually the safest starting point for MNIST."
    ),
    "seed": (
        "Optional random seed.\n\n"
        "Leave this empty for a fresh random initialization, or set a value for repeatable runs."
    ),
    "out_channels": (
        "Number of convolution kernels in this layer.\n\n"
        "More kernels means the layer can learn more visual patterns."
    ),
    "kernel_size": (
        "Width and height of each convolution kernel.\n\n"
        "Smaller kernels focus on local details. Larger kernels see a wider area at once."
    ),
    "stride": (
        "Step size for moving the convolution kernel across the image.\n\n"
        "Higher stride reduces spatial resolution faster."
    ),
    "pool_size": (
        "Max-pooling size after the activation.\n\n"
        "Pooling shrinks the feature map and makes the layer more translation tolerant."
    ),
    "units": (
        "Number of neurons in this dense layer.\n\n"
        "Dense layers combine the learned visual features before the fixed 10-digit output."
    ),
    "sample_index": (
        "Which test sample to inspect.\n\n"
        "When training is paused or finished, step through different test digits to see predictions."
    ),
    "visual_zoom": (
        "Scales the network visualizer in the main display area.\n\n"
        "Use Fit to View to resize the whole architecture so it fits the available canvas more comfortably."
    ),
}


@dataclass(slots=True)
class ConvLayerConfig:
    out_channels: int = 8
    kernel_size: int = 5
    stride: int = 1
    pool_size: int = 2


@dataclass(slots=True)
class DenseLayerConfig:
    units: int = 64


@dataclass(slots=True)
class AppConfig:
    data_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    activation: str
    conv_layers: list[ConvLayerConfig] = field(default_factory=list)
    dense_layers: list[DenseLayerConfig] = field(default_factory=list)
    seed: int | None = None
    smoke_test: bool = False


@dataclass(slots=True)
class MetricPoint:
    epoch: int
    batch: int
    train_loss: float
    train_acc: float
    test_loss: float | None
    test_acc: float | None
    status: str


@dataclass(slots=True)
class InspectionSnapshot:
    image: np.ndarray
    target: int
    prediction: int
    probabilities: np.ndarray
    conv_kernels: list[np.ndarray]
    conv_activations: list[np.ndarray]
    dense_activations: list[np.ndarray]
    logits: np.ndarray


def activation_module(name: str) -> nn.Module:
    if nn is None:
        raise RuntimeError("PyTorch is required for activation modules.")
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def dataset_present(data_dir: Path) -> bool:
    if datasets is None:
        raise RuntimeError("torchvision is required for MNIST support.")
    try:
        datasets.MNIST(root=str(data_dir), train=True, download=False)
        datasets.MNIST(root=str(data_dir), train=False, download=False)
    except RuntimeError:
        return False
    return True


@contextlib.contextmanager
def safe_download_streams():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    fallback_stream = None
    try:
        if (
            original_stdout is None
            or original_stderr is None
            or not hasattr(original_stdout, "write")
            or not hasattr(original_stderr, "write")
        ):
            fallback_stream = open(os.devnull, "w", encoding="utf-8")
            if original_stdout is None or not hasattr(original_stdout, "write"):
                sys.stdout = fallback_stream
            if original_stderr is None or not hasattr(original_stderr, "write"):
                sys.stderr = fallback_stream
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if fallback_stream is not None:
            fallback_stream.close()


def download_mnist_dataset(data_dir: Path) -> None:
    if datasets is None:
        raise RuntimeError("torchvision is required for MNIST support.")

    data_dir.mkdir(parents=True, exist_ok=True)
    mnist_root = data_dir / "MNIST"
    raw_dir = mnist_root / "raw"
    processed_dir = mnist_root / "processed"

    try:
        with safe_download_streams():
            datasets.MNIST(root=str(data_dir), train=True, download=True)
            datasets.MNIST(root=str(data_dir), train=False, download=True)
        return
    except Exception as first_exc:
        # Interrupted or partially corrupted raw downloads can leave MNIST in a state
        # where a clean retry is more reliable than reusing the leftover archives.
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)
        try:
            with safe_download_streams():
                datasets.MNIST(root=str(data_dir), train=True, download=True)
                datasets.MNIST(root=str(data_dir), train=False, download=True)
            return
        except Exception as second_exc:
            raise RuntimeError(
                "Could not download the MNIST dataset.\n\n"
                f"First attempt: {first_exc}\n\n"
                f"Retry after clearing partial files: {second_exc}"
            ) from second_exc


def ensure_mnist_dataset(parent: tk.Misc | None, data_dir: Path, allow_prompt: bool) -> bool:
    if dataset_present(data_dir):
        return True
    if not allow_prompt:
        return False
    approved = False
    if parent is not None:
        approved = messagebox.askyesno(
            "Download MNIST?",
            f"MNIST was not found in:\n{data_dir}\n\nDownload it now?",
            parent=parent,
        )
    if not approved:
        return False
    try:
        download_mnist_dataset(data_dir)
    except Exception as exc:  # pragma: no cover - depends on runtime/network
        if parent is not None:
            messagebox.showerror("MNIST download failed", str(exc), parent=parent)
        else:
            raise
        return False
    return True


def blend_color(start_hex: str, end_hex: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    start_rgb = tuple(int(start_hex[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end_hex[index : index + 2], 16) for index in (1, 3, 5))
    blended = tuple(int(round(a + (b - a) * amount)) for a, b in zip(start_rgb, end_rgb))
    return "#" + "".join(f"{value:02x}" for value in blended)


def kernel_color(value: float, scale: float) -> str:
    if scale <= 1e-8:
        return "#080b10"
    amount = min(1.0, abs(value) / scale)
    if value >= 0.0:
        return blend_color("#080b10", "#f5db2b", amount)
    return blend_color("#080b10", "#4c86ff", amount)


def image_color(value: float) -> str:
    return blend_color("#090b10", "#f2df60", max(0.0, min(1.0, value)))


def activation_color(value: float, scale: float) -> str:
    if scale <= 1e-8:
        return "#0b1218"
    amount = min(1.0, max(0.0, value) / scale)
    return blend_color("#0b1218", "#49d9ff", amount)


class HoverTooltip:
    def __init__(self, widget: tk.Widget, text: str, wraplength: int = 460) -> None:
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tipwindow: tk.Toplevel | None = None
        self.after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Motion>", self._move, add="+")

    def _schedule(self, _event=None) -> None:
        self._unschedule()
        self.after_id = self.widget.after(260, self._show)

    def _unschedule(self) -> None:
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def _show(self) -> None:
        if self.tipwindow is not None:
            return
        self.tipwindow = tk.Toplevel(self.widget)
        self.tipwindow.wm_overrideredirect(True)
        try:
            self.tipwindow.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        frame = tk.Frame(self.tipwindow, background="#fff8dd", borderwidth=1, relief="solid")
        frame.pack(fill="both", expand=True)
        label = tk.Label(
            frame,
            text=self.text,
            justify="left",
            background="#fff8dd",
            foreground="#22343c",
            font=("Segoe UI", 11),
            wraplength=self.wraplength,
            padx=14,
            pady=12,
        )
        label.pack(fill="both", expand=True)
        self._position()

    def _position(self) -> None:
        if self.tipwindow is None:
            return
        x = self.widget.winfo_pointerx() + 18
        y = self.widget.winfo_pointery() + 18
        self.tipwindow.wm_geometry(f"+{x}+{y}")

    def _move(self, _event=None) -> None:
        if self.tipwindow is not None:
            self._position()

    def _hide(self, _event=None) -> None:
        self._unschedule()
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


class ScrollableSidebar(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg="#f4f1e8", highlightthickness=0, width=380)
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.inner = ttk.Frame(self.canvas, padding=14)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind("<Configure>", self._sync_width)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _sync_width(self, event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)


class MnistCNN(nn.Module):
    def __init__(
        self,
        conv_layers: list[ConvLayerConfig],
        dense_layers: list[DenseLayerConfig],
        activation_name: str,
    ) -> None:
        super().__init__()
        if nn is None:
            raise RuntimeError("PyTorch is required for the MNIST visualizer.")
        self.activation_name = activation_name
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for layer_config in conv_layers:
            block = nn.ModuleDict(
                {
                    "conv": nn.Conv2d(
                        in_channels,
                        layer_config.out_channels,
                        kernel_size=layer_config.kernel_size,
                        stride=layer_config.stride,
                        padding=layer_config.kernel_size // 2,
                    ),
                    "activation": activation_module(activation_name),
                    "pool": nn.MaxPool2d(layer_config.pool_size) if layer_config.pool_size > 1 else nn.Identity(),
                }
            )
            self.conv_layers.append(block)
            in_channels = layer_config.out_channels

        flat_dim = self._infer_flattened_size()
        self.dense_layers = nn.ModuleList()
        self.dense_activations = nn.ModuleList()
        previous_units = flat_dim
        for dense_config in dense_layers:
            self.dense_layers.append(nn.Linear(previous_units, dense_config.units))
            self.dense_activations.append(activation_module(activation_name))
            previous_units = dense_config.units
        self.output_layer = nn.Linear(previous_units, OUTPUT_CLASSES)

    def _infer_flattened_size(self) -> int:
        with torch.no_grad():
            sample = torch.zeros(1, 1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
            for block in self.conv_layers:
                sample = block["conv"](sample)
                sample = block["activation"](sample)
                sample = block["pool"](sample)
            if sample.numel() <= 0:
                raise ValueError("The current convolution and pooling settings reduce the image to nothing.")
            return int(sample.numel())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for block in self.conv_layers:
            values = block["conv"](values)
            values = block["activation"](values)
            values = block["pool"](values)
        values = torch.flatten(values, 1)
        for layer, activation in zip(self.dense_layers, self.dense_activations):
            values = activation(layer(values))
        return self.output_layer(values)

    def inspect(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        values = inputs
        conv_activations: list[torch.Tensor] = []
        for block in self.conv_layers:
            values = block["conv"](values)
            values = block["activation"](values)
            values = block["pool"](values)
            conv_activations.append(values)
        values = torch.flatten(values, 1)
        dense_activations: list[torch.Tensor] = []
        for layer, activation in zip(self.dense_layers, self.dense_activations):
            values = activation(layer(values))
            dense_activations.append(values)
        logits = self.output_layer(values)
        return logits, conv_activations, dense_activations

    def conv_weight_grids(self) -> list[np.ndarray]:
        kernels: list[np.ndarray] = []
        for block in self.conv_layers:
            weight = block["conv"].weight.detach().cpu().numpy()
            kernels.append(weight.mean(axis=1))
        return kernels


class MnistTrainingEngine:
    def __init__(self, config: AppConfig, data_dir: Path) -> None:
        if torch is None or nn is None or datasets is None or transforms is None or DataLoader is None:
            raise RuntimeError("PyTorch and torchvision are required for the MNIST visualizer.")

        self.config = config
        self.data_dir = data_dir
        self.device = torch.device("cpu")
        self.metric_queue: queue.Queue[MetricPoint] = queue.Queue()
        self.model_lock = threading.Lock()
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.training_thread: threading.Thread | None = None
        self.status = "idle"
        self.current_epoch = 0
        self.global_step = 0
        self.latest_metric = MetricPoint(0, 0, 0.0, 0.0, None, None, "idle")
        self.history: list[MetricPoint] = []

        if config.seed is not None:
            torch.manual_seed(config.seed)

        transform = transforms.ToTensor()
        self.train_dataset = datasets.MNIST(root=str(data_dir), train=True, transform=transform, download=False)
        self.test_dataset = datasets.MNIST(root=str(data_dir), train=False, transform=transform, download=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False)

        self.model = MnistCNN(config.conv_layers, config.dense_layers, config.activation).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def start(self) -> None:
        if self.training_thread is not None and self.training_thread.is_alive():
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.status = "running"
        self.training_thread = threading.Thread(target=self._train_loop, daemon=True)
        self.training_thread.start()

    def pause(self) -> None:
        self.pause_event.set()
        self.status = "paused"

    def resume(self) -> None:
        if self.training_thread is None or not self.training_thread.is_alive():
            return
        self.pause_event.clear()
        self.status = "running"

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        if self.training_thread is not None and self.training_thread.is_alive():
            self.training_thread.join(timeout=1.0)
        self.status = "stopped"

    def _train_loop(self) -> None:
        for epoch in range(self.current_epoch, self.config.epochs):
            if self.stop_event.is_set():
                break
            self.current_epoch = epoch + 1
            self._run_epoch(epoch + 1)
            if self.stop_event.is_set():
                break
            test_loss, test_acc = self.evaluate()
            epoch_metric = MetricPoint(
                epoch=epoch + 1,
                batch=len(self.train_loader),
                train_loss=self.latest_metric.train_loss,
                train_acc=self.latest_metric.train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                status="epoch_end",
            )
            self.latest_metric = epoch_metric
            self.history.append(epoch_metric)
            self.metric_queue.put(epoch_metric)

        self.status = "complete" if not self.stop_event.is_set() else "stopped"
        final_metric = MetricPoint(
            epoch=self.current_epoch,
            batch=self.latest_metric.batch,
            train_loss=self.latest_metric.train_loss,
            train_acc=self.latest_metric.train_acc,
            test_loss=self.latest_metric.test_loss,
            test_acc=self.latest_metric.test_acc,
            status=self.status,
        )
        self.metric_queue.put(final_metric)

    def _run_epoch(self, epoch_number: int) -> None:
        running_loss = 0.0
        correct = 0
        total = 0
        self.model.train()
        for batch_index, (images, labels) in enumerate(self.train_loader, start=1):
            if self.stop_event.is_set():
                return
            while self.pause_event.is_set():
                if self.stop_event.wait(0.05):
                    return
            images = images.to(self.device)
            labels = labels.to(self.device)
            with self.model_lock:
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
            self.global_step += 1

            if batch_index % 25 == 0 or batch_index == len(self.train_loader):
                metric = MetricPoint(
                    epoch=epoch_number,
                    batch=batch_index,
                    train_loss=running_loss / max(total, 1),
                    train_acc=correct / max(total, 1),
                    test_loss=self.latest_metric.test_loss,
                    test_acc=self.latest_metric.test_acc,
                    status="running",
                )
                self.latest_metric = metric
                self.metric_queue.put(metric)

    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                with self.model_lock:
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)
                total_loss += float(loss.item()) * labels.size(0)
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_items += int(labels.size(0))
        return total_loss / max(total_items, 1), total_correct / max(total_items, 1)

    def inspect_sample(self, sample_index: int) -> InspectionSnapshot:
        if sample_index < 0 or sample_index >= len(self.test_dataset):
            raise IndexError("sample index out of range")
        image_tensor, target = self.test_dataset[sample_index]
        inputs = image_tensor.unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            with self.model_lock:
                logits, conv_activations, dense_activations = self.model.inspect(inputs)
                probabilities = torch.softmax(logits[0], dim=0)
                prediction = int(torch.argmax(probabilities).item())
                kernels = self.model.conv_weight_grids()

        activation_grids: list[np.ndarray] = []
        for activation in conv_activations:
            activation_grids.append(activation[0].detach().cpu().numpy())
        dense_values = [activation[0].detach().cpu().numpy() for activation in dense_activations]
        return InspectionSnapshot(
            image=image_tensor.squeeze(0).numpy(),
            target=int(target),
            prediction=prediction,
            probabilities=probabilities.detach().cpu().numpy(),
            conv_kernels=kernels,
            conv_activations=activation_grids,
            dense_activations=dense_values,
            logits=logits[0].detach().cpu().numpy(),
        )


def default_config() -> AppConfig:
    return AppConfig(
        data_dir=str(Path(__file__).resolve().with_name("mnist_data")),
        epochs=6,
        batch_size=64,
        learning_rate=0.0010,
        weight_decay=0.00005,
        activation="relu",
        conv_layers=[ConvLayerConfig(8, 5, 1, 2), ConvLayerConfig(16, 3, 1, 2)],
        dense_layers=[DenseLayerConfig(64)],
        seed=None,
    )


def validate_config(config: AppConfig) -> list[str]:
    errors: list[str] = []
    if not config.data_dir.strip():
        errors.append("Dataset folder cannot be empty.")
    if config.epochs <= 0:
        errors.append("Epochs must be greater than zero.")
    if config.batch_size <= 0:
        errors.append("Batch size must be greater than zero.")
    if config.learning_rate <= 0.0:
        errors.append("Learning rate must be greater than zero.")
    if config.weight_decay < 0.0:
        errors.append("Weight decay cannot be negative.")
    if config.activation not in SUPPORTED_ACTIVATIONS:
        errors.append(f"Activation must be one of: {', '.join(SUPPORTED_ACTIVATIONS)}.")
    for index, layer in enumerate(config.conv_layers, start=1):
        if layer.out_channels <= 0:
            errors.append(f"Conv layer {index}: kernels must be greater than zero.")
        if layer.kernel_size <= 0:
            errors.append(f"Conv layer {index}: kernel size must be greater than zero.")
        if layer.stride <= 0:
            errors.append(f"Conv layer {index}: stride must be greater than zero.")
        if layer.pool_size <= 0:
            errors.append(f"Conv layer {index}: pool size must be greater than zero.")
    for index, layer in enumerate(config.dense_layers, start=1):
        if layer.units <= 0:
            errors.append(f"Dense layer {index}: units must be greater than zero.")
    if not config.conv_layers and not config.dense_layers:
        errors.append("Add at least one convolution or dense layer before starting.")

    if errors or torch is None or nn is None:
        return errors

    try:
        MnistCNN(config.conv_layers, config.dense_layers, config.activation)
    except Exception as exc:
        errors.append(f"Architecture is not valid for a 28x28 MNIST image: {exc}")
    return errors


class MnistVisualizerApp:
    def __init__(self, root: tk.Tk, initial_config: AppConfig, prompt_for_dataset: bool = True) -> None:
        if torch is None or nn is None or datasets is None or transforms is None:
            raise RuntimeError("PyTorch and torchvision are required. Install them in sd-env first.")

        self.root = root
        self.prompt_for_dataset = prompt_for_dataset
        self.engine: MnistTrainingEngine | None = None
        self.current_snapshot: InspectionSnapshot | None = None
        self.current_sample_index = 0
        self.metric_history: list[MetricPoint] = []
        self.tooltips: list[HoverTooltip] = []
        self.redraw_after_id: str | None = None

        self.root.title("MNIST CNN Kernel Visualizer")
        self.root.geometry("1720x980")
        self.root.minsize(1320, 840)
        self.root.configure(bg="#e9e3d5")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.data_dir_var = tk.StringVar(value=initial_config.data_dir)
        self.epochs_var = tk.StringVar(value=str(initial_config.epochs))
        self.batch_size_var = tk.StringVar(value=str(initial_config.batch_size))
        self.learning_rate_var = tk.StringVar(value=f"{initial_config.learning_rate:.6f}")
        self.weight_decay_var = tk.StringVar(value=f"{initial_config.weight_decay:.6f}")
        self.activation_var = tk.StringVar(value=initial_config.activation)
        self.seed_var = tk.StringVar(value="" if initial_config.seed is None else str(initial_config.seed))
        self.sample_index_var = tk.StringVar(value="0")
        self.visual_zoom_var = tk.DoubleVar(value=100.0)
        self.zoom_label_var = tk.StringVar(value="100%")

        self.dataset_var = tk.StringVar(value="Dataset: checking...")
        self.progress_var = tk.StringVar(
            value="Epoch 0/0 | Train acc 0.0% | Test acc -- | Loss --"
        )
        self.status_var = tk.StringVar(
            value="Configure the layers on the left, then press Start / Restart."
        )
        self.sample_var = tk.StringVar(value="Sample 0 | Prediction -- | Target --")

        self.conv_layer_vars: list[dict[str, tk.StringVar]] = []
        self.dense_layer_vars: list[dict[str, tk.StringVar]] = []

        self._build_shell()
        self._load_layer_variables(initial_config)
        self._render_layer_editors()
        self._update_dataset_status()
        self._update_button_states()
        self._update_zoom_label()
        self._redraw_network()
        self.root.after(180, self._fit_visualizer)
        self._poll_engine()

        if self.prompt_for_dataset:
            self.root.after(350, self._initial_dataset_check)

    def _build_shell(self) -> None:
        panes = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)

        self.sidebar = ScrollableSidebar(panes)
        panes.add(self.sidebar, weight=0)
        self._build_sidebar(self.sidebar.inner)

        right = ttk.Frame(panes, padding=(14, 14, 14, 12))
        panes.add(right, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        header = ttk.Frame(right)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="MNIST CNN Kernel Visualizer",
            font=("Segoe UI Semibold", 24),
            foreground="#183443",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            textvariable=self.progress_var,
            font=("Segoe UI Semibold", 18),
            foreground="#0f4764",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(
            header,
            textvariable=self.dataset_var,
            font=("Segoe UI", 12),
            foreground="#315261",
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))

        ttk.Separator(right, orient=tk.HORIZONTAL).grid(row=1, column=0, sticky="ew", pady=(12, 12))

        viewer_controls = ttk.Frame(right)
        viewer_controls.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        viewer_controls.columnconfigure(2, weight=1)

        zoom_title = ttk.Label(viewer_controls, text="Visualizer zoom", font=("Segoe UI Semibold", 11))
        zoom_title.grid(row=0, column=0, sticky="w")
        self._attach_tooltip(zoom_title, SETTING_TOOLTIPS["visual_zoom"])

        zoom_value = ttk.Label(viewer_controls, textvariable=self.zoom_label_var, font=("Segoe UI", 11))
        zoom_value.grid(row=0, column=1, sticky="w", padx=(10, 10))
        self._attach_tooltip(zoom_value, SETTING_TOOLTIPS["visual_zoom"])

        zoom_slider = ttk.Scale(
            viewer_controls,
            from_=55.0,
            to=185.0,
            variable=self.visual_zoom_var,
            command=self._on_zoom_slider,
        )
        zoom_slider.grid(row=0, column=2, sticky="ew", padx=(0, 10))
        self._attach_tooltip(zoom_slider, SETTING_TOOLTIPS["visual_zoom"])

        zoom_out = ttk.Button(viewer_controls, text="-", width=3, command=lambda: self._change_zoom(-10.0))
        zoom_out.grid(row=0, column=3, padx=(0, 6))
        self._attach_tooltip(zoom_out, SETTING_TOOLTIPS["visual_zoom"])

        zoom_in = ttk.Button(viewer_controls, text="+", width=3, command=lambda: self._change_zoom(10.0))
        zoom_in.grid(row=0, column=4, padx=(0, 6))
        self._attach_tooltip(zoom_in, SETTING_TOOLTIPS["visual_zoom"])

        fit_button = ttk.Button(viewer_controls, text="Fit to View", command=self._fit_visualizer)
        fit_button.grid(row=0, column=5)
        self._attach_tooltip(fit_button, SETTING_TOOLTIPS["visual_zoom"])

        canvas_frame = ttk.Frame(right)
        canvas_frame.grid(row=3, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.network_canvas = tk.Canvas(
            canvas_frame,
            bg="#05070b",
            highlightthickness=0,
            width=1150,
            height=840,
        )
        self.network_canvas.grid(row=0, column=0, sticky="nsew")

        x_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.network_canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.network_canvas.yview)
        self.network_canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        x_scroll.grid(row=1, column=0, sticky="ew")
        y_scroll.grid(row=0, column=1, sticky="ns")

        self.network_canvas.bind("<Configure>", lambda _event: self._schedule_redraw_network())
        self.network_canvas.bind("<Control-MouseWheel>", self._on_canvas_zoom)
        self.network_canvas.bind("<Control-Button-4>", self._on_canvas_zoom)
        self.network_canvas.bind("<Control-Button-5>", self._on_canvas_zoom)

        status_bar = ttk.Label(
            right,
            textvariable=self.status_var,
            font=("Segoe UI", 12),
            anchor="w",
            foreground="#233943",
        )
        status_bar.grid(row=4, column=0, sticky="ew", pady=(10, 0))

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="Settings",
            font=("Segoe UI Semibold", 18),
            foreground="#183443",
        ).pack(anchor="w", pady=(0, 12))

        train_frame = ttk.LabelFrame(parent, text="Training")
        train_frame.pack(fill=tk.X, pady=(0, 12))

        self._add_labeled_entry(train_frame, "Dataset folder", self.data_dir_var, "data_dir", width=32)
        self._add_labeled_entry(train_frame, "Epochs", self.epochs_var, "epochs", width=12)
        self._add_labeled_entry(train_frame, "Batch size", self.batch_size_var, "batch_size", width=12)
        self._add_labeled_entry(train_frame, "Learning rate", self.learning_rate_var, "learning_rate", width=12)
        self._add_labeled_entry(train_frame, "Weight decay", self.weight_decay_var, "weight_decay", width=12)
        self._add_labeled_combobox(train_frame, "Activation", self.activation_var, SUPPORTED_ACTIVATIONS, "activation")
        self._add_labeled_entry(train_frame, "Seed", self.seed_var, "seed", width=12)

        buttons = ttk.Frame(train_frame)
        buttons.pack(fill=tk.X, padx=10, pady=(6, 10))
        buttons.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(buttons, text="Start / Restart", command=self._start_training)
        self.pause_button = ttk.Button(buttons, text="Pause", command=self._pause_training)
        self.resume_button = ttk.Button(buttons, text="Resume", command=self._resume_training)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop_training)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        self.pause_button.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        self.resume_button.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self.stop_button.grid(row=1, column=1, sticky="ew")

        inspect_frame = ttk.LabelFrame(parent, text="Inspect Test Sample")
        inspect_frame.pack(fill=tk.X, pady=(0, 12))

        sample_row = ttk.Frame(inspect_frame)
        sample_row.pack(fill=tk.X, padx=10, pady=(8, 4))
        sample_label = ttk.Label(sample_row, text="Sample index")
        sample_label.pack(side=tk.LEFT)
        sample_entry = ttk.Entry(sample_row, textvariable=self.sample_index_var, width=10)
        sample_entry.pack(side=tk.LEFT, padx=(8, 6))
        self._attach_tooltip(sample_label, SETTING_TOOLTIPS["sample_index"])
        self._attach_tooltip(sample_entry, SETTING_TOOLTIPS["sample_index"])
        ttk.Button(sample_row, text="Go", command=self._jump_to_sample).pack(side=tk.LEFT)

        sample_buttons = ttk.Frame(inspect_frame)
        sample_buttons.pack(fill=tk.X, padx=10, pady=(4, 8))
        sample_buttons.columnconfigure((0, 1, 2), weight=1)
        self.prev_button = ttk.Button(sample_buttons, text="Previous", command=lambda: self._step_sample(-1))
        self.next_button = ttk.Button(sample_buttons, text="Next", command=lambda: self._step_sample(1))
        self.random_button = ttk.Button(sample_buttons, text="Random", command=self._random_sample)
        self.prev_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.next_button.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.random_button.grid(row=0, column=2, sticky="ew")
        ttk.Label(
            inspect_frame,
            textvariable=self.sample_var,
            font=("Segoe UI", 11),
            foreground="#2e4a52",
        ).pack(anchor="w", padx=10, pady=(0, 10))

        arch_frame = ttk.LabelFrame(parent, text="Architecture")
        arch_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        ttk.Label(
            arch_frame,
            text="Add layers, then use Start / Restart to apply the updated architecture.",
            wraplength=310,
            font=("Segoe UI", 10),
            foreground="#415761",
        ).pack(anchor="w", padx=10, pady=(8, 10))

        arch_buttons = ttk.Frame(arch_frame)
        arch_buttons.pack(fill=tk.X, padx=10, pady=(0, 10))
        arch_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(arch_buttons, text="Add Conv Layer", command=self._add_conv_layer).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 6),
        )
        ttk.Button(arch_buttons, text="Add Dense Layer", command=self._add_dense_layer).grid(
            row=0,
            column=1,
            sticky="ew",
        )
        ttk.Button(arch_buttons, text="Refresh View", command=self._redraw_network).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 6),
            pady=(6, 0),
        )
        ttk.Button(arch_buttons, text="Reset Defaults", command=self._reset_defaults).grid(
            row=1,
            column=1,
            sticky="ew",
            pady=(6, 0),
        )

        self.conv_section = ttk.LabelFrame(arch_frame, text="Convolution Layers")
        self.conv_section.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.dense_section = ttk.LabelFrame(arch_frame, text="Dense Layers")
        self.dense_section.pack(fill=tk.X, padx=10, pady=(0, 10))

    def _attach_tooltip(self, widget: tk.Widget, text: str | None) -> None:
        if text:
            self.tooltips.append(HoverTooltip(widget, text))

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        label_text: str,
        variable: tk.StringVar,
        tooltip_key: str,
        width: int = 14,
    ) -> ttk.Entry:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=10, pady=4)
        label = ttk.Label(row, text=label_text, width=16)
        label.pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=variable, width=width)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._attach_tooltip(label, SETTING_TOOLTIPS.get(tooltip_key))
        self._attach_tooltip(entry, SETTING_TOOLTIPS.get(tooltip_key))
        if tooltip_key == "data_dir":
            variable.trace_add("write", lambda *_args: self._update_dataset_status())
            return entry
        if tooltip_key == "activation":
            return entry
        variable.trace_add("write", lambda *_args: self._schedule_redraw_network())
        return entry

    def _add_labeled_combobox(
        self,
        parent: ttk.Frame,
        label_text: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
        tooltip_key: str,
    ) -> ttk.Combobox:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=10, pady=4)
        label = ttk.Label(row, text=label_text, width=16)
        label.pack(side=tk.LEFT)
        combo = ttk.Combobox(row, textvariable=variable, values=values, width=12, state="readonly")
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._attach_tooltip(label, SETTING_TOOLTIPS.get(tooltip_key))
        self._attach_tooltip(combo, SETTING_TOOLTIPS.get(tooltip_key))
        variable.trace_add("write", lambda *_args: self._schedule_redraw_network())
        return combo

    def _make_var(self, value: int | float | str) -> tk.StringVar:
        variable = tk.StringVar(value=str(value))
        variable.trace_add("write", lambda *_args: self._schedule_redraw_network())
        return variable

    def _load_layer_variables(self, config: AppConfig) -> None:
        self.conv_layer_vars = [
            {
                "out_channels": self._make_var(layer.out_channels),
                "kernel_size": self._make_var(layer.kernel_size),
                "stride": self._make_var(layer.stride),
                "pool_size": self._make_var(layer.pool_size),
            }
            for layer in config.conv_layers
        ]
        self.dense_layer_vars = [{"units": self._make_var(layer.units)} for layer in config.dense_layers]

    def _render_layer_editors(self) -> None:
        for frame in (self.conv_section, self.dense_section):
            for child in frame.winfo_children():
                child.destroy()

        if not self.conv_layer_vars:
            ttk.Label(
                self.conv_section,
                text="No convolution layers yet.",
                foreground="#556972",
            ).pack(anchor="w", padx=10, pady=8)
        for index, layer_vars in enumerate(self.conv_layer_vars, start=1):
            card = ttk.Frame(self.conv_section, padding=8)
            card.pack(fill=tk.X, padx=6, pady=4)
            header = ttk.Frame(card)
            header.pack(fill=tk.X)
            ttk.Label(header, text=f"Conv {index}", font=("Segoe UI Semibold", 11)).pack(side=tk.LEFT)
            ttk.Button(
                header,
                text="Remove",
                command=lambda layer_index=index - 1: self._remove_conv_layer(layer_index),
            ).pack(side=tk.RIGHT)

            self._card_entry(card, "Kernels", layer_vars["out_channels"], "out_channels")
            self._card_entry(card, "Kernel size", layer_vars["kernel_size"], "kernel_size")
            self._card_entry(card, "Stride", layer_vars["stride"], "stride")
            self._card_entry(card, "Pool size", layer_vars["pool_size"], "pool_size")

        if not self.dense_layer_vars:
            ttk.Label(
                self.dense_section,
                text="No dense hidden layers yet. Output stays fixed at 10 digits.",
                foreground="#556972",
                wraplength=290,
            ).pack(anchor="w", padx=10, pady=8)
        for index, layer_vars in enumerate(self.dense_layer_vars, start=1):
            card = ttk.Frame(self.dense_section, padding=8)
            card.pack(fill=tk.X, padx=6, pady=4)
            header = ttk.Frame(card)
            header.pack(fill=tk.X)
            ttk.Label(header, text=f"Dense {index}", font=("Segoe UI Semibold", 11)).pack(side=tk.LEFT)
            ttk.Button(
                header,
                text="Remove",
                command=lambda layer_index=index - 1: self._remove_dense_layer(layer_index),
            ).pack(side=tk.RIGHT)

            self._card_entry(card, "Units", layer_vars["units"], "units")

    def _card_entry(self, parent: ttk.Frame, label_text: str, variable: tk.StringVar, tooltip_key: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        label = ttk.Label(row, text=label_text, width=12)
        label.pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=variable, width=10)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._attach_tooltip(label, SETTING_TOOLTIPS.get(tooltip_key))
        self._attach_tooltip(entry, SETTING_TOOLTIPS.get(tooltip_key))

    def _add_conv_layer(self) -> None:
        self.conv_layer_vars.append(
            {
                "out_channels": self._make_var(16),
                "kernel_size": self._make_var(3),
                "stride": self._make_var(1),
                "pool_size": self._make_var(2),
            }
        )
        self._render_layer_editors()
        self._redraw_network()

    def _add_dense_layer(self) -> None:
        self.dense_layer_vars.append({"units": self._make_var(64)})
        self._render_layer_editors()
        self._redraw_network()

    def _remove_conv_layer(self, index: int) -> None:
        if 0 <= index < len(self.conv_layer_vars):
            self.conv_layer_vars.pop(index)
            self._render_layer_editors()
            self._redraw_network()

    def _remove_dense_layer(self, index: int) -> None:
        if 0 <= index < len(self.dense_layer_vars):
            self.dense_layer_vars.pop(index)
            self._render_layer_editors()
            self._redraw_network()

    def _reset_defaults(self) -> None:
        config = default_config()
        self.data_dir_var.set(config.data_dir)
        self.epochs_var.set(str(config.epochs))
        self.batch_size_var.set(str(config.batch_size))
        self.learning_rate_var.set(f"{config.learning_rate:.6f}")
        self.weight_decay_var.set(f"{config.weight_decay:.6f}")
        self.activation_var.set(config.activation)
        self.seed_var.set("" if config.seed is None else str(config.seed))
        self.sample_index_var.set("0")
        self._load_layer_variables(config)
        self._render_layer_editors()
        self._redraw_network()

    def _parse_int(self, value: str, label: str, minimum: int = 1, allow_empty: bool = False) -> int | None:
        stripped = value.strip()
        if not stripped and allow_empty:
            return None
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc
        if parsed < minimum:
            raise ValueError(f"{label} must be at least {minimum}.")
        return parsed

    def _parse_float(self, value: str, label: str, minimum: float = 0.0, strictly_positive: bool = False) -> float:
        try:
            parsed = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc
        if strictly_positive and parsed <= minimum:
            raise ValueError(f"{label} must be greater than {minimum}.")
        if not strictly_positive and parsed < minimum:
            raise ValueError(f"{label} must be at least {minimum}.")
        return parsed

    def _config_from_ui(self, show_errors: bool) -> AppConfig | None:
        try:
            conv_layers = [
                ConvLayerConfig(
                    out_channels=self._parse_int(values["out_channels"].get(), f"Conv {index} kernels"),
                    kernel_size=self._parse_int(values["kernel_size"].get(), f"Conv {index} kernel size"),
                    stride=self._parse_int(values["stride"].get(), f"Conv {index} stride"),
                    pool_size=self._parse_int(values["pool_size"].get(), f"Conv {index} pool size"),
                )
                for index, values in enumerate(self.conv_layer_vars, start=1)
            ]
            dense_layers = [
                DenseLayerConfig(units=self._parse_int(values["units"].get(), f"Dense {index} units"))
                for index, values in enumerate(self.dense_layer_vars, start=1)
            ]
            seed = self._parse_int(self.seed_var.get(), "Seed", minimum=0, allow_empty=True)
            config = AppConfig(
                data_dir=self.data_dir_var.get().strip(),
                epochs=self._parse_int(self.epochs_var.get(), "Epochs"),
                batch_size=self._parse_int(self.batch_size_var.get(), "Batch size"),
                learning_rate=self._parse_float(
                    self.learning_rate_var.get(),
                    "Learning rate",
                    minimum=0.0,
                    strictly_positive=True,
                ),
                weight_decay=self._parse_float(self.weight_decay_var.get(), "Weight decay", minimum=0.0),
                activation=self.activation_var.get().strip().lower(),
                conv_layers=conv_layers,
                dense_layers=dense_layers,
                seed=seed,
            )
        except ValueError as exc:
            if show_errors:
                messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            return None

        errors = validate_config(config)
        if errors:
            if show_errors:
                messagebox.showerror("Invalid architecture", "\n".join(errors), parent=self.root)
            return None
        return config

    def _active_visual_config(self) -> AppConfig | None:
        if self.engine is not None:
            return self.engine.config
        return self._config_from_ui(show_errors=False)

    def _initial_dataset_check(self) -> None:
        data_dir = Path(self.data_dir_var.get().strip())
        if not data_dir:
            return
        if dataset_present(data_dir):
            self._update_dataset_status()
            return
        found = ensure_mnist_dataset(self.root, data_dir, allow_prompt=True)
        self._update_dataset_status()
        if found:
            self.status_var.set("MNIST is ready. Press Start / Restart to begin training.")
        else:
            self.status_var.set("MNIST is missing. Start / Restart will ask to download it again.")

    def _update_dataset_status(self) -> None:
        data_dir = Path(self.data_dir_var.get().strip() or default_config().data_dir)
        try:
            present = dataset_present(data_dir)
        except Exception:
            present = False
        if present:
            self.dataset_var.set(f"Dataset: MNIST ready in {data_dir}")
        else:
            self.dataset_var.set(f"Dataset: MNIST missing in {data_dir} | the GUI can download it")

    def _start_training(self) -> None:
        config = self._config_from_ui(show_errors=True)
        if config is None:
            return

        data_dir = Path(config.data_dir)
        if not ensure_mnist_dataset(self.root, data_dir, allow_prompt=True):
            self._update_dataset_status()
            self.status_var.set("Training did not start because MNIST is still missing.")
            return

        self._stop_training(silent=True)
        self._update_dataset_status()

        try:
            self.engine = MnistTrainingEngine(config, data_dir)
        except Exception as exc:
            messagebox.showerror("Could not start training", str(exc), parent=self.root)
            self.engine = None
            self.status_var.set("Could not start training.")
            return

        self.metric_history = []
        self.current_sample_index = 0
        self.sample_index_var.set("0")
        self.current_snapshot = None
        self.engine.start()
        self._refresh_snapshot()
        self.status_var.set("Training is running. Pause to browse individual test samples more deliberately.")
        self._update_button_states()
        self._redraw_network()

    def _pause_training(self) -> None:
        if self.engine is None:
            return
        self.engine.pause()
        self.status_var.set("Training paused. Use Previous / Next or Random to inspect test samples.")
        self._refresh_snapshot()
        self._update_button_states()

    def _resume_training(self) -> None:
        if self.engine is None:
            return
        self.engine.resume()
        self.status_var.set("Training resumed.")
        self._update_button_states()

    def _stop_training(self, silent: bool = False) -> None:
        if self.engine is not None:
            self.engine.stop()
        if not silent:
            self.status_var.set("Training stopped. Edit the settings, then restart when ready.")
        self._update_button_states()

    def _test_sample_count(self) -> int:
        if self.engine is not None:
            return len(self.engine.test_dataset)
        return 10000

    def _jump_to_sample(self) -> None:
        try:
            index = self._parse_int(self.sample_index_var.get(), "Sample index", minimum=0)
        except ValueError as exc:
            messagebox.showerror("Invalid sample index", str(exc), parent=self.root)
            return
        count = self._test_sample_count()
        self.current_sample_index = min(index, count - 1)
        self.sample_index_var.set(str(self.current_sample_index))
        self._refresh_snapshot()

    def _step_sample(self, delta: int) -> None:
        count = self._test_sample_count()
        self.current_sample_index = (self.current_sample_index + delta) % count
        self.sample_index_var.set(str(self.current_sample_index))
        self._refresh_snapshot()

    def _random_sample(self) -> None:
        count = self._test_sample_count()
        self.current_sample_index = int(np.random.default_rng().integers(0, count))
        self.sample_index_var.set(str(self.current_sample_index))
        self._refresh_snapshot()

    def _refresh_snapshot(self) -> None:
        if self.engine is None:
            self.current_snapshot = None
            self.sample_var.set(f"Sample {self.current_sample_index} | Prediction -- | Target --")
            self._redraw_network()
            return
        try:
            snapshot = self.engine.inspect_sample(self.current_sample_index)
        except Exception as exc:
            self.status_var.set(f"Could not inspect sample: {exc}")
            return
        self.current_snapshot = snapshot
        confidence = float(np.max(snapshot.probabilities) * 100.0)
        self.sample_var.set(
            f"Sample {self.current_sample_index} | Prediction {snapshot.prediction} "
            f"({confidence:.1f}%) | Target {snapshot.target}"
        )
        self._redraw_network()

    def _poll_engine(self) -> None:
        if self.engine is not None:
            updated = False
            while True:
                try:
                    metric = self.engine.metric_queue.get_nowait()
                except queue.Empty:
                    break
                updated = True
                self.metric_history = list(self.engine.history)
                self._set_progress_text(metric)
                if metric.status == "complete":
                    self.status_var.set("Training complete. Pause-like browsing is now active for the final model.")
                elif metric.status == "stopped":
                    self.status_var.set("Training stopped.")
            if updated:
                self._refresh_snapshot()
            self._update_button_states()
        self.root.after(140, self._poll_engine)

    def _set_progress_text(self, metric: MetricPoint) -> None:
        total_batches = len(self.engine.train_loader) if self.engine is not None else 0
        test_acc_text = "--" if metric.test_acc is None else f"{metric.test_acc * 100.0:.1f}%"
        test_loss_text = "--" if metric.test_loss is None else f"{metric.test_loss:.4f}"
        self.progress_var.set(
            f"Epoch {metric.epoch}/{self.engine.config.epochs if self.engine else 0} | "
            f"Batch {metric.batch}/{total_batches} | "
            f"Train acc {metric.train_acc * 100.0:.1f}% | "
            f"Test acc {test_acc_text} | "
            f"Train loss {metric.train_loss:.4f} | Test loss {test_loss_text}"
        )

    def _update_button_states(self) -> None:
        status = "idle" if self.engine is None else self.engine.status
        if status == "running":
            self.pause_button.state(["!disabled"])
            self.resume_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
        elif status == "paused":
            self.pause_button.state(["disabled"])
            self.resume_button.state(["!disabled"])
            self.stop_button.state(["!disabled"])
        elif status in {"complete", "stopped"}:
            self.pause_button.state(["disabled"])
            self.resume_button.state(["disabled"])
            self.stop_button.state(["disabled"])
        else:
            self.pause_button.state(["disabled"])
            self.resume_button.state(["disabled"])
            self.stop_button.state(["disabled"])

        state_flag = "!disabled" if self.engine is not None else "disabled"
        for button in (self.prev_button, self.next_button, self.random_button):
            button.state([state_flag])

    def _schedule_redraw_network(self) -> None:
        if self.redraw_after_id is not None:
            self.root.after_cancel(self.redraw_after_id)
        self.redraw_after_id = self.root.after(160, self._redraw_network)

    def _zoom_scale(self) -> float:
        return max(0.55, min(1.85, float(self.visual_zoom_var.get()) / 100.0))

    def _scale(self, value: float) -> float:
        return value * self._zoom_scale()

    def _ui_font(self, size: float, *styles: str) -> tuple:
        scaled_size = max(7, int(round(size * self._zoom_scale())))
        return ("Segoe UI", scaled_size, *styles)

    def _mono_font(self, size: float, *styles: str) -> tuple:
        scaled_size = max(7, int(round(size * self._zoom_scale())))
        return ("Consolas", scaled_size, *styles)

    def _line_width(self, value: float) -> float:
        return max(1.0, value * self._zoom_scale())

    def _update_zoom_label(self) -> None:
        self.zoom_label_var.set(f"{int(round(float(self.visual_zoom_var.get())))}%")

    def _on_zoom_slider(self, _value=None) -> None:
        self._update_zoom_label()
        self._schedule_redraw_network()

    def _change_zoom(self, delta_percent: float) -> None:
        updated = max(55.0, min(185.0, float(self.visual_zoom_var.get()) + delta_percent))
        self.visual_zoom_var.set(updated)
        self._on_zoom_slider()

    def _on_canvas_zoom(self, event) -> str:
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self._change_zoom(10.0)
        else:
            self._change_zoom(-10.0)
        return "break"

    def _fit_visualizer(self) -> None:
        self.root.update_idletasks()
        region_text = self.network_canvas.cget("scrollregion")
        if not region_text:
            self._redraw_network()
            region_text = self.network_canvas.cget("scrollregion")
        if not region_text:
            return

        x0, y0, x1, y1 = (float(value) for value in region_text.split())
        content_width = max(1.0, x1 - x0)
        content_height = max(1.0, y1 - y0)
        viewport_width = max(240.0, float(self.network_canvas.winfo_width()) - 20.0)
        viewport_height = max(240.0, float(self.network_canvas.winfo_height()) - 20.0)
        fit_factor = min(viewport_width / content_width, viewport_height / content_height)
        target_zoom = max(55.0, min(185.0, float(self.visual_zoom_var.get()) * fit_factor))
        self.visual_zoom_var.set(target_zoom)
        self._update_zoom_label()
        self._redraw_network()
        self.network_canvas.xview_moveto(0.0)
        self.network_canvas.yview_moveto(0.0)

    def _draw_matrix(
        self,
        canvas: tk.Canvas,
        matrix: np.ndarray,
        x: float,
        y: float,
        cell: float,
        palette: str,
        title: str | None = None,
    ) -> tuple[float, float]:
        if title:
            canvas.create_text(
                x,
                y - self._scale(18.0),
                text=title,
                fill="#d8dde8",
                font=self._ui_font(11, "bold"),
                anchor="w",
            )
        rows, cols = matrix.shape
        if palette == "kernel":
            scale = float(np.max(np.abs(matrix))) if matrix.size else 1.0
        elif palette == "activation":
            scale = float(np.max(matrix)) if matrix.size else 1.0
        else:
            scale = 1.0
        for row in range(rows):
            for col in range(cols):
                value = float(matrix[row, col])
                if palette == "kernel":
                    fill = kernel_color(value, scale)
                elif palette == "activation":
                    fill = activation_color(value, scale)
                else:
                    fill = image_color(value)
                x0 = x + col * cell
                y0 = y + row * cell
                canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + cell,
                    y0 + cell,
                    fill=fill,
                    outline="#555e6f",
                    width=self._line_width(1),
                )
        return x + cols * cell, y + rows * cell

    def _draw_input_panel(self, canvas: tk.Canvas, x: float, y: float, snapshot: InspectionSnapshot | None) -> float:
        canvas.create_text(x, y, text="Input Digit", fill="#f4f5f7", font=self._ui_font(16, "bold"), anchor="nw")
        canvas.create_text(
            x,
            y + self._scale(28.0),
            text="28 x 28 grayscale test sample",
            fill="#94a3ba",
            font=self._ui_font(11),
            anchor="nw",
        )
        matrix = np.zeros((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), dtype=float)
        if snapshot is not None:
            matrix = snapshot.image
        self._draw_matrix(canvas, matrix, x, y + self._scale(60.0), self._scale(11.0), "image")
        if snapshot is not None:
            color = "#68df83" if snapshot.prediction == snapshot.target else "#ff8a7a"
            canvas.create_text(
                x,
                y + self._scale(388.0),
                text=f"Prediction {snapshot.prediction} | Target {snapshot.target}",
                fill=color,
                font=self._ui_font(14, "bold"),
                anchor="nw",
            )
        else:
            canvas.create_text(
                x,
                y + self._scale(388.0),
                text="Start training to inspect predictions.",
                fill="#b6c0cf",
                font=self._ui_font(13),
                anchor="nw",
            )
        return y + self._scale(430.0)

    def _input_panel_width(self) -> float:
        return self._scale(336.0)

    def _select_feature_map(self, activations: np.ndarray | None) -> np.ndarray:
        if activations is not None and activations.size:
            strengths = np.mean(np.abs(activations), axis=(1, 2))
            return activations[int(np.argmax(strengths))]
        return np.zeros((12, 12), dtype=float)

    def _conv_panel_size(self, config: ConvLayerConfig, activations: np.ndarray | None) -> tuple[float, float]:
        zoom = self._zoom_scale()
        preview_count = min(config.out_channels, 8)
        columns = 2 if preview_count > 3 else 1
        kernel_size = config.kernel_size
        cell = max(10.0, min(22.0, 120.0 / max(kernel_size, 1))) * zoom
        gap = 18.0 * zoom
        block_width = kernel_size * cell
        block_height = kernel_size * cell
        grid_rows = max(1, (preview_count + columns - 1) // columns)
        grid_width = columns * block_width + max(0, columns - 1) * gap
        grid_height = grid_rows * block_height + max(0, grid_rows - 1) * 30.0 * zoom

        feature_map = self._select_feature_map(activations)
        feature_cell = max(5.0, min(10.0, 112.0 / max(feature_map.shape[0], feature_map.shape[1]))) * zoom
        feature_width = feature_map.shape[1] * feature_cell
        feature_height = feature_map.shape[0] * feature_cell

        panel_width = max(248.0 * zoom, grid_width + 32.0 * zoom, feature_width + 40.0 * zoom)
        panel_height = max(392.0 * zoom, 120.0 * zoom + grid_height + 44.0 * zoom + feature_height + 50.0 * zoom)
        return panel_width, panel_height

    def _dense_panel_width(self, config: DenseLayerConfig) -> float:
        base_width = 188.0 if config.units > 9 else 176.0
        return self._scale(base_width)

    def _layout_gap_and_start(self, viewport_width: float, section_widths: list[float]) -> tuple[float, float]:
        zoom = self._zoom_scale()
        margin = 44.0 * zoom
        base_gap = 28.0 * zoom
        usable_width = max(980.0 * zoom, viewport_width - margin * 2.0)
        if len(section_widths) <= 1:
            return base_gap, margin
        base_content_width = sum(section_widths) + base_gap * (len(section_widths) - 1)
        if usable_width <= base_content_width:
            return base_gap, margin

        extra_space = usable_width - base_content_width
        additional_gap = min(76.0 * zoom, extra_space / max(len(section_widths) - 1, 1))
        gap = base_gap + additional_gap
        content_width = sum(section_widths) + gap * (len(section_widths) - 1)
        start_x = margin + max(0.0, (usable_width - content_width) / 2.0)
        return gap, start_x

    def _draw_conv_layer(
        self,
        canvas: tk.Canvas,
        x: float,
        y: float,
        index: int,
        config: ConvLayerConfig,
        kernels: np.ndarray | None,
        activations: np.ndarray | None,
    ) -> tuple[float, float]:
        zoom = self._zoom_scale()
        panel_width, panel_height = self._conv_panel_size(config, activations)
        canvas.create_rectangle(
            x,
            y,
            x + panel_width,
            y + panel_height,
            fill="#0d1119",
            outline="#2b3547",
            width=self._line_width(2),
        )
        canvas.create_text(
            x + self._scale(14.0),
            y + self._scale(14.0),
            text=f"Conv {index + 1}",
            fill="#f6f7f9",
            font=self._ui_font(16, "bold"),
            anchor="nw",
        )
        canvas.create_text(
            x + self._scale(14.0),
            y + self._scale(44.0),
            text=(
                f"{config.out_channels} kernels | size {config.kernel_size} | "
                f"stride {config.stride} | pool {config.pool_size}"
            ),
            fill="#94a3ba",
            font=self._ui_font(11),
            anchor="nw",
        )

        preview_count = min(config.out_channels, 8)
        columns = 2 if preview_count > 3 else 1
        kernel_size = config.kernel_size
        cell = max(10.0, min(22.0, 120.0 / max(kernel_size, 1))) * zoom
        gap = 18.0 * zoom
        block_width = kernel_size * cell
        block_height = kernel_size * cell
        grid_rows = max(1, (preview_count + columns - 1) // columns)
        grid_height = grid_rows * block_height + max(0, grid_rows - 1) * 30.0 * zoom
        start_y = y + self._scale(82.0)
        for preview_index in range(preview_count):
            row = preview_index // columns
            col = preview_index % columns
            kernel_x = x + self._scale(16.0) + col * (block_width + gap)
            kernel_y = start_y + row * (block_height + self._scale(30.0))
            matrix = np.zeros((kernel_size, kernel_size), dtype=float)
            if kernels is not None and preview_index < kernels.shape[0]:
                matrix = kernels[preview_index]
            self._draw_matrix(
                canvas,
                matrix,
                kernel_x,
                kernel_y,
                cell,
                "kernel",
                title=f"K{preview_index + 1}",
            )

        if config.out_channels > preview_count:
            extra_y = min(y + panel_height - self._scale(74.0), start_y + grid_height + self._scale(6.0))
            canvas.create_text(
                x + self._scale(16.0),
                extra_y,
                text=f"+ {config.out_channels - preview_count} more kernels",
                fill="#7f90ac",
                font=self._ui_font(10),
                anchor="nw",
            )

        act_matrix = self._select_feature_map(activations)
        feature_cell = max(5.0, min(10.0, 112.0 / max(act_matrix.shape[0], act_matrix.shape[1]))) * zoom
        feature_width = act_matrix.shape[1] * feature_cell
        feature_x = x + max(self._scale(16.0), (panel_width - feature_width) / 2.0)
        feature_y = start_y + grid_height + self._scale(44.0)
        canvas.create_text(
            x + self._scale(16.0),
            feature_y - self._scale(26.0),
            text="Feature Map",
            fill="#d7e0ed",
            font=self._ui_font(12, "bold"),
            anchor="nw",
        )
        self._draw_matrix(canvas, act_matrix, feature_x, feature_y, feature_cell, "activation")
        canvas.create_text(
            x + self._scale(16.0),
            feature_y + act_matrix.shape[0] * feature_cell + self._scale(12.0),
            text=f"response size {act_matrix.shape[1]} x {act_matrix.shape[0]}",
            fill="#7f90ac",
            font=self._ui_font(10),
            anchor="nw",
        )
        return x + panel_width, y + panel_height

    def _draw_dense_layer(
        self,
        canvas: tk.Canvas,
        x: float,
        y: float,
        index: int,
        config: DenseLayerConfig,
        activations: np.ndarray | None,
    ) -> tuple[float, float]:
        zoom = self._zoom_scale()
        visible = min(config.units, 12)
        spacing = self._scale(28.0)
        radius = self._scale(11.0)
        column_height = max(visible * spacing + self._scale(40.0), self._scale(340.0))
        panel_width = self._dense_panel_width(config)
        canvas.create_rectangle(
            x,
            y,
            x + panel_width,
            y + column_height,
            fill="#0d1119",
            outline="#2b3547",
            width=self._line_width(2),
        )
        canvas.create_text(
            x + self._scale(14.0),
            y + self._scale(14.0),
            text=f"Dense {index + 1}",
            fill="#f6f7f9",
            font=self._ui_font(16, "bold"),
            anchor="nw",
        )
        canvas.create_text(
            x + self._scale(14.0),
            y + self._scale(42.0),
            text=f"{config.units} neurons",
            fill="#94a3ba",
            font=self._ui_font(11),
            anchor="nw",
        )
        values = np.zeros(visible, dtype=float)
        if activations is not None and activations.size:
            values = activations[:visible]
        scale = float(np.max(np.abs(values))) if values.size else 1.0
        start_y = y + self._scale(76.0)
        for node_index in range(visible):
            value = float(values[node_index]) if node_index < values.shape[0] else 0.0
            intensity = 0.0 if scale <= 1e-8 else min(1.0, abs(value) / scale)
            fill = blend_color("#16212f", "#67d8ff" if value >= 0.0 else "#4c86ff", intensity)
            cy = start_y + node_index * spacing
            canvas.create_oval(
                x + self._scale(22.0),
                cy,
                x + self._scale(22.0) + radius * 2,
                cy + radius * 2,
                fill=fill,
                outline="#dce4ef",
                width=self._line_width(1.5),
            )
            canvas.create_text(
                x + self._scale(56.0),
                cy + radius,
                text=f"{value:+.2f}",
                fill="#e8edf6",
                font=self._mono_font(10),
                anchor="w",
            )
        if config.units > visible:
            canvas.create_text(
                x + self._scale(14.0),
                y + column_height - self._scale(24.0),
                text=f"+ {config.units - visible} more neurons",
                fill="#7f90ac",
                font=self._ui_font(10),
                anchor="nw",
            )
        return x + panel_width, y + column_height

    def _draw_output_layer(
        self,
        canvas: tk.Canvas,
        x: float,
        y: float,
        snapshot: InspectionSnapshot | None,
    ) -> tuple[float, float]:
        panel_width = self._scale(260.0)
        panel_height = self._scale(430.0)
        canvas.create_rectangle(
            x,
            y,
            x + panel_width,
            y + panel_height,
            fill="#0d1119",
            outline="#2b3547",
            width=self._line_width(2),
        )
        canvas.create_text(
            x + self._scale(14.0),
            y + self._scale(14.0),
            text="Output Digits",
            fill="#f6f7f9",
            font=self._ui_font(16, "bold"),
            anchor="nw",
        )
        canvas.create_text(
            x + self._scale(14.0),
            y + self._scale(42.0),
            text="Fixed 10-way classifier",
            fill="#94a3ba",
            font=self._ui_font(11),
            anchor="nw",
        )

        probabilities = np.zeros(OUTPUT_CLASSES, dtype=float)
        target = None
        prediction = None
        if snapshot is not None:
            probabilities = snapshot.probabilities
            target = snapshot.target
            prediction = snapshot.prediction
        bar_x = x + self._scale(24.0)
        base_y = y + self._scale(350.0)
        bar_spacing = self._scale(22.0)
        bar_width = self._scale(16.0)
        bar_height_scale = self._scale(250.0)
        for digit in range(OUTPUT_CLASSES):
            bar_height = bar_height_scale * float(probabilities[digit])
            fill = "#49d971" if prediction == digit and prediction == target else "#ff8a7a" if prediction == digit else "#4c86ff"
            canvas.create_rectangle(
                bar_x + digit * bar_spacing,
                base_y - bar_height,
                bar_x + digit * bar_spacing + bar_width,
                base_y,
                fill=fill,
                outline="",
            )
            canvas.create_text(
                bar_x + digit * bar_spacing + bar_width * 0.5,
                base_y + self._scale(18.0),
                text=str(digit),
                fill="#dde5f1",
                font=self._ui_font(10, "bold"),
            )
            canvas.create_text(
                bar_x + digit * bar_spacing + bar_width * 0.5,
                base_y - bar_height - self._scale(12.0),
                text=f"{probabilities[digit] * 100.0:.0f}",
                fill="#c7d3e3",
                font=self._ui_font(8),
            )
        if snapshot is not None:
            color = "#68df83" if prediction == target else "#ff8a7a"
            canvas.create_text(
                x + self._scale(14.0),
                y + self._scale(382.0),
                text=f"Predicted {prediction} | Target {target}",
                fill=color,
                font=self._ui_font(14, "bold"),
                anchor="nw",
            )
        else:
            canvas.create_text(
                x + self._scale(14.0),
                y + self._scale(382.0),
                text="Prediction bars appear after the model is initialized.",
                fill="#b6c0cf",
                font=self._ui_font(12),
                anchor="nw",
            )
        return x + panel_width, y + panel_height

    def _draw_metric_chart(self, canvas: tk.Canvas, x: float, y: float, width: float) -> float:
        chart_height = self._scale(210.0)
        canvas.create_rectangle(
            x,
            y,
            x + width,
            y + chart_height,
            fill="#0d1119",
            outline="#2b3547",
            width=self._line_width(2),
        )
        canvas.create_text(
            x + self._scale(14.0),
            y + self._scale(14.0),
            text="Training Progress",
            fill="#f6f7f9",
            font=self._ui_font(16, "bold"),
            anchor="nw",
        )
        if not self.metric_history:
            canvas.create_text(
                x + self._scale(14.0),
                y + self._scale(50.0),
                text="Epoch summaries appear here as training completes each epoch.",
                fill="#a3afc0",
                font=self._ui_font(12),
                anchor="nw",
            )
            return y + chart_height

        left = x + self._scale(54.0)
        top = y + self._scale(44.0)
        right = x + width - self._scale(16.0)
        bottom = y + chart_height - self._scale(34.0)
        canvas.create_rectangle(left, top, right, bottom, outline="#556274", width=self._line_width(1))
        for tick in range(6):
            line_y = top + (bottom - top) * tick / 5.0
            canvas.create_line(left, line_y, right, line_y, fill="#1f2937")
            canvas.create_text(
                left - self._scale(10.0),
                line_y,
                text=f"{100 - tick * 20}",
                fill="#93a4ba",
                font=self._ui_font(9),
                anchor="e",
            )

        epochs = [point.epoch for point in self.metric_history]
        train_acc = [point.train_acc * 100.0 for point in self.metric_history]
        test_acc = [0.0 if point.test_acc is None else point.test_acc * 100.0 for point in self.metric_history]
        max_epoch = max(epochs)
        if max_epoch == 1:
            x_positions = [left]
        else:
            x_positions = [left + (right - left) * (epoch - 1) / (max_epoch - 1) for epoch in epochs]

        def draw_series(values: list[float], color: str) -> None:
            points: list[float] = []
            for x_pos, value in zip(x_positions, values):
                y_pos = bottom - (bottom - top) * max(0.0, min(100.0, value)) / 100.0
                points.extend([x_pos, y_pos])
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=self._line_width(3), smooth=True)

        draw_series(train_acc, "#49d9ff")
        draw_series(test_acc, "#49d971")

        legend_y = y + self._scale(182.0)
        canvas.create_line(
            x + self._scale(22.0),
            legend_y,
            x + self._scale(46.0),
            legend_y,
            fill="#49d9ff",
            width=self._line_width(3),
        )
        canvas.create_text(
            x + self._scale(52.0),
            legend_y,
            text="Train accuracy",
            fill="#d9e2ef",
            font=self._ui_font(10),
            anchor="w",
        )
        canvas.create_line(
            x + self._scale(166.0),
            legend_y,
            x + self._scale(190.0),
            legend_y,
            fill="#49d971",
            width=self._line_width(3),
        )
        canvas.create_text(
            x + self._scale(196.0),
            legend_y,
            text="Test accuracy",
            fill="#d9e2ef",
            font=self._ui_font(10),
            anchor="w",
        )
        return y + chart_height

    def _redraw_network(self) -> None:
        self.redraw_after_id = None
        canvas = self.network_canvas
        canvas.delete("all")
        viewport_width = max(float(canvas.winfo_width()), 760.0)
        viewport_height = max(float(canvas.winfo_height()), 560.0)
        canvas.create_rectangle(
            0,
            0,
            max(viewport_width + self._scale(600.0), self._scale(1800.0)),
            max(viewport_height + self._scale(500.0), self._scale(1400.0)),
            fill="#05070b",
            outline="",
        )

        active_config = self._active_visual_config()
        if active_config is None:
            canvas.create_text(
                self._scale(50.0),
                self._scale(60.0),
                text="Enter valid layer settings to preview the network.",
                fill="#e6ebf2",
                font=self._ui_font(20, "bold"),
                anchor="nw",
            )
            canvas.configure(scrollregion=(0, 0, max(viewport_width, self._scale(1200.0)), max(viewport_height, self._scale(900.0))))
            return

        snapshot = self.current_snapshot
        canvas.create_text(
            self._scale(44.0),
            self._scale(26.0),
            text="Yellow weights are positive, blue weights are negative. Bright activations show strong responses.",
            fill="#d9e0ee",
            font=self._ui_font(12),
            anchor="nw",
        )

        section_widths: list[float] = [self._input_panel_width()]
        conv_activations = snapshot.conv_activations if snapshot is not None else []
        dense_activations = snapshot.dense_activations if snapshot is not None else []
        for index, layer in enumerate(active_config.conv_layers):
            activations = conv_activations[index] if index < len(conv_activations) else None
            section_widths.append(self._conv_panel_size(layer, activations)[0])
        if active_config.conv_layers:
            section_widths.append(54.0)
        for layer in active_config.dense_layers:
            section_widths.append(self._dense_panel_width(layer))
        section_widths.append(self._scale(260.0))

        gap, x = self._layout_gap_and_start(viewport_width, section_widths)
        top_y = self._scale(70.0)
        lower_bound = self._draw_input_panel(canvas, x, top_y, snapshot)
        x += self._input_panel_width() + gap

        conv_bottom = lower_bound
        for index, layer in enumerate(active_config.conv_layers):
            kernels = None
            activations = None
            if snapshot is not None and index < len(snapshot.conv_kernels):
                kernels = snapshot.conv_kernels[index]
            if snapshot is not None and index < len(snapshot.conv_activations):
                activations = snapshot.conv_activations[index]
            x, layer_bottom = self._draw_conv_layer(canvas, x, top_y, index, layer, kernels, activations)
            conv_bottom = max(conv_bottom, layer_bottom)
            if index < len(active_config.conv_layers) - 1:
                x += gap

        if active_config.conv_layers:
            canvas.create_text(
                x + self._scale(8.0),
                top_y + self._scale(160.0),
                text="Flatten",
                fill="#d9e0ee",
                font=self._ui_font(14, "bold"),
                anchor="nw",
            )
            canvas.create_rectangle(
                x + self._scale(12.0),
                top_y + self._scale(192.0),
                x + self._scale(40.0),
                top_y + self._scale(242.0),
                fill="#14342d",
                outline="#59c49f",
                width=self._line_width(2),
            )
            x += self._scale(54.0) + gap

        dense_bottom = top_y
        for index, layer in enumerate(active_config.dense_layers):
            activations = None
            if snapshot is not None and index < len(snapshot.dense_activations):
                activations = snapshot.dense_activations[index]
            x, layer_bottom = self._draw_dense_layer(canvas, x, top_y + self._scale(8.0), index, layer, activations)
            dense_bottom = max(dense_bottom, layer_bottom)
            if index < len(active_config.dense_layers) - 1:
                x += gap

        if active_config.dense_layers:
            x += gap

        output_x = x
        _, output_bottom = self._draw_output_layer(canvas, output_x, top_y, snapshot)

        chart_top = max(conv_bottom, dense_bottom, output_bottom) + self._scale(28.0)
        content_right = output_x + self._scale(260.0)
        chart_left = self._scale(44.0)
        chart_width = max(content_right - chart_left, viewport_width - self._scale(88.0))
        chart_bottom = self._draw_metric_chart(canvas, chart_left, chart_top, chart_width)

        if snapshot is None:
            canvas.create_text(
                self._scale(44.0),
                chart_bottom + self._scale(26.0),
                text=(
                    "The right panel is already showing the architecture. "
                    "Press Start / Restart to initialize weights and begin MNIST training."
                ),
                fill="#b9c6d8",
                font=self._ui_font(13),
                anchor="nw",
            )

        scroll_width = max(content_right + self._scale(44.0), viewport_width)
        scroll_height = max(chart_bottom + self._scale(90.0), viewport_height)
        canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))

    def _on_close(self) -> None:
        self._stop_training(silent=True)
        self.root.destroy()


def run_smoke_test_model() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for the smoke test.")
    config = default_config()
    model = MnistCNN(config.conv_layers, config.dense_layers, config.activation)
    sample = torch.randn(1, 1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    logits, conv_activations, dense_activations = model.inspect(sample)
    print(
        {
            "logits_shape": tuple(logits.shape),
            "conv_layers": len(conv_activations),
            "dense_layers": len(dense_activations),
        }
    )


def run_smoke_test_ui() -> None:
    root = tk.Tk()
    app = MnistVisualizerApp(root, default_config(), prompt_for_dataset=False)
    root.update_idletasks()
    app._redraw_network()
    print("gui-ok")
    root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive MNIST CNN kernel visualizer.")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional override for the MNIST data folder.")
    parser.add_argument(
        "--smoke-test-model",
        action="store_true",
        help="Build the CNN and run a tiny tensor through it without starting the GUI.",
    )
    parser.add_argument(
        "--smoke-test-ui",
        action="store_true",
        help="Instantiate the GUI without the startup dataset prompt, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test_model:
        run_smoke_test_model()
        return
    if args.smoke_test_ui:
        run_smoke_test_ui()
        return

    config = default_config()
    if args.data_dir:
        config.data_dir = args.data_dir

    root = tk.Tk()
    MnistVisualizerApp(root, config, prompt_for_dataset=True)
    root.mainloop()


if __name__ == "__main__":
    main()

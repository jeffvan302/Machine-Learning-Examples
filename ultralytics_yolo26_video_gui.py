from __future__ import annotations

import argparse
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

RUNTIME_DIR = Path(__file__).resolve().with_name("ultralytics_runtime")
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_DIR))

from ultralytics import YOLO


DEFAULT_MODELS = {
    "detect": "yolo26n.pt",
    "segment": "yolo26n-seg.pt",
    "pose": "yolo26n-pose.pt",
    "obb": "yolo26n-obb.pt",
    "classify": "yolo26n-cls.pt",
}

DEFAULT_IMAGE_SIZES = {
    "detect": 640,
    "segment": 640,
    "pose": 640,
    "obb": 1024,
    "classify": 224,
}


@dataclass(slots=True)
class DemoConfig:
    source_type: str
    source_value: str
    task: str
    model_name: str
    conf: float
    iou: float
    imgsz: int
    device: str


@dataclass(slots=True)
class FramePacket:
    frame_rgb: np.ndarray
    frame_index: int
    total_frames: int
    fps: float
    inference_ms: float
    detections: int
    class_counts: dict[str, float]
    source_name: str
    task: str


class YoloVideoWorker:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.status = "idle"

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.status = "running"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def pause(self) -> None:
        self.pause_event.set()
        self.status = "paused"

    def resume(self) -> None:
        self.pause_event.clear()
        self.status = "running"

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.status = "stopped"

    def _emit(self, kind: str, payload: object) -> None:
        self.queue.put((kind, payload))

    def _resolve_device(self) -> str | None:
        return None if self.config.device == "auto" else self.config.device

    def _collect_counts(self, result) -> tuple[int, dict[str, float]]:
        counts: dict[str, float] = {}
        names = result.names
        probs = result.probs
        if probs is not None:
            for class_id, score in zip(probs.top5, probs.top5conf.tolist()):
                if isinstance(names, dict):
                    name = str(names.get(class_id, class_id))
                else:
                    name = str(names[class_id])
                counts[name] = float(score)
            return 1, counts

        boxes_like = result.obb if result.obb is not None else result.boxes
        if boxes_like is None or len(boxes_like) == 0:
            return 0, counts

        for class_id in boxes_like.cls.int().tolist():
            if isinstance(names, dict):
                name = str(names.get(class_id, class_id))
            else:
                name = str(names[class_id])
            counts[name] = counts.get(name, 0.0) + 1.0
        return int(sum(counts.values())), counts

    def _annotation_to_rgb(self, annotated) -> np.ndarray:
        if isinstance(annotated, Image.Image):
            return np.asarray(annotated.convert("RGB"))
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    def _run(self) -> None:
        capture = None
        try:
            self._emit("status", f"Loading model {self.config.model_name} ...")
            model = YOLO(self.config.model_name)

            if self.config.source_type == "camera":
                capture = cv2.VideoCapture(int(self.config.source_value))
                source_name = f"Camera {self.config.source_value}"
            else:
                capture = cv2.VideoCapture(self.config.source_value)
                source_name = Path(self.config.source_value).name
            if not capture.isOpened():
                raise RuntimeError(f"Could not open {self.config.source_type}: {self.config.source_value}")

            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 1e-6:
                fps = 30.0
            total_frames = 0 if self.config.source_type == "camera" else int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            device = self._resolve_device()

            self._emit("status", f"Running {self.config.task} inference on {source_name}")
            frame_index = 0
            frame_interval = 1.0 / max(fps, 1.0)

            while not self.stop_event.is_set():
                while self.pause_event.is_set():
                    if self.stop_event.wait(0.05):
                        return

                loop_start = time.perf_counter()
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                frame_index += 1
                infer_start = time.perf_counter()
                results = model.predict(
                    source=frame_bgr,
                    conf=self.config.conf,
                    iou=self.config.iou,
                    imgsz=self.config.imgsz,
                    device=device,
                    verbose=False,
                )
                inference_ms = (time.perf_counter() - infer_start) * 1000.0

                result = results[0]
                detections, counts = self._collect_counts(result)
                annotated = result.plot()
                annotated_rgb = self._annotation_to_rgb(annotated)

                self._emit(
                    "frame",
                    FramePacket(
                        frame_rgb=annotated_rgb,
                        frame_index=frame_index,
                        total_frames=total_frames,
                        fps=fps,
                        inference_ms=inference_ms,
                        detections=detections,
                        class_counts=counts,
                        source_name=source_name,
                        task=self.config.task,
                    ),
                )

                remaining = frame_interval - (time.perf_counter() - loop_start)
                if remaining > 0.0:
                    time.sleep(remaining)

            if not self.stop_event.is_set():
                self.status = "complete"
                if self.config.source_type == "camera":
                    self._emit("complete", f"The {source_name} stream stopped.")
                else:
                    self._emit("complete", f"Reached the end of {source_name}.")
        except Exception as exc:
            self.status = "error"
            self._emit("error", str(exc))
        finally:
            if capture is not None:
                capture.release()


class YoloVideoDemoApp:
    def __init__(self, root: tk.Tk, prompt_for_video: bool = False) -> None:
        self.root = root
        self.prompt_for_video = prompt_for_video
        self.worker: YoloVideoWorker | None = None
        self.last_frame_rgb: np.ndarray | None = None
        self.current_image: ImageTk.PhotoImage | None = None
        self.last_task = "detect"

        self.root.title("Ultralytics YOLO26 Video Demo")
        self.root.geometry("1640x940")
        self.root.minsize(1280, 760)
        self.root.configure(bg="#e8e4d7")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.source_type_var = tk.StringVar(value="video")
        self.video_path_var = tk.StringVar(value="")
        self.camera_index_var = tk.StringVar(value="0")
        self.task_var = tk.StringVar(value="detect")
        self.model_var = tk.StringVar(value=DEFAULT_MODELS["detect"])
        self.conf_var = tk.StringVar(value="0.35")
        self.iou_var = tk.StringVar(value="0.45")
        self.imgsz_var = tk.StringVar(value=str(DEFAULT_IMAGE_SIZES["detect"]))
        self.device_var = tk.StringVar(value="auto")

        self.status_var = tk.StringVar(
            value="Choose a video file or switch to Camera, then run YOLO26 detection, segmentation, pose, OBB, or classification."
        )
        self.progress_var = tk.StringVar(value="Frame -- | Detections -- | Inference --")
        self.counts_var = tk.StringVar(value="Classes: --")

        self._build_ui()
        self._update_button_states()
        self._poll_worker()
        self.task_var.trace_add("write", self._on_task_changed)
        self.source_type_var.trace_add("write", self._on_source_changed)
        self.video_path_var.trace_add("write", self._on_source_changed)
        self.camera_index_var.trace_add("write", self._on_source_changed)
        self._sync_source_controls()

    def _build_ui(self) -> None:
        panes = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(panes, padding=16)
        panes.add(left, weight=0)
        right = ttk.Frame(panes, padding=(8, 12, 12, 12))
        panes.add(right, weight=1)

        self._build_sidebar(left)
        self._build_video_area(right)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="YOLO26 Demo",
            font=("Segoe UI Semibold", 20),
            foreground="#173746",
        ).pack(anchor="w", pady=(0, 12))

        source_frame = ttk.LabelFrame(parent, text="Source")
        source_frame.pack(fill=tk.X, pady=(0, 12))
        self._labeled_combo(source_frame, "Source", self.source_type_var, ("video", "camera"))

        video_row = ttk.Frame(source_frame)
        video_row.pack(fill=tk.X, padx=10, pady=(6, 4))
        ttk.Label(video_row, text="Video file", width=12).pack(side=tk.LEFT)
        self.video_entry = ttk.Entry(video_row, textvariable=self.video_path_var, width=24)
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.video_button = ttk.Button(video_row, text="Choose...", command=self._browse_video)
        self.video_button.pack(side=tk.LEFT, padx=(6, 0))

        camera_row = ttk.Frame(source_frame)
        camera_row.pack(fill=tk.X, padx=10, pady=(4, 10))
        ttk.Label(camera_row, text="Camera", width=12).pack(side=tk.LEFT)
        self.camera_entry = ttk.Entry(camera_row, textvariable=self.camera_index_var, width=8)
        self.camera_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        settings = ttk.LabelFrame(parent, text="Inference")
        settings.pack(fill=tk.X, pady=(0, 12))
        self._labeled_combo(settings, "Task", self.task_var, ("detect", "segment", "pose", "obb", "classify"))
        self._labeled_entry(settings, "Model", self.model_var)
        self._labeled_entry(settings, "Confidence", self.conf_var)
        self._labeled_entry(settings, "IoU", self.iou_var)
        self._labeled_entry(settings, "Image size", self.imgsz_var)
        self._labeled_combo(settings, "Device", self.device_var, ("auto", "cpu"))

        ttk.Label(
            settings,
            text=(
                "Defaults use the official YOLO26 nano models for detect, segment, pose, oriented boxes, "
                "and frame classification."
            ),
            wraplength=310,
            foreground="#49606b",
        ).pack(anchor="w", padx=10, pady=(4, 10))

        buttons = ttk.LabelFrame(parent, text="Controls")
        buttons.pack(fill=tk.X, pady=(0, 12))
        buttons.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(buttons, text="Start / Restart", command=self._start_demo)
        self.pause_button = ttk.Button(buttons, text="Pause", command=self._pause_demo)
        self.resume_button = ttk.Button(buttons, text="Resume", command=self._resume_demo)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop_demo)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(10, 6), pady=(10, 6))
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 6))
        self.resume_button.grid(row=1, column=0, sticky="ew", padx=(10, 6), pady=(0, 10))
        self.stop_button.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(0, 10))

        info = ttk.LabelFrame(parent, text="Status")
        info.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            info,
            textvariable=self.status_var,
            wraplength=320,
            foreground="#304b57",
            font=("Segoe UI", 11),
        ).pack(anchor="w", padx=10, pady=(10, 8))
        ttk.Label(
            info,
            text=(
                "If the default weights are not already on disk, Ultralytics may download them the first time "
                "you run a model. Start stays disabled until a video file is selected or a camera index is ready."
            ),
            wraplength=320,
            foreground="#5f6d73",
        ).pack(anchor="w", padx=10, pady=(0, 10))

    def _build_video_area(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Detection / Instance Segmentation Output",
            font=("Segoe UI Semibold", 22),
            foreground="#173746",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            textvariable=self.progress_var,
            font=("Segoe UI Semibold", 16),
            foreground="#0d5677",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(
            header,
            textvariable=self.counts_var,
            font=("Segoe UI", 12),
            foreground="#3b5861",
        ).grid(row=2, column=0, sticky="w", pady=(6, 10))

        self.video_canvas = tk.Canvas(parent, bg="#05070b", highlightthickness=0)
        self.video_canvas.grid(row=1, column=0, sticky="nsew")
        self.video_canvas.bind("<Configure>", lambda _event: self._render_last_frame())

    def _labeled_entry(self, parent: ttk.Frame, label_text: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=10, pady=4)
        ttk.Label(row, text=label_text, width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=18).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _labeled_combo(
        self,
        parent: ttk.Frame,
        label_text: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=10, pady=4)
        ttk.Label(row, text=label_text, width=12).pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=16).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
        )

    def _browse_video(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a video file",
            filetypes=(
                ("Video Files", "*.mp4 *.mov *.avi *.mkv *.wmv *.m4v"),
                ("All Files", "*.*"),
            ),
        )
        if selected:
            self.video_path_var.set(selected)

    def _source_ready(self) -> bool:
        source_type = self.source_type_var.get().strip().lower()
        if source_type == "camera":
            try:
                return int(self.camera_index_var.get().strip()) >= 0
            except ValueError:
                return False
        return bool(self.video_path_var.get().strip()) and Path(self.video_path_var.get().strip()).exists()

    def _sync_source_controls(self) -> None:
        source_type = self.source_type_var.get().strip().lower()
        if source_type == "camera":
            self.video_entry.state(["disabled"])
            self.video_button.state(["disabled"])
            self.camera_entry.state(["!disabled"])
        else:
            self.video_entry.state(["!disabled"])
            self.video_button.state(["!disabled"])
            self.camera_entry.state(["disabled"])

    def _on_source_changed(self, *_args) -> None:
        self._sync_source_controls()
        if self.source_type_var.get().strip().lower() == "camera":
            self.status_var.set("Camera mode is selected. Set the camera index, usually 0 for the first camera.")
        elif self.video_path_var.get().strip():
            self.status_var.set(f"Video source ready: {Path(self.video_path_var.get().strip()).name}")
        else:
            self.status_var.set("Choose a video file to enable Start / Restart, or switch the source to Camera.")
        self._update_button_states()

    def _on_task_changed(self, *_args) -> None:
        new_task = self.task_var.get().strip().lower()
        current_model = self.model_var.get().strip()
        current_imgsz = self.imgsz_var.get().strip()
        if current_model == DEFAULT_MODELS.get(self.last_task, "") or not current_model:
            self.model_var.set(DEFAULT_MODELS[new_task])
        if current_imgsz == str(DEFAULT_IMAGE_SIZES.get(self.last_task, 640)) or not current_imgsz:
            self.imgsz_var.set(str(DEFAULT_IMAGE_SIZES[new_task]))
        self.last_task = new_task
        if new_task == "classify":
            self.status_var.set("Classification labels the whole frame. The top classes for each frame are shown below.")
        elif new_task == "obb":
            self.status_var.set("OBB uses oriented bounding boxes for rotated objects when the model supports them.")
        elif new_task == "pose":
            self.status_var.set("Pose mode draws keypoints and skeletons where the model detects them.")
        elif new_task == "segment":
            self.status_var.set("Segment mode draws instance masks plus object boxes.")
        else:
            self.status_var.set("Detect mode draws standard object boxes on the chosen video.")

    def _parse_float(self, value: str, label: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            messagebox.showerror("Invalid value", f"{label} must be a number.", parent=self.root)
            return None

    def _parse_int(self, value: str, label: str) -> int | None:
        try:
            return int(value)
        except ValueError:
            messagebox.showerror("Invalid value", f"{label} must be an integer.", parent=self.root)
            return None

    def _config_from_ui(self) -> DemoConfig | None:
        source_type = self.source_type_var.get().strip().lower()
        if source_type == "camera":
            camera_index = self._parse_int(self.camera_index_var.get().strip(), "Camera index")
            if camera_index is None:
                return None
            if camera_index < 0:
                messagebox.showerror("Invalid value", "Camera index must be zero or higher.", parent=self.root)
                return None
            source_value = str(camera_index)
        else:
            video_path = self.video_path_var.get().strip()
            if not video_path:
                self.status_var.set("Choose a video file to enable Start / Restart.")
                return None
            if not Path(video_path).exists():
                messagebox.showerror("Video not found", f"Could not find:\n{video_path}", parent=self.root)
                return None
            source_value = video_path

        conf = self._parse_float(self.conf_var.get().strip(), "Confidence")
        iou = self._parse_float(self.iou_var.get().strip(), "IoU")
        imgsz = self._parse_int(self.imgsz_var.get().strip(), "Image size")
        if conf is None or iou is None or imgsz is None:
            return None
        if not 0.0 <= conf <= 1.0:
            messagebox.showerror("Invalid value", "Confidence must be between 0 and 1.", parent=self.root)
            return None
        if not 0.0 <= iou <= 1.0:
            messagebox.showerror("Invalid value", "IoU must be between 0 and 1.", parent=self.root)
            return None
        if imgsz <= 0:
            messagebox.showerror("Invalid value", "Image size must be greater than zero.", parent=self.root)
            return None

        task = self.task_var.get().strip().lower()
        model_name = self.model_var.get().strip() or DEFAULT_MODELS[task]
        return DemoConfig(
            source_type=source_type,
            source_value=source_value,
            task=task,
            model_name=model_name,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device_var.get().strip().lower(),
        )

    def _start_demo(self) -> None:
        config = self._config_from_ui()
        if config is None:
            return
        self._stop_demo(silent=True)
        self.worker = YoloVideoWorker(config)
        if config.source_type == "camera":
            source_label = f"camera {config.source_value}"
        else:
            source_label = Path(config.source_value).name
        self.status_var.set(f"Starting {config.task} with {config.model_name} on {source_label} ...")
        self.progress_var.set("Frame -- | Detections -- | Inference --")
        self.counts_var.set("Classes: --")
        self.last_frame_rgb = None
        self.video_canvas.delete("all")
        self.worker.start()
        self._update_button_states()

    def _pause_demo(self) -> None:
        if self.worker is None:
            return
        self.worker.pause()
        self.status_var.set("Playback paused.")
        self._update_button_states()

    def _resume_demo(self) -> None:
        if self.worker is None:
            return
        self.worker.resume()
        self.status_var.set("Playback resumed.")
        self._update_button_states()

    def _stop_demo(self, silent: bool = False) -> None:
        if self.worker is not None:
            self.worker.stop()
        if not silent:
            self.status_var.set("Playback stopped.")
        self._update_button_states()

    def _update_button_states(self) -> None:
        status = "idle" if self.worker is None else self.worker.status
        source_ready = self._source_ready()
        if status in {"running", "paused"}:
            self.start_button.state(["!disabled"])
        elif source_ready:
            self.start_button.state(["!disabled"])
        else:
            self.start_button.state(["disabled"])
        if status == "running":
            self.pause_button.state(["!disabled"])
            self.resume_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
        elif status == "paused":
            self.pause_button.state(["disabled"])
            self.resume_button.state(["!disabled"])
            self.stop_button.state(["!disabled"])
        else:
            self.pause_button.state(["disabled"])
            self.resume_button.state(["disabled"])
            self.stop_button.state(["disabled"])

    def _poll_worker(self) -> None:
        if self.worker is not None:
            while True:
                try:
                    kind, payload = self.worker.queue.get_nowait()
                except queue.Empty:
                    break

                if kind == "frame":
                    packet = payload
                    assert isinstance(packet, FramePacket)
                    self._handle_frame(packet)
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "complete":
                    self.status_var.set(str(payload))
                    self.worker.status = "complete"
                elif kind == "error":
                    self.status_var.set(str(payload))
                    self.worker.status = "error"
                    messagebox.showerror("Ultralytics inference failed", str(payload), parent=self.root)
                self._update_button_states()
        self.root.after(40, self._poll_worker)

    def _handle_frame(self, packet: FramePacket) -> None:
        self.last_frame_rgb = packet.frame_rgb
        total_text = "?" if packet.total_frames <= 0 else str(packet.total_frames)
        if packet.class_counts:
            if packet.task == "classify":
                counts_text = ", ".join(
                    f"{name}: {score:.2f}" for name, score in sorted(packet.class_counts.items(), key=lambda item: item[1], reverse=True)
                )
                self.counts_var.set(f"Top classes: {counts_text}")
            else:
                counts_text = ", ".join(
                    f"{name}: {int(count)}" for name, count in sorted(packet.class_counts.items())
                )
                self.counts_var.set(f"Classes: {counts_text}")
        else:
            self.counts_var.set("Classes: none on this frame")
        if packet.task == "classify":
            top_name = next(iter(sorted(packet.class_counts.items(), key=lambda item: item[1], reverse=True)), None)
            top_label = "--" if top_name is None else f"{top_name[0]} {top_name[1]:.2f}"
            self.progress_var.set(
                f"Frame {packet.frame_index}/{total_text} | "
                f"Top class {top_label} | "
                f"Inference {packet.inference_ms:.1f} ms | "
                f"{packet.task}"
            )
        else:
            self.progress_var.set(
                f"Frame {packet.frame_index}/{total_text} | "
                f"Detections {packet.detections} | "
                f"Inference {packet.inference_ms:.1f} ms | "
                f"{packet.task}"
            )
        self.status_var.set(
            f"{packet.source_name} at {packet.fps:.1f} FPS source rate | "
            f"model {self.model_var.get().strip()}"
        )
        self._render_last_frame()

    def _render_last_frame(self) -> None:
        self.video_canvas.delete("all")
        if self.last_frame_rgb is None:
            source_text = "choose a video" if self.source_type_var.get().strip().lower() == "video" else "set a camera index"
            self.video_canvas.create_text(
                self.video_canvas.winfo_width() / 2,
                self.video_canvas.winfo_height() / 2,
                text=f"{source_text.capitalize()} and start the demo to see YOLO26 output here.",
                fill="#dbe4ef",
                font=("Segoe UI", 18),
            )
            return

        canvas_w = max(self.video_canvas.winfo_width(), 100)
        canvas_h = max(self.video_canvas.winfo_height(), 100)
        frame_h, frame_w = self.last_frame_rgb.shape[:2]
        scale = min(canvas_w / frame_w, canvas_h / frame_h)
        new_w = max(1, int(frame_w * scale))
        new_h = max(1, int(frame_h * scale))

        image = Image.fromarray(self.last_frame_rgb)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = image.resize((new_w, new_h), resampling)
        self.current_image = ImageTk.PhotoImage(resized)
        self.video_canvas.create_image(canvas_w / 2, canvas_h / 2, image=self.current_image, anchor="center")

    def _on_close(self) -> None:
        self._stop_demo(silent=True)
        self.root.destroy()


def run_smoke_test_import() -> None:
    print(
        {
            "runtime_dir": str(RUNTIME_DIR),
            "default_models": DEFAULT_MODELS,
            "default_image_sizes": DEFAULT_IMAGE_SIZES,
        }
    )


def run_smoke_test_ui() -> None:
    root = tk.Tk()
    app = YoloVideoDemoApp(root, prompt_for_video=False)
    root.update_idletasks()
    app._render_last_frame()
    print("gui-ok")
    root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultralytics YOLO26 video demo GUI.")
    parser.add_argument("--smoke-test-import", action="store_true", help="Verify import/runtime setup and exit.")
    parser.add_argument("--smoke-test-ui", action="store_true", help="Create the GUI without prompting for a video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test_import:
        run_smoke_test_import()
        return
    if args.smoke_test_ui:
        run_smoke_test_ui()
        return

    root = tk.Tk()
    YoloVideoDemoApp(root, prompt_for_video=False)
    root.mainloop()


if __name__ == "__main__":
    main()

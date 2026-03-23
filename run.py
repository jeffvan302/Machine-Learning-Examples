from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk


WORKSPACE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DemoEntry:
    title: str
    script_name: str
    subtitle: str
    required_files: tuple[str, ...] = ()

    @property
    def script_path(self) -> Path:
        return WORKSPACE_DIR / self.script_name

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        if not self.script_path.exists():
            missing.append(self.script_name)
        for filename in self.required_files:
            if not (WORKSPACE_DIR / filename).exists():
                missing.append(filename)
        return missing


DEMOS = (
    DemoEntry(
        title="1) Forager vs Raider Deep RL GUI",
        script_name="forager_raider_drl_gui.py",
        subtitle="Gridworld deep-RL demo with arena and network visualization.",
    ),
    DemoEntry(
        title="2) Rocket Landing Deep RL GUI",
        script_name="rocket_landing_drl_gui.py",
        subtitle="PPO / REINFORCE rocket demo with replay, graph, and network panel.",
        required_files=("rocket_landing_rl_demo.py",),
    ),
    DemoEntry(
        title="3) MNIST CNN Visualizer",
        script_name="mnist_cnn_visualizer_gui.py",
        subtitle="Interactive CNN training and kernel visualization for handwritten digits.",
    ),
    DemoEntry(
        title="4) Ultralytics YOLO26 Video GUI",
        script_name="ultralytics_yolo26_video_gui.py",
        subtitle="Video or camera inference with detect, segment, pose, OBB, and classify modes.",
    ),
)


class DemoLauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Presentation Demo Launcher")
        self.root.geometry("920x720")
        self.root.minsize(760, 560)
        self.root.configure(bg="#ece6da")

        self.status_var = tk.StringVar(
            value="Launch any available demo below. Start this launcher from the Conda environment you want to use."
        )

        self._build_ui()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)

        ttk.Label(
            outer,
            text="Quick Launch",
            font=("Segoe UI Semibold", 24),
            foreground="#173746",
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            outer,
            text="Open the demos in one click. The launcher uses the same Python interpreter that started it.",
            font=("Segoe UI", 12),
            foreground="#49606b",
        ).grid(row=1, column=0, sticky="w", pady=(6, 18))

        cards = ttk.Frame(outer)
        cards.grid(row=2, column=0, sticky="nsew")
        cards.columnconfigure(0, weight=1)

        for index, demo in enumerate(DEMOS):
            self._add_demo_card(cards, index, demo)

        status_frame = ttk.LabelFrame(outer, text="Status")
        status_frame.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            wraplength=820,
            foreground="#304b57",
            font=("Segoe UI", 11),
        ).pack(anchor="w", padx=12, pady=10)

    def _add_demo_card(self, parent: ttk.Frame, row: int, demo: DemoEntry) -> None:
        card = ttk.Frame(parent, padding=14)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.columnconfigure(0, weight=1)

        missing = demo.missing_requirements()
        title_color = "#173746" if not missing else "#7c4f23"

        ttk.Label(
            card,
            text=demo.title,
            font=("Segoe UI Semibold", 16),
            foreground=title_color,
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            card,
            text=demo.subtitle,
            font=("Segoe UI", 11),
            foreground="#4b5b64",
            wraplength=650,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        if missing:
            note_text = "Unavailable right now: missing " + ", ".join(missing)
            note_color = "#8a5a2b"
        else:
            note_text = f"Script: {demo.script_name}"
            note_color = "#5f6d73"

        ttk.Label(
            card,
            text=note_text,
            font=("Segoe UI", 10),
            foreground=note_color,
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

        button = ttk.Button(
            card,
            text="Launch",
            command=lambda entry=demo: self._launch_demo(entry),
        )
        button.grid(row=0, column=1, rowspan=3, sticky="e")
        if missing:
            button.state(["disabled"])

    def _launch_demo(self, demo: DemoEntry) -> None:
        missing = demo.missing_requirements()
        if missing:
            messagebox.showerror(
                "Cannot launch demo",
                f"{demo.title} cannot be launched because these files are missing:\n\n" + "\n".join(missing),
                parent=self.root,
            )
            self.status_var.set(f"{demo.title} is unavailable because required files are missing.")
            return

        try:
            subprocess.Popen(
                [sys.executable, str(demo.script_path)],
                cwd=str(WORKSPACE_DIR),
            )
        except Exception as exc:
            messagebox.showerror(
                "Launch failed",
                f"Could not launch {demo.title}.\n\n{exc}",
                parent=self.root,
            )
            self.status_var.set(f"Could not launch {demo.title}.")
            return

        self.status_var.set(f"Launched {demo.title} with {Path(sys.executable).name}.")


def main() -> None:
    root = tk.Tk()
    DemoLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

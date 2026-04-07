from __future__ import annotations

import os
import stat
import shutil
import subprocess
import sys
import tempfile
import tkinter as tk
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable


WORKSPACE_DIR = Path(__file__).resolve().parent
EXTERNAL_PROJECTS_DIR = WORKSPACE_DIR / "external"
SCRIPT_BOOTSTRAP = (
    "import runpy, sys; "
    "from pathlib import Path; "
    "script = Path(sys.argv[1]).resolve(); "
    "sys.path.insert(0, str(script.parent)); "
    "sys.argv = [str(script), *sys.argv[2:]]; "
    "runpy.run_path(str(script), run_name='__main__')"
)


@dataclass(frozen=True)
class ManagedProject:
    display_name: str
    repo_url: str
    archive_url: str
    install_dir: Path
    entry_script: str = "run.py"

    @property
    def run_path(self) -> Path:
        return self.install_dir / self.entry_script

    @property
    def repo_name(self) -> str:
        return self.repo_url.rstrip("/").split("/")[-1].removesuffix(".git")

    @property
    def relative_install_dir(self) -> str:
        try:
            return str(self.install_dir.relative_to(WORKSPACE_DIR))
        except ValueError:
            return str(self.install_dir)


ROCKET_PROJECT = ManagedProject(
    display_name="Rocket Landing Lab",
    repo_url="https://github.com/jeffvan302/ml_rocket_lander.git",
    archive_url="https://github.com/jeffvan302/ml_rocket_lander/archive/refs/heads/main.zip",
    install_dir=EXTERNAL_PROJECTS_DIR / "ml_rocket_lander",
)
CAR_PROJECT = ManagedProject(
    display_name="Car Driver Lab",
    repo_url="https://github.com/jeffvan302/ml-car-driver.git",
    archive_url="https://github.com/jeffvan302/ml-car-driver/archive/refs/heads/main.zip",
    install_dir=EXTERNAL_PROJECTS_DIR / "ml-car-driver",
)


def _handle_remove_readonly(func: Callable[..., object], path: str, exc_info: object) -> None:
    _exc_type, exc, _traceback = exc_info
    if not isinstance(exc, PermissionError):
        raise exc
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _clear_install_target(target_dir: Path) -> None:
    if not target_dir.exists():
        return
    if target_dir.is_dir():
        shutil.rmtree(target_dir, onerror=_handle_remove_readonly)
        return
    target_dir.unlink()


def _download_project_archive(project: ManagedProject, target_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=f"{project.repo_name}_", dir=WORKSPACE_DIR) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        archive_path = temp_dir / f"{project.repo_name}.zip"
        with urllib.request.urlopen(project.archive_url, timeout=90) as response:
            archive_path.write_bytes(response.read())
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_dir)
        extracted_dir = next(temp_dir.glob(f"{project.repo_name}-*"), None)
        if extracted_dir is None:
            raise RuntimeError(
                f"Downloaded archive did not contain the expected {project.repo_name} folder."
            )
        shutil.move(str(extracted_dir), str(target_dir))


def _git_head(target_dir: Path, git_executable: str) -> str | None:
    head_result = subprocess.run(
        [git_executable, "-C", str(target_dir), "rev-parse", "HEAD"],
        cwd=str(WORKSPACE_DIR),
        capture_output=True,
        text=True,
    )
    if head_result.returncode != 0:
        return None
    return head_result.stdout.strip() or None


def _install_project_into_dir(project: ManagedProject, target_dir: Path) -> str | None:
    errors: list[str] = []
    git_executable = shutil.which("git")

    _clear_install_target(target_dir)
    if git_executable:
        clone_result = subprocess.run(
            [git_executable, "clone", "--depth", "1", project.repo_url, str(target_dir)],
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
        )
        if clone_result.returncode == 0 and (target_dir / project.entry_script).exists():
            return _git_head(target_dir, git_executable)
        _clear_install_target(target_dir)
        error_output = clone_result.stderr.strip() or clone_result.stdout.strip() or "git clone failed."
        errors.append(f"git clone: {error_output}")
    else:
        errors.append("git clone: Git is not installed or not on PATH.")

    try:
        _download_project_archive(project, target_dir)
    except Exception as exc:
        _clear_install_target(target_dir)
        errors.append(f"zip download: {exc}")
    else:
        if (target_dir / project.entry_script).exists():
            return None
        _clear_install_target(target_dir)
        errors.append(
            f"zip download: install completed but {project.entry_script} was missing afterwards."
        )

    details = "\n\n".join(errors)
    raise RuntimeError(
        f"Could not download the {project.display_name} project from {project.repo_url}.\n\n"
        f"{details}"
    )


def _replace_project_with_latest(project: ManagedProject) -> str | None:
    EXTERNAL_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    replacement_dir = EXTERNAL_PROJECTS_DIR / f".{project.install_dir.name}_update"
    backup_dir = EXTERNAL_PROJECTS_DIR / f".{project.install_dir.name}_backup"

    _clear_install_target(replacement_dir)
    _clear_install_target(backup_dir)
    installed_commit = _install_project_into_dir(project, replacement_dir)

    try:
        if project.install_dir.exists():
            shutil.move(str(project.install_dir), str(backup_dir))
        shutil.move(str(replacement_dir), str(project.install_dir))
    except Exception:
        if not project.install_dir.exists() and backup_dir.exists():
            shutil.move(str(backup_dir), str(project.install_dir))
        raise
    finally:
        _clear_install_target(replacement_dir)
        _clear_install_target(backup_dir)

    return installed_commit


def ensure_external_project(project: ManagedProject) -> Path:
    if project.run_path.exists():
        return project.run_path

    EXTERNAL_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    _install_project_into_dir(project, project.install_dir)
    return project.run_path


def update_external_project(project: ManagedProject) -> str:
    if not project.run_path.exists():
        ensure_external_project(project)
        return f"Installed {project.display_name} into {project.relative_install_dir}."

    installed_commit = _replace_project_with_latest(project)
    if installed_commit:
        return (
            f"Replaced {project.display_name} with a fresh upstream copy at commit "
            f"{installed_commit[:7]}."
        )
    return f"Replaced {project.display_name} with the latest GitHub archive snapshot."


def ensure_rocket_project() -> Path:
    return ensure_external_project(ROCKET_PROJECT)


def update_rocket_project() -> str:
    return update_external_project(ROCKET_PROJECT)


def ensure_car_project() -> Path:
    return ensure_external_project(CAR_PROJECT)


def update_car_project() -> str:
    return update_external_project(CAR_PROJECT)


def launch_script(script_path: Path, *script_args: str) -> None:
    subprocess.Popen(
        [sys.executable, "-c", SCRIPT_BOOTSTRAP, str(script_path), *script_args],
        cwd=str(script_path.parent),
    )


@dataclass(frozen=True)
class DemoEntry:
    title: str
    script_name: str
    subtitle: str
    required_files: tuple[str, ...] = ()
    base_dir: Path = WORKSPACE_DIR
    install_note: str | None = None
    bootstrap: Callable[[], Path] | None = None
    updater: Callable[[], str] | None = None

    @property
    def script_path(self) -> Path:
        return self.base_dir / self.script_name

    @property
    def can_self_install(self) -> bool:
        return self.install_note is not None and self.bootstrap is not None

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        if not self.script_path.exists() and not self.can_self_install:
            missing.append(self.script_name)
        for filename in self.required_files:
            if not (self.base_dir / filename).exists():
                missing.append(filename)
        return missing

    def display_location(self) -> str:
        try:
            return str(self.script_path.relative_to(WORKSPACE_DIR))
        except ValueError:
            return str(self.script_path)


DEMOS = (
    DemoEntry(
        title="1) Rocket Landing Lab",
        script_name="run.py",
        subtitle="Jeff van Niekerk's PPO rocket lander, installed into its own subfolder on first launch.",
        base_dir=ROCKET_PROJECT.install_dir,
        install_note="First launch downloads https://github.com/jeffvan302/ml_rocket_lander into external/ml_rocket_lander.",
        bootstrap=ensure_rocket_project,
        updater=update_rocket_project,
    ),
    DemoEntry(
        title="2) Car Driver Lab",
        script_name="run.py",
        subtitle="Jeff van Niekerk's PPO car driver, installed into its own subfolder on first launch.",
        base_dir=CAR_PROJECT.install_dir,
        install_note="First launch downloads https://github.com/jeffvan302/ml-car-driver into external/ml-car-driver.",
        bootstrap=ensure_car_project,
        updater=update_car_project,
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
        elif demo.install_note and not demo.script_path.exists():
            note_text = demo.install_note
            note_color = "#40606f"
        else:
            note_text = f"Script: {demo.display_location()}"
            note_color = "#5f6d73"

        ttk.Label(
            card,
            text=note_text,
            font=("Segoe UI", 10),
            foreground=note_color,
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

        buttons = ttk.Frame(card)
        buttons.grid(row=0, column=1, rowspan=3, sticky="e")

        launch_button = ttk.Button(
            buttons,
            text="Launch",
            command=lambda entry=demo: self._launch_demo(entry),
        )
        launch_button.grid(row=0, column=0, sticky="e")
        if missing:
            launch_button.state(["disabled"])

        if demo.updater is not None:
            ttk.Button(
                buttons,
                text="Update",
                command=lambda entry=demo: self._update_demo(entry),
            ).grid(row=1, column=0, sticky="e", pady=(8, 0))

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

        script_path = demo.script_path
        try:
            if demo.bootstrap is not None:
                self.status_var.set(f"Preparing {demo.title}...")
                self.root.update_idletasks()
                script_path = demo.bootstrap()
            launch_script(script_path)
        except Exception as exc:
            messagebox.showerror(
                "Launch failed",
                f"Could not launch {demo.title}.\n\n{exc}",
                parent=self.root,
            )
            self.status_var.set(f"Could not launch {demo.title}.")
            return

        self.status_var.set(f"Launched {demo.title} with {Path(sys.executable).name}.")

    def _update_demo(self, demo: DemoEntry) -> None:
        if demo.updater is None:
            messagebox.showinfo(
                "No updater available",
                f"{demo.title} does not have an updater action.",
                parent=self.root,
            )
            return

        try:
            self.status_var.set(f"Updating {demo.title}...")
            self.root.update_idletasks()
            result_message = demo.updater()
        except Exception as exc:
            messagebox.showerror(
                "Update failed",
                f"Could not update {demo.title}.\n\n{exc}",
                parent=self.root,
            )
            self.status_var.set(f"Could not update {demo.title}.")
            return

        messagebox.showinfo(
            "Update complete",
            result_message,
            parent=self.root,
        )
        self.status_var.set(result_message)


def main() -> None:
    root = tk.Tk()
    DemoLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

from __future__ import annotations

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


def _clear_install_target(target_dir: Path) -> None:
    if not target_dir.exists():
        return
    if target_dir.is_dir():
        shutil.rmtree(target_dir)
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


def _replace_project_from_archive(project: ManagedProject) -> None:
    EXTERNAL_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    replacement_dir = EXTERNAL_PROJECTS_DIR / f".{project.install_dir.name}_update"
    backup_dir = EXTERNAL_PROJECTS_DIR / f".{project.install_dir.name}_backup"

    _clear_install_target(replacement_dir)
    _clear_install_target(backup_dir)
    _download_project_archive(project, replacement_dir)

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


def _project_has_local_changes(project: ManagedProject, git_executable: str) -> bool:
    status_result = subprocess.run(
        [git_executable, "-C", str(project.install_dir), "status", "--porcelain"],
        cwd=str(WORKSPACE_DIR),
        capture_output=True,
        text=True,
    )
    return status_result.returncode == 0 and bool(status_result.stdout.strip())


def _project_local_changes(
    project: ManagedProject,
    git_executable: str,
) -> list[tuple[str, str]]:
    status_result = subprocess.run(
        [git_executable, "-C", str(project.install_dir), "status", "--porcelain"],
        cwd=str(WORKSPACE_DIR),
        capture_output=True,
        text=True,
    )
    if status_result.returncode != 0:
        return []

    changes: list[tuple[str, str]] = []
    for raw_line in status_result.stdout.splitlines():
        if len(raw_line) < 4:
            continue
        status = raw_line[:2]
        path = raw_line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        if path:
            changes.append((status, path))
    return changes


def _is_transient_python_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.endswith(".pyc") or "/__pycache__/" in f"/{normalized}"


def _discard_transient_project_changes(
    project: ManagedProject,
    git_executable: str,
    changes: list[tuple[str, str]],
) -> None:
    transient_changes = [
        (status, path)
        for status, path in changes
        if _is_transient_python_artifact(path)
    ]
    if not transient_changes:
        return

    tracked_paths = [path for status, path in transient_changes if status != "??"]
    untracked_paths = [path for status, path in transient_changes if status == "??"]

    if tracked_paths:
        restore_result = subprocess.run(
            [
                git_executable,
                "-C",
                str(project.install_dir),
                "restore",
                "--staged",
                "--worktree",
                "--",
                *tracked_paths,
            ],
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
        )
        if restore_result.returncode != 0:
            error_output = (
                restore_result.stderr.strip()
                or restore_result.stdout.strip()
                or "git restore failed."
            )
            raise RuntimeError(
                f"Could not clean generated Python cache files before updating {project.display_name}.\n\n"
                f"{error_output}"
            )

    for relative_path in untracked_paths:
        target = project.install_dir / Path(relative_path)
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()


def ensure_external_project(project: ManagedProject) -> Path:
    if project.run_path.exists():
        return project.run_path

    EXTERNAL_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    git_executable = shutil.which("git")
    if git_executable:
        _clear_install_target(project.install_dir)
        clone_result = subprocess.run(
            [git_executable, "clone", "--depth", "1", project.repo_url, str(project.install_dir)],
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
        )
        if clone_result.returncode == 0 and project.run_path.exists():
            return project.run_path
        _clear_install_target(project.install_dir)
        error_output = clone_result.stderr.strip() or clone_result.stdout.strip() or "git clone failed."
        errors.append(f"git clone: {error_output}")
    else:
        errors.append("git clone: Git is not installed or not on PATH.")

    try:
        _clear_install_target(project.install_dir)
        _download_project_archive(project, project.install_dir)
    except Exception as exc:
        _clear_install_target(project.install_dir)
        errors.append(f"zip download: {exc}")
    else:
        if project.run_path.exists():
            return project.run_path
        _clear_install_target(project.install_dir)
        errors.append(
            f"zip download: install completed but {project.entry_script} was missing afterwards."
        )

    details = "\n\n".join(errors)
    raise RuntimeError(
        f"Could not download the {project.display_name} project from {project.repo_url}.\n\n"
        f"{details}"
    )


def update_external_project(project: ManagedProject) -> str:
    if not project.run_path.exists():
        ensure_external_project(project)
        return f"Installed {project.display_name} into {project.relative_install_dir}."

    git_executable = shutil.which("git")
    has_git_checkout = (project.install_dir / ".git").exists()
    if git_executable and has_git_checkout:
        changes = _project_local_changes(project, git_executable)
        if changes:
            _discard_transient_project_changes(project, git_executable, changes)
            changes = _project_local_changes(project, git_executable)

        if changes:
            display_paths = "\n".join(
                f"- {path}"
                for _status, path in changes[:8]
            )
            if len(changes) > 8:
                display_paths += "\n- ..."
            raise RuntimeError(
                f"{project.display_name} has local changes in {project.relative_install_dir}, so the launcher "
                "will not auto-update it. Commit, stash, or remove those changes first.\n\n"
                f"Blocking files:\n{display_paths}"
            )

        old_head_result = subprocess.run(
            [git_executable, "-C", str(project.install_dir), "rev-parse", "HEAD"],
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
        )
        pull_result = subprocess.run(
            [git_executable, "-C", str(project.install_dir), "pull", "--ff-only"],
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
        )
        if pull_result.returncode != 0:
            error_output = pull_result.stderr.strip() or pull_result.stdout.strip() or "git pull failed."
            raise RuntimeError(f"Could not update {project.display_name}.\n\n{error_output}")

        new_head_result = subprocess.run(
            [git_executable, "-C", str(project.install_dir), "rev-parse", "HEAD"],
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
        )
        old_head = old_head_result.stdout.strip()
        new_head = new_head_result.stdout.strip()
        if old_head and new_head and old_head != new_head:
            return f"Updated {project.display_name} to commit {new_head[:7]}."
        return f"{project.display_name} is already up to date."

    _replace_project_from_archive(project)
    return f"Refreshed {project.display_name} from the latest GitHub archive."


def ensure_rocket_project() -> Path:
    return ensure_external_project(ROCKET_PROJECT)


def update_rocket_project() -> str:
    return update_external_project(ROCKET_PROJECT)


def ensure_car_project() -> Path:
    return ensure_external_project(CAR_PROJECT)


def update_car_project() -> str:
    return update_external_project(CAR_PROJECT)


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
            subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(script_path.parent),
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

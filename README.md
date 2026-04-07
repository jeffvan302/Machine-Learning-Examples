# Presentation Tools

This folder contains several presentation-friendly Python GUIs and demos for reinforcement learning, CNN visualization, and Ultralytics YOLO26 inference.

## If you are new to Python then here is a quick way around.

The launch.exe will setup a runtime folder and download and configure the stand alone version of python. Then it will run a launcher that will allow running the 4 examples.
The first time will take a bit of time, since it is a setup effectively.  But it runs independently so no conflict issues.  
Subsequent runs will just run the launcher while the runtime folder exists. The launcher now downloads [jeffvan302/ml_rocket_lander](https://github.com/jeffvan302/ml_rocket_lander) and [jeffvan302/ml-car-driver](https://github.com/jeffvan302/ml-car-driver) into the `external` folder the first time you open those demos, so this repo does not need to carry duplicate local copies.

## Vibe Coding your own rocket game with learning ability
You can see the Vibe Coding with project requirements at this repository: [https://github.com/jeffvan302/ml_rocket_lander](https://github.com/jeffvan302/ml_rocket_lander).
If you provide the requirements document to a coding LLM then it should code a similar "game" trainer.

## Install Python With Miniconda

Official Miniconda getting-started guide:

- https://www.anaconda.com/docs/getting-started/main

**Download instructions of Miniconda**
1. Click the Download button at the top right and follow the instructions.
2. Click on Get Started.
3. Signup!
4. Select Miniconda to download.

Recommended Windows setup for this folder:

1. Download and install Miniconda by following the official guide above.
2. During installation, use the standard Miniconda Prompt / Anaconda Prompt workflow from the official docs.
3. Open `Miniconda Prompt` after installation finishes.
4. Create the environment used by these tools:

```powershell
conda create -n sd-env python=3.13 -y
```

5. Activate the environment:

```powershell
conda activate sd-env
```

6. Change into this project folder:

```powershell
cd "<Folder>"
```

7. Install the packages from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

8. Verify the active Python version if you want:

```powershell
python --version
```

How to use `requirements.txt`:

- Always activate the target Conda environment first.
- Run `pip install -r requirements.txt` from this folder to install or refresh the required packages.
- If `requirements.txt` changes later, run the same command again to bring the environment up to date.
- `requirements.txt` handles the pip-installed libraries. `tkinter` normally comes with Python itself.

## Setup

Notes:

- `tkinter` is used by the GUIs but is usually included with Python rather than installed through `pip`.
- The Ultralytics demo may download missing model weights the first time a model is used.
- The MNIST GUI will ask to download the dataset if it is missing from the configured data folder.

## Files In This Folder

- `run.py`
- `mnist_cnn_visualizer_gui.py`
- `ultralytics_yolo26_video_gui.py`
- `requirements.txt`
- `external/ml_rocket_lander` (created automatically on first rocket launch)
- `external/ml-car-driver` (created automatically on first car-driver launch)

## Quick Launcher

Script:

```powershell
python run.py
```

What it does:

- Opens a small launcher window with quick-launch buttons for the main GUI demos.
- Uses the same Python interpreter and active Conda environment that started the launcher.
- Shows when a demo cannot be launched because a required file is missing.
- Downloads `jeffvan302/ml_rocket_lander` into `external/ml_rocket_lander` the first time `Rocket Landing Lab` is launched.
- Downloads `jeffvan302/ml-car-driver` into `external/ml-car-driver` the first time `Car Driver Lab` is launched.
- Adds an `Update` button on the `Rocket Landing Lab` and `Car Driver Lab` cards so you can refresh those vendored copies later.

Button order:

1. `Rocket Landing Lab`
2. `Car Driver Lab`
3. `MNIST CNN Visualizer`
4. `Ultralytics YOLO26 Video GUI`

## Rocket Landing Lab

Script:

```powershell
python run.py
```

What it does:

- Launches the top-level demo picker.
- When you choose `Rocket Landing Lab`, the launcher downloads [jeffvan302/ml_rocket_lander](https://github.com/jeffvan302/ml_rocket_lander) into `external/ml_rocket_lander` if it is not already present, then starts that project's own `run.py`.
- Keeps the upstream rocket project isolated in its own subfolder so there is no `run.py` filename conflict in the root folder.


1. Launch the script.
2. Click `Rocket Landing Lab`.
3. Wait for the first-run download to finish if the subfolder has not been installed yet.
4. Use the upstream rocket project's GUI as normal.

The `Update` button on that card will replace the local external copy with a fresh upstream version. Any local edits inside `external/ml_rocket_lander` are overwritten by design.

Once installed, you can also run the upstream project directly:

```powershell
python external/ml_rocket_lander/run.py
python external/ml_rocket_lander/run.py gui
python external/ml_rocket_lander/run.py smoke-test
```

## Car Driver Lab

Script:

```powershell
python run.py
```

What it does:

- Launches the top-level demo picker.
- When you choose `Car Driver Lab`, the launcher downloads [jeffvan302/ml-car-driver](https://github.com/jeffvan302/ml-car-driver) into `external/ml-car-driver` if it is not already present, then starts that project's own `run.py`.
- Keeps the upstream car-driver project isolated in its own subfolder so there is no `run.py` filename conflict in the root folder.

1. Launch the script.
2. Click `Car Driver Lab`.
3. Wait for the first-run download to finish if the subfolder has not been installed yet.
4. Use the upstream car-driver project's GUI as normal.

The `Update` button on that card will replace the local external copy with a fresh upstream version. Any local edits inside `external/ml-car-driver` are overwritten by design.

Once installed, you can also run the upstream project directly:

```powershell
python external/ml-car-driver/run.py
python external/ml-car-driver/run.py gui
python external/ml-car-driver/run.py smoke-test
```

## MNIST CNN Visualizer

Script:

```powershell
python mnist_cnn_visualizer_gui.py
```

What it does:

- Trains a configurable CNN on handwritten digits from the MNIST dataset.
- Shows the input digit, learned convolution kernels, feature maps, dense activations, and the fixed `0-9` output layer.
- Lets you add or remove convolution and dense layers in the GUI.
- Includes visualizer zoom controls plus `Fit to View` so the architecture can be scaled to the available display area.

How to use it:

1. Launch the script.
2. If MNIST is not present in the configured data folder, the GUI will ask whether to download it.
3. Adjust `epochs`, `batch size`, `learning rate`, `weight decay`, `activation`, and the layer layout on the left.
4. Press `Start / Restart` to build the current architecture and begin training.
5. Press `Pause` to inspect individual test samples more carefully.
6. Use `Previous`, `Next`, `Random`, or the sample index box to browse test digits while paused or after training finishes.

Useful command-line options:

- `--data-dir <path>`: use a different MNIST folder.
- `--smoke-test-model`: build the network and run a tiny tensor through it without opening the GUI.
- `--smoke-test-ui`: instantiate the GUI without the startup dataset prompt.

Examples:

```powershell
python mnist_cnn_visualizer_gui.py
```

## Ultralytics YOLO26 Video GUI

Script:

```powershell
python ultralytics_yolo26_video_gui.py
```

What it does:

- Runs Ultralytics YOLO26 inference on either a video file or a local camera stream.
- Supports `detect`, `segment`, `pose`, `obb`, and `classify`.
- Shows the annotated output frame, inference timing, and per-frame class summary.

How to use it:

1. Launch the script.
2. In the `Source` section choose either `video` or `camera`.
3. If `video` is selected, click `Choose...` and pick a file.
4. If `camera` is selected, enter the camera index, usually `0` for the first camera.
5. Choose the task:
   - `detect` for bounding boxes
   - `segment` for instance segmentation masks
   - `pose` for keypoints and skeletons
   - `obb` for oriented bounding boxes
   - `classify` for whole-frame classification
6. Press `Start / Restart` once the source is ready.
7. Use `Pause`, `Resume`, and `Stop` while the stream or video is running.

Notes:

- `Start / Restart` stays disabled until a valid video is selected or a camera index is provided.
- The default nano models are used automatically for each task.
- `yolo26n-cls.pt` is not currently present in this folder, so the first `classify` run may trigger a model download.

Useful command-line options:

- `--smoke-test-import`: verify the local Ultralytics runtime setup and default models
- `--smoke-test-ui`: instantiate the GUI without starting inference

# Miniconda and Python Quick Guide

This guide shows how to install Miniconda on Windows, open a terminal in a folder, create and manage Conda environments, install Python packages, and run Python scripts.

## 1. Install Miniconda on Windows

1. Open your web browser and go to the Miniconda download page: https://www.anaconda.com/docs/getting-started/main
2. Download the latest **Miniconda installer for Windows**.
3. Double-click the installer to start setup.
4. Click **Next** through the welcome pages.
5. Accept the license agreement.
6. Choose **Just Me** unless you specifically want it available for all users.
7. Choose the install location, or keep the default location.
8. On the advanced options screen:
   - You can usually leave the defaults as they are.
   - If you are unsure, do **not** change the PATH option manually.
9. Click **Install**.
10. When installation is complete, click **Finish**.
11. Open **Miniconda Prompt** from the Start menu.
12. Check that Conda works by running:

```powershell
conda --version
```

If a version number appears, Miniconda is installed correctly.

## 2. Open a Folder in Terminal from File Explorer

### Windows 11

1. Open **File Explorer**.
2. Browse to the folder you want to work in.
3. Right-click inside the folder background.
4. Click **Open in Terminal**.
5. A terminal window will open already pointed at that folder.

### Another easy method

1. Open the folder in **File Explorer**.
2. Click in the address bar.
3. Type:

```text
cmd
```

or:

```text
powershell
```

4. Press **Enter**.
5. A terminal opens in that folder.

### If you do not see "Open in Terminal"

1. Try **Shift + right-click** inside the folder.
2. On some systems, you may see options like **Open PowerShell window here**.
3. You can also use the address bar method shown above.

## 3. Create, Activate, and List Conda Environments

### Create a new environment

1. Open **Miniconda Prompt** or a terminal where Conda is available.
2. Run:

```powershell
conda create --name myenv python=3.11
```

3. Press `y` when Conda asks for confirmation.

This creates a new environment named `myenv` with Python 3.11 installed.

### Activate an environment

1. In the terminal, run:

```powershell
conda activate myenv
```

2. You should now see the environment name at the start of the prompt, for example:

```text
(myenv)
```

### How to know which environment you are currently using

You can check the active environment in a few easy ways:

1. Look at the beginning of the terminal prompt.
2. If you see something like `(myenv)`, then `myenv` is the active environment.
3. If you see `(base)`, then the default Conda base environment is active.

You can also confirm it with a command:

```powershell
conda info --envs
```

The active environment will have a `*` next to it.

In PowerShell, you can also display the current Conda environment directly:

```powershell
echo $env:CONDA_DEFAULT_ENV
```

### Deactivate the current environment

```powershell
conda deactivate
```

### List all Conda environments

Use either of these commands:

```powershell
conda env list
```

```powershell
conda info --envs
```

### If `conda activate` does not work in PowerShell

Run this once:

```powershell
conda init powershell
```

Then close the terminal and open it again.

## 4. Install Python Packages with `pip` or `requirements.txt`

It is best to activate your Conda environment before installing packages.

### Install a single package with `pip`

1. Activate your environment:

```powershell
conda activate myenv
```

2. Install a package:

```powershell
pip install numpy
```

3. To install multiple packages, repeat the command with different package names.

### Install packages from `requirements.txt`

1. Make sure your terminal is in the project folder.
2. Activate your environment:

```powershell
conda activate myenv
```

3. Run:

```powershell
pip install -r requirements.txt
```

This installs all packages listed in the `requirements.txt` file.

### Save your current pip packages to `requirements.txt`

```powershell
pip freeze > requirements.txt
```

## 5. Run Python Scripts

### Run a script from the current folder

1. Open a terminal in the folder that contains your script.
2. Activate your environment if needed:

```powershell
conda activate myenv
```

3. Run the script:

```powershell
python script_name.py
```

Example:

```powershell
python hello.py
```

### Run a script from another folder

Use the full or relative path:

```powershell
python C:\path\to\script_name.py
```

### Check which Python is being used

```powershell
where python
```

This helps confirm that your Conda environment is active.

## 6. Other Useful Conda Commands

### Check Conda version

```powershell
conda --version
```

### See installed packages in the active environment

```powershell
conda list
```

### Install a package with Conda

```powershell
conda install pandas
```

### Remove a package

```powershell
conda remove pandas
```

### Update Conda itself

```powershell
conda update conda
```

### Update all packages in the active environment

```powershell
conda update --all
```

### Export an environment to a file

```powershell
conda env export > environment.yml
```

### Create an environment from a file

```powershell
conda env create -f environment.yml
```

### Remove an environment

```powershell
conda env remove --name myenv
```

### Rename an environment

Conda does not have a direct rename command on older setups, so the common approach is:

1. Export the environment:

```powershell
conda env export --name myenv > environment.yml
```

2. Create a new environment with a new name:

```powershell
conda env create --name myenv_new -f environment.yml
```

3. Remove the old environment if you no longer need it:

```powershell
conda env remove --name myenv
```

## Quick Example Workflow

```powershell
conda create --name demoenv python=3.11
conda activate demoenv
pip install -r requirements.txt
python demo_launcher.py
```

## Tips

- Use one Conda environment per project when possible.
- Activate the environment before installing packages or running scripts.
- If Conda is not recognized in a normal terminal, try **Miniconda Prompt** first.
- Use `conda list` and `pip list` to inspect installed packages.

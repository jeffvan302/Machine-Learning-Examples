# VS Code and GitHub Quick Guide

This guide shows how to install Git on Windows for use with GitHub, install Visual Studio Code, add useful extensions for Python and GitHub, configure them, open folders, and clone repositories to your computer.

## 1. Install Git on Windows for GitHub

This section uses **Git for Windows**, which is the tool most people install so they can work with GitHub repositories from VS Code and the terminal.

1. Open your web browser and go to the Git for Windows download page:

```text
https://git-scm.com/download/win
```

2. Download the Windows installer.
3. Double-click the installer to start setup.
4. Click through the install steps.
5. If you are unsure, the default options are usually fine.
6. Finish the installation.
7. Open **PowerShell**, **Command Prompt**, or **Windows Terminal**.
8. Check that Git is installed:

```powershell
git --version
```

If a version number appears, Git is installed correctly.

### Optional: Set your Git name and email

These details are attached to your commits.

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## 2. Install Visual Studio Code

1. Open your web browser and go to the VS Code download page:

```text
https://code.visualstudio.com/
```

2. Download the **Windows** installer.
3. Run the installer.
4. Accept the license agreement.
5. Choose the install location, or keep the default.
6. On the additional tasks page, these options are usually useful:
   - Add VS Code to PATH
   - Register Code as an editor
   - Add "Open with Code" actions if available
7. Click **Install**.
8. When setup finishes, open **Visual Studio Code**.

## 3. Install VS Code Extensions for Python and GitHub

### Open the Extensions view

1. Open **VS Code**.
2. Click the **Extensions** icon on the left sidebar.
3. Search for each extension by name.
4. Click **Install**.

### Useful extensions for Python

- **Python** by Microsoft: main Python support for VS Code.
- **Pylance** by Microsoft: better IntelliSense, type checking, and code navigation.
- **Jupyter** by Microsoft: useful if you work with notebooks.

### Useful extensions for Git and GitHub

- **GitHub Pull Requests and Issues** by GitHub: helps you review pull requests and issues inside VS Code.
- **GitLens**: adds stronger Git history, blame, and repository insight inside the editor.
- **GitHub Actions**: useful if your repository uses GitHub Actions workflows.

## 4. Configure the Extensions If Needed

### Select the correct Python interpreter

1. Open a Python project folder in VS Code.
2. Press `Ctrl+Shift+P` to open the Command Palette.
3. Search for:

```text
Python: Select Interpreter
```

4. Choose the Python or Conda environment you want to use.

If you are using Miniconda, choose the interpreter from the environment you created, not just any Python installation.

### Sign in to GitHub

1. In VS Code, click the **Accounts** icon in the top-right corner.
2. Choose **Sign in with GitHub**.
3. Follow the browser prompts to complete sign-in.

This is helpful for GitHub extensions, pull requests, and repository access.

### Recommended Python settings

These are optional, but useful for many Python projects.

1. Open **Settings** in VS Code.
2. Search for these settings and enable them if you want:
   - `Python: Terminal Activate Environment`
   - `Editor: Format On Save`
3. If you use a formatter such as `black`, install it in your environment first:

```powershell
pip install black
```

### If Git is not detected in VS Code

1. Make sure Git is installed by running:

```powershell
git --version
```

2. Close and reopen VS Code.
3. If needed, restart your computer so PATH changes are applied.

## 5. Open a Folder in VS Code

### Open a folder from inside VS Code

1. Open **VS Code**.
2. Click **File**.
3. Click **Open Folder...**
4. Browse to your project folder.
5. Click **Select Folder**.

### Open a folder from File Explorer

If the installer added the right-click option, you may be able to:

1. Open **File Explorer**.
2. Right-click the folder.
3. Click **Open with Code**.

### Open a folder from the terminal

If the `code` command is available, go to your folder in the terminal and run:

```powershell
code .
```

That opens the current folder in VS Code.

## 6. Clone a GitHub Repository to Your Computer

### Method 1: Clone with VS Code

1. Open **VS Code**.
2. Press `Ctrl+Shift+P`.
3. Search for:

```text
Git: Clone
```

4. Paste the GitHub repository URL, for example:

```text
https://github.com/username/repository-name.git
```

5. Choose the local folder where you want to save the repository.
6. Wait for the clone to finish.
7. When prompted, click **Open** to open the cloned repository in VS Code.

### Method 2: Clone with the terminal

1. Open **PowerShell**, **Command Prompt**, or **Windows Terminal**.
2. Move to the folder where you want to store the project:

```powershell
cd C:\Users\YourName\Code
```

3. Run:

```powershell
git clone https://github.com/username/repository-name.git
```

4. Move into the new folder:

```powershell
cd repository-name
```

5. Open it in VS Code:

```powershell
code .
```

## Quick Example Workflow

```powershell
git clone https://github.com/username/repository-name.git
cd repository-name
code .
```

## Tips

- Install Git before trying to clone repositories in VS Code.
- After opening a Python project, always confirm the selected interpreter in VS Code.
- If VS Code does not show Git features, first confirm that `git --version` works in the terminal.
- If a repository uses Python, open the project folder and then select the correct Conda or Python interpreter.

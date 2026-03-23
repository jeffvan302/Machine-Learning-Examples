@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if not exist "!VSWHERE!" (
    echo Could not find vswhere.exe at:
    echo   !VSWHERE!
    exit /b 1
)

for /f "usebackq delims=" %%I in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VSINSTALL=%%~I"
)

if not defined VSINSTALL (
    echo Could not find a Visual Studio installation with C++ build tools.
    exit /b 1
)

call "!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

pushd "!SCRIPT_DIR!"
cl /std:c++20 /EHsc /nologo /DUNICODE /D_UNICODE /DWIN32_LEAN_AND_MEAN /utf-8 ^
    src\main.cpp ^
    /Fe:launch.exe ^
    /link comctl32.lib urlmon.lib user32.lib shell32.lib
set "BUILD_EXIT=!ERRORLEVEL!"
popd

if not "!BUILD_EXIT!"=="0" (
    echo.
    echo Build failed with exit code !BUILD_EXIT!.
    exit /b !BUILD_EXIT!
)

echo.
echo Built "!SCRIPT_DIR!launch.exe"
exit /b 0

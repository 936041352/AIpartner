@echo off
setlocal EnableExtensions

cd /d "%~dp0"
title AIpartner Launcher

set "APP_URL=http://127.0.0.1:8000"
set "CONDA_ENV=aipartner"
set "CONDA_EXE="

rem Find Conda.
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" set "CONDA_EXE=C:\ProgramData\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "C:\ProgramData\anaconda3\Scripts\conda.exe" set "CONDA_EXE=C:\ProgramData\anaconda3\Scripts\conda.exe"

if not defined CONDA_EXE (
    for /f "delims=" %%I in ('where conda.exe 2^>nul') do (
        if not defined CONDA_EXE set "CONDA_EXE=%%I"
    )
)

if not defined CONDA_EXE (
    echo [ERROR] Conda was not found.
    echo Please check your Miniconda or Anaconda installation.
    pause
    exit /b 1
)

if not exist "%~dp0main.py" (
    echo [ERROR] main.py was not found.
    echo Put this BAT file in the AIpartner project root.
    pause
    exit /b 1
)

rem If AIpartner is already running, only open the page.
curl.exe --silent --fail --max-time 2 "%APP_URL%/api/characters" >nul 2>&1
if not errorlevel 1 (
    start "" "%APP_URL%"
    exit /b 0
)

echo Checking Conda environment: %CONDA_ENV%
call "%CONDA_EXE%" run -n "%CONDA_ENV%" python -c "import uvicorn" >nul 2>&1

if errorlevel 1 (
    echo [ERROR] The Conda environment is unavailable or incomplete.
    echo Environment name: %CONDA_ENV%
    pause
    exit /b 1
)

echo Starting AIpartner...

start "AIpartner Server" cmd.exe /d /k ""%CONDA_EXE%" run --no-capture-output -n "%CONDA_ENV%" python -m uvicorn main:app --host 127.0.0.1 --port 8000"

rem Wait until the web service is ready.
for /l %%I in (1,1,60) do (
    curl.exe --silent --fail --max-time 2 "%APP_URL%/api/characters" >nul 2>&1
    if not errorlevel 1 goto server_ready
    timeout /t 1 /nobreak >nul
)

echo [ERROR] AIpartner did not start within 60 seconds.
echo Check the "AIpartner Server" window for error details.
pause
exit /b 1

:server_ready
echo AIpartner is ready.
start "" "%APP_URL%"
exit /b 0
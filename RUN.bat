@echo off
setlocal EnableDelayedExpansion

echo.
echo  ============================================
echo    SPLATT2 — Target Shooting Trainer
echo  ============================================
echo.

REM ── Check Python is available ────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo.
    echo  Please install Python 3.9 or later from:
    echo    https://www.python.org/downloads/
    echo.
    echo  During installation, tick "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Python %PYVER% found.

REM ── Install / update dependencies (fast if already installed) ────────────
echo  Checking dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERROR] Could not install dependencies.
    echo  Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

REM ── Launch ───────────────────────────────────────────────────────────────
echo  Starting Splatt2...
echo.
python main.py
if errorlevel 1 (
    echo.
    echo  [ERROR] Splatt2 exited with an error.
    if exist splatt2_crash.log (
        echo  Crash details saved to: splatt2_crash.log
    )
    echo.
    pause
)

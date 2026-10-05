@echo off
rem Drag a PDF / EPUB / TXT file onto this icon to make an audiobook with your settings.toml.
cd /d "%~dp0.."
if "%~1"=="" (
  echo Drag a PDF, EPUB or TXT file onto this icon.
  pause
  exit /b 0
)
if not exist .venv\Scripts\python.exe (
  echo Run windows\install.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m inkvoice convert "%~1"
pause

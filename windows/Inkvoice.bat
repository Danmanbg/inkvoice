@echo off
rem Opens the Inkvoice web app in your browser.
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe (
  echo Run windows\install.bat first.
  pause
  exit /b 1
)
echo Starting Inkvoice... (keep this window open while you use the app)
.venv\Scripts\python.exe -m inkvoice gui
pause

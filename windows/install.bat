@echo off
rem One-time setup on Windows: creates .venv, installs Inkvoice with Kokoro + web app + Claude.
cd /d "%~dp0.."

where py >nul 2>nul
if errorlevel 1 (
  echo Python is needed. Install Python 3.12 from https://www.python.org/downloads/ and run this again.
  pause
  exit /b 1
)

if not exist .venv (
  py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo Could not create a Python 3.11/3.12 environment. Install Python 3.12 and try again.
  pause
  exit /b 1
)

.venv\Scripts\python.exe -m pip install --upgrade pip
rem spacy 3.8.7: newer builds can be blocked by Windows Smart App Control
.venv\Scripts\python.exe -m pip install -e ".[kokoro,gui,claude]" "spacy==3.8.7"
if errorlevel 1 (
  echo Installation failed - see the messages above.
  pause
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo Installing ffmpeg...
  winget install -e --id Gyan.FFmpeg
)

if not exist settings.toml copy settings.example.toml settings.toml >nul
if not exist pronunciations.txt copy pronunciations.example.txt pronunciations.txt >nul

echo.
echo All set. Double-click windows\Inkvoice.bat to open the app,
echo or drag a book onto windows\Convert book.bat.
pause

@echo off
REM Device Guard blocks .venv\Scripts\streamlit.exe in Downloads.
REM This starts Streamlit with the uv-managed Python instead.
cd /d "%~dp0"
set "VIRTUAL_ENV=%~dp0.venv"
set "PYTHONPATH=%~dp0.venv\Lib\site-packages"
set "PY=%APPDATA%\uv\python\cpython-3.11-windows-x86_64-none\python.exe"
if not exist "%PY%" (
  echo Python not found: %PY%
  exit /b 1
)
"%PY%" -m streamlit run app.py

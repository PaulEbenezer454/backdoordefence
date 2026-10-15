@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" web_demo.py
) else (
  py web_demo.py
)
pause


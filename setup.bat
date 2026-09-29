@echo off
REM Double-click to install NotesNPU (dependencies + models).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
pause

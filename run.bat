@echo off
REM Double-click to start NotesNPU. Opens in your browser; runs 100%% offline.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
pause

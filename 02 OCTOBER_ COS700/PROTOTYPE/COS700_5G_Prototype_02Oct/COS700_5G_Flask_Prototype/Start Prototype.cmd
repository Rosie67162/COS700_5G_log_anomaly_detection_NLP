@echo off
REM Open the local research application with its PowerShell launcher.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_prototype.ps1"
if errorlevel 1 pause

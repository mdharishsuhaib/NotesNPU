<#
  NotesNPU one-click setup.
  - Picks a *native ARM64* Python on Snapdragon PCs (required for the Qualcomm NPU / QNN EP)
  - Creates .venv, installs dependencies, downloads models, creates a sample lecture.

  Usage:  .\setup.ps1            (full: Whisper NPU + CPU, Phi-3.5-mini LLM ~2.6 GB)
          .\setup.ps1 -NoLLM     (skip the LLM; app runs in fast extractive mode)
#>
param([switch]$NoLLM)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Test-Arm64Host {
  return ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") -or ($env:PROCESSOR_ARCHITEW6432 -eq "ARM64") -or
         ((Get-CimInstance Win32_Processor).Name -match "Snapdragon")
}

function Find-Python {
  $arm = Test-Arm64Host
  $candidates = @()
  if (Get-Command py -ErrorAction SilentlyContinue) {
    if ($arm) { $candidates += @("-3.12-arm64", "-3.13-arm64", "-3.11-arm64") }
    $candidates += @("-3.12", "-3.13", "-3.11", "-3")
    foreach ($c in $candidates) {
      & py $c -c "import sys" 2>$null
      if ($LASTEXITCODE -eq 0) { return @("py", $c) }
    }
  }
  if (Get-Command python -ErrorAction SilentlyContinue) { return @("python") }
  throw "Python 3.11+ not found. Install it from https://www.python.org/downloads/windows/ (choose ARM64 on Snapdragon)."
}

Write-Host "=== NotesNPU setup ===" -ForegroundColor Cyan
$py = Find-Python
$exe = $py[0]; $pyArgs = @(); if ($py.Count -gt 1) { $pyArgs = @($py[1]) }
$machine = & $exe @pyArgs -c "import platform; print(platform.machine())"
Write-Host "Python: $(& $exe @pyArgs --version) [$machine]"
if ((Test-Arm64Host) -and $machine -ne "ARM64") {
  Write-Warning "This is a Snapdragon PC but Python is $machine (emulated). The NPU needs native ARM64 Python:"
  Write-Warning "  winget install -e --id Python.Python.3.12 --architecture arm64   then re-run .\setup.ps1"
  Write-Warning "Continuing with CPU-only mode..."
}

if (-not (Test-Path .venv)) { & $exe @pyArgs -m venv .venv }
$vpy = ".\.venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip --quiet
Write-Host "Installing dependencies..." -ForegroundColor Cyan
& $vpy -m pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
foreach ($opt in @("sounddevice", "scipy")) {
  & $vpy -m pip install $opt --quiet 2>$null
  if ($LASTEXITCODE -ne 0) { Write-Host "  (optional $opt not available on this platform - skipped)" }
}

Write-Host "Downloading models..." -ForegroundColor Cyan
$dlArgs = @("scripts\download_models.py")
if ($NoLLM) { $dlArgs += @("--llm", "none") }
& $vpy @dlArgs
if ($LASTEXITCODE -ne 0) { throw "model download failed" }

if (-not (Test-Path samples\sample_lecture.wav)) { & $vpy scripts\make_sample.py }

Write-Host "`nChecking hardware acceleration..." -ForegroundColor Cyan
& $vpy -c "from core.device import system_summary as s; import json; print(json.dumps(s(), indent=2))"
Write-Host "`nSetup complete. Launch with .\run.ps1 (or double-click run.bat)" -ForegroundColor Green

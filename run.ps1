<#
  Launch NotesNPU (opens http://127.0.0.1:7860 in your browser). Everything runs locally.
  .\run.ps1            -> auto (NPU if available)
  .\run.ps1 -Cpu       -> force CPU (for comparison)
#>
param([switch]$Cpu, [int]$Port = 7860)
Set-Location $PSScriptRoot
if (-not (Test-Path .\.venv\Scripts\python.exe)) { Write-Host "Run .\setup.ps1 first."; exit 1 }
$env:HF_HUB_OFFLINE = "1"          # guarantee: no network calls at runtime
$env:GRADIO_ANALYTICS_ENABLED = "False"
$appArgs = @("app.py", "--port", "$Port")
if ($Cpu) { $appArgs += @("--asr", "cpu") }
& .\.venv\Scripts\python.exe @appArgs

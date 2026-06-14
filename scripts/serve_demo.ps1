# Presentation mode - the rock-solid way to run costsight for a live demo.
#
#   powershell -ExecutionPolicy Bypass -File scripts/serve_demo.ps1
#
# vs. dev mode (`npm run dev`), this:
#   * serves the PRODUCTION build via `vite preview` (no HMR, no on-the-fly
#     recompiles, no dep-optimizer hiccups mid-demo),
#   * runs uvicorn WITHOUT --reload (no file-watch restarts),
#   * sets COSTSIGHT_OFFLINE=1 so AI Explain uses the deterministic template
#     (zero network calls - nothing to hang on venue wifi),
#   * pre-warms the snapshot cache for the demo scenarios (no cold waits),
#   * opens the browser when everything is ready.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "1/4  Building the frontend (production)..." -ForegroundColor Cyan
npm --prefix frontend install --silent
npm --prefix frontend run build

Write-Host "2/4  Starting FastAPI on :8000 (no reload, offline)..." -ForegroundColor Cyan
Start-Process powershell -WorkingDirectory $root -ArgumentList @(
  '-NoExit','-Command',
  "`$env:PYTHONPATH='src'; `$env:COSTSIGHT_OFFLINE='1'; python -m uvicorn cloud_anomaly.api:app --port 8000 --log-level warning"
)

Write-Host "3/4  Serving the web app on :5173 (static production build)..." -ForegroundColor Cyan
Start-Process powershell -WorkingDirectory $root -ArgumentList @(
  '-NoExit','-Command',
  "npm --prefix frontend run preview"
)

Write-Host "4/4  Waiting for servers and warming the cache..." -ForegroundColor Cyan
function Wait-Url($url) {
  for ($i = 0; $i -lt 80; $i++) {
    try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null; return $true }
    catch { Start-Sleep -Milliseconds 500 }
  }
  return $false
}

if (Wait-Url "http://localhost:8000/api/scenarios") {
  foreach ($q in @(
      "scenario=default&n_days=90&seed=42",
      "scenario=spike_storm&n_days=60&seed=7",
      "scenario=stealth_leak&n_days=90&seed=3",
      "scenario=multi_region&n_days=90&seed=5")) {
    try { Invoke-WebRequest "http://localhost:8000/api/snapshot?$q" -UseBasicParsing -TimeoutSec 30 | Out-Null } catch {}
  }
}
[void](Wait-Url "http://localhost:5173/")

Write-Host ""
Write-Host "READY  ->  http://localhost:5173" -ForegroundColor Green
Write-Host "  API docs : http://localhost:8000/docs"
Write-Host "  Mode     : production build, offline (AI Explain = template), cache warmed"
Write-Host "  Stop     : close the two PowerShell windows that opened"
Start-Process "http://localhost:5173"

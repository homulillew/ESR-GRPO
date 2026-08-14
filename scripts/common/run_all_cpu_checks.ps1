$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = Join-Path $projectRoot "src"
Push-Location $projectRoot
try {
    python -P -m compileall -q src scripts tests
    python -P -m pytest -q
    python -P scripts/experiment1_pipeline/run_smoke_episode.py --output-dir outputs/smoke --overwrite
} finally {
    Pop-Location
}

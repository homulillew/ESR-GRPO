[CmdletBinding()]
param(
    [string]$DestinationRoot,
    [string]$Revision = "main",
    [int]$MaxWorkers = 2,
    [switch]$VerifyOnly,
    [switch]$VerifyLfsSha256,
    [switch]$IgnoreEnvironmentProxy,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $DestinationRoot) {
    $DestinationRoot = Join-Path $ProjectRoot "models"
}
$Downloader = Join-Path $PSScriptRoot "download_hf_model.py"
$ModelOutput = Join-Path $DestinationRoot "Qwen3-32B"
$Arguments = @(
    $Downloader,
    "--repo-id", "Qwen/Qwen3-32B",
    "--output", $ModelOutput,
    "--revision", $Revision,
    "--max-workers", $MaxWorkers
)
if ($VerifyOnly) { $Arguments += "--verify-only" }
if ($VerifyLfsSha256) { $Arguments += "--verify-lfs-sha256" }
if ($IgnoreEnvironmentProxy) { $Arguments += "--ignore-environment-proxy" }
if ($DryRun) { $Arguments += "--dry-run" }

python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

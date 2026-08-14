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
$Common = @{ Revision = $Revision; MaxWorkers = $MaxWorkers }
if ($DestinationRoot) { $Common.DestinationRoot = $DestinationRoot }
if ($VerifyOnly) { $Common.VerifyOnly = $true }
if ($VerifyLfsSha256) { $Common.VerifyLfsSha256 = $true }
if ($IgnoreEnvironmentProxy) { $Common.IgnoreEnvironmentProxy = $true }
if ($DryRun) { $Common.DryRun = $true }

& (Join-Path $PSScriptRoot "download_qwen3_30b_a3b_instruct.ps1") @Common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $PSScriptRoot "download_qwen3_32b_judge.ps1") @Common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

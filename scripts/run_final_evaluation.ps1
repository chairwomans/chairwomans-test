param(
    [string]$Backbone = 'ViT-B/16',
    [string]$Datasets = 'all',
    [int]$BatchSize = 512,
    [int]$Seed = 42,
    [switch]$DryRun,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$entryScript = Join-Path $repoRoot 'run_final_evaluation.py'
$dataRoot = Join-Path $repoRoot 'data'

$arguments = @(
    $entryScript,
    '--data-root', $dataRoot,
    '--backbone', $Backbone,
    '--datasets', $Datasets,
    '--batch-size', $BatchSize,
    '--seed', $Seed
)
if ($DryRun) {
    $arguments += '--dry-run'
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Domain-hint evaluation failed with exit code $LASTEXITCODE."
}

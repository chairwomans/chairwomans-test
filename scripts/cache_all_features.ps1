param(
    [string]$Backbone = 'ViT-B/16',
    [int]$BatchSize = 128,
    [ValidateSet('test', 'val')]
    [string]$Split = 'test'
)

$ErrorActionPreference = 'Stop'

if ($Split -eq 'val') {
    $datasets = @(
        'caltech101', 'dtd', 'eurosat', 'fgvc_aircraft', 'food101',
        'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101'
    )
} else {
    $datasets = @(
        'caltech101', 'dtd', 'eurosat', 'fgvc_aircraft', 'food101',
        'imagenet', 'imagenetv2', 'imagenet_a', 'imagenet_r', 'imagenet_sketch',
        'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101'
    )
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$cacheScript = Join-Path $repoRoot 'scripts\cache_features.py'
$dataRoot = Join-Path $repoRoot 'data'
$cacheBackbone = $Backbone.Replace('/', '-')

foreach ($dataset in $datasets) {
    $cacheDirectory = Join-Path $dataRoot "cache\$cacheBackbone\$dataset"
    if ($Split -ne 'test') {
        $cacheDirectory = Join-Path $cacheDirectory $Split
    }
    $required = 'raw.pt', 'avg.pt', 'metadata.json'
    if (($required | ForEach-Object { Test-Path -LiteralPath (Join-Path $cacheDirectory $_) }) -notcontains $false) {
        Write-Output "Skipping ${dataset}: cache already complete."
        continue
    }

    Write-Output "Starting $dataset"
    & $python $cacheScript --data-root $dataRoot --dataset $dataset --backbone $Backbone --split $Split --batch-size $BatchSize
    if ($LASTEXITCODE -ne 0) {
        throw "Cache creation failed for $dataset with exit code $LASTEXITCODE."
    }
    Write-Output "Finished $dataset"
}

Write-Output "All $Backbone caches are complete."

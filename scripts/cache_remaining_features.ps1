$ErrorActionPreference = 'Stop'

$datasets = @(
    'caltech101',
    'dtd',
    'fgvc_aircraft',
    'food101',
    'imagenetv2',
    'imagenet_a',
    'imagenet_r',
    'imagenet_sketch',
    'oxford_flowers',
    'oxford_pets',
    'stanford_cars',
    'sun397',
    'ucf101'
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$cacheScript = Join-Path $repoRoot 'scripts\cache_features.py'
$dataRoot = Join-Path $repoRoot 'data'

foreach ($dataset in $datasets) {
    $cacheDirectory = Join-Path $dataRoot "cache\ViT-B-16\$dataset"
    if ((Test-Path -LiteralPath (Join-Path $cacheDirectory 'raw.pt')) -and
        (Test-Path -LiteralPath (Join-Path $cacheDirectory 'avg.pt')) -and
        (Test-Path -LiteralPath (Join-Path $cacheDirectory 'metadata.json'))) {
        Write-Output "Skipping ${dataset}: cache already complete."
        continue
    }

    Write-Output "Starting $dataset"
    & $python $cacheScript --data-root $dataRoot --dataset $dataset --backbone 'ViT-B/16' --batch-size 128
    if ($LASTEXITCODE -ne 0) {
        throw "Cache creation failed for $dataset with exit code $LASTEXITCODE."
    }
    Write-Output "Finished $dataset"
}

Write-Output 'All requested dataset caches are complete.'

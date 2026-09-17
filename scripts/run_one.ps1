param(
    # CoOp-style shared data root. The datasets themselves stay outside Git.
    [string]$DataRoot = "C:\datasets\coop",
    [Parameter(Mandatory=$true)][string]$Dataset,
    [ValidateSet("ViT-B/16", "ViT-B/32", "ViT-L/14")][string]$Backbone = "ViT-B/16",
    [int]$Seed = 42,
    [string]$Beta = "auto"
)

python run_comparison.py --data-root $DataRoot --dataset $Dataset --backbone $Backbone --seed $Seed --beta $Beta

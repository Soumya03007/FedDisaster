param(
  [int]$Count = 3,
  [int]$StartCid = 1,
  [int]$BatchSize = 32,
  [int]$TrainableBlocks = 1
)

# Resolve repo root (this script lives in scripts/)
$RepoRoot = Split-Path -Parent $PSScriptRoot

# Activate virtual environment if present (helps child pwsh sessions inherit env)
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
  & $VenvActivate
}

for ($i = 0; $i -lt $Count; $i++) {
  $cid = $StartCid + $i
  # Launch each client in a new PowerShell window, set working dir to repo root
  $command = "Set-Location `"$RepoRoot`"; python client.py --cid $cid --backbone efficientnet_b0 --train_backbone --trainable_blocks $TrainableBlocks --batch_size $BatchSize"
  Start-Process -FilePath "pwsh" -ArgumentList "-NoExit", "-Command", $command
}

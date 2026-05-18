param(
  [int]$Rounds = 5,
  [int]$Epochs = 1,
  [int]$BatchSize = 32,
  [string]$Address = "127.0.0.1:8080",
  [int]$RfEvalInterval = 2,
  [double]$FractionFit = 1.0
)

# Resolve repo root (this script lives in scripts/)
$RepoRoot = Split-Path -Parent $PSScriptRoot

# Activate virtual environment if present
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
  & $VenvActivate
}

# Ensure logs flush immediately
$env:PYTHONUNBUFFERED = "1"

Push-Location $RepoRoot
try {
  python server.py --backbone efficientnet_b0 --num_rounds $Rounds --epochs $Epochs --batch_size $BatchSize --address $Address --rf_eval_interval $RfEvalInterval --fraction_fit $FractionFit
} finally {
  Pop-Location
}

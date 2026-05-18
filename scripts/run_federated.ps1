param(
  [int]$NumClients = 3,
  [int]$NumRounds = 5,
  [int]$Epochs = 1,
  [int]$BatchSize = 32,
  [int]$MaxBatchesPerRound = 0,
  [string]$Backbone = "efficientnet_b0",
  [string]$ProgressiveUnfreezeSchedule = "",
  [ValidateSet("all", "sampled")]
  [string]$ClientSelection = "all",
  [int]$TrainableBlocks = 1,
  [int]$RfEvalInterval = 0,
  [double]$FractionFit = 1.0,
  [int]$MinFitClients = 0,
  [int]$MinAvailableClients = 0,
  [string]$Address = "127.0.0.1:8080"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
  & $VenvActivate
}

$env:PYTHONUNBUFFERED = "1"

Push-Location $RepoRoot
try {
  if ($ClientSelection -eq "all") {
    $FractionArg = 1.0
    $MinFitArg = $NumClients
    $MinAvailableArg = $NumClients
  } else {
    $FractionArg = $FractionFit
    $MinFitArg = if ($MinFitClients -gt 0) { $MinFitClients } else { [Math]::Max(2, [Math]::Round($NumClients * $FractionFit)) }
    $MinAvailableArg = if ($MinAvailableClients -gt 0) { $MinAvailableClients } else { $NumClients }
  }

  $ArgsList = @(
    "scripts/run_federated.py",
    "--num_clients", $NumClients,
    "--num_rounds", $NumRounds,
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--max_batches_per_round", $MaxBatchesPerRound,
    "--backbone", $Backbone,
    "--client_selection", $ClientSelection,
    "--trainable_blocks", $TrainableBlocks,
    "--rf_eval_interval", $RfEvalInterval,
    "--fraction_fit", $FractionArg,
    "--min_fit_clients", $MinFitArg,
    "--min_available_clients", $MinAvailableArg,
    "--address", $Address
  )

  if (-not [string]::IsNullOrWhiteSpace($ProgressiveUnfreezeSchedule)) {
    $ArgsList += @("--progressive_unfreeze_schedule", $ProgressiveUnfreezeSchedule)
  }

  python @ArgsList
} finally {
  Pop-Location
}

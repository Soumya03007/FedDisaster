param(
  [int]$NumClients = 3,
  [string[]]$Classes = @(
    "fire_disaster",
    "water_disaster",
    "land_disaster",
    "not_disaster"
  ),
  [string]$DataRoot = "data",
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Remove-IfExists([string]$Path) {
  if (Test-Path -LiteralPath $Path) {
    Remove-Item -LiteralPath $Path -Recurse -Force
  }
}

if ($Clean) {
  for ($i = 1; $i -le $NumClients; $i++) {
    Remove-IfExists (Join-Path $DataRoot ("client_{0}" -f $i))
  }
  Remove-IfExists (Join-Path $DataRoot "global_test")
}

for ($i = 1; $i -le $NumClients; $i++) {
  foreach ($split in @("train", "test")) {
    foreach ($cls in $Classes) {
      $p = Join-Path $DataRoot (Join-Path ("client_{0}" -f $i) (Join-Path $split $cls))
      New-Item -ItemType Directory -Force -Path $p | Out-Null
    }
  }
}

foreach ($cls in $Classes) {
  $p = Join-Path $DataRoot (Join-Path "global_test" $cls)
  New-Item -ItemType Directory -Force -Path $p | Out-Null
}

Write-Host "[OK] Created multi-class folder skeleton under '$DataRoot'."
Write-Host "Classes: $($Classes -join ', ')"
Write-Host "Clients: $NumClients"
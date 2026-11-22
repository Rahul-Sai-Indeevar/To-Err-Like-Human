param([switch]$Report)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$env:PYTHONPATH="$repo/src"
if(Test-Path -LiteralPath "$repo/work/cuda_runtime/torch"){$env:PYTHONPATH="$repo/work/cuda_runtime;$repo/src"}
$env:OMP_NUM_THREADS='3'
$env:MKL_NUM_THREADS='3'
$env:MPLCONFIGDIR="$repo/work/matplotlib"
$env:TEMP="$repo/work/tmp"
$env:TMP="$repo/work/tmp"
if($Report){& "$repo/.venv/Scripts/python.exe" scripts/report_spatial_frequency.py}
else{& "$repo/.venv/Scripts/python.exe" -u scripts/run_spatial_frequency.py --per-class 0 --out outputs/full_spatial_frequency}
if($LASTEXITCODE -ne 0){throw "Experiment failed: $LASTEXITCODE"}

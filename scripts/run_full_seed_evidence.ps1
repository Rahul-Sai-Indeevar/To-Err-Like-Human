param([ValidateSet('cache','benchmark','train','report','replay')][string]$Stage='train',[string]$Output='outputs/full_seed_replication')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$env:PYTHONPATH="$repo/src"
if($Stage -in @('benchmark','train','replay')){$env:PYTHONPATH="$repo/work/cuda_runtime;$repo/src"}
$env:TORCH_HOME="$repo/data/torch_cache"
$env:OMP_NUM_THREADS='3'
$env:MKL_NUM_THREADS='3'
$env:MPLCONFIGDIR="$repo/work/matplotlib"
$env:TEMP="$repo/work/tmp"
$env:TMP="$repo/work/tmp"
$py="$repo/.venv/Scripts/python.exe"
switch($Stage){
 'cache' {& $py -u scripts/run_full_seed_evidence.py --device cpu --cache-only --out $Output}
 'benchmark' {& $py -u scripts/run_full_seed_evidence.py --methods ce --seeds 99 --epochs 1 --skip-test --out outputs/full_seed_benchmark}
 'train' {& $py -u scripts/run_full_seed_evidence.py --out $Output}
 'replay' {& $py -u scripts/verify_full_checkpoints.py --study $Output}
 'report' {& $py scripts/report_full_seed_evidence.py --out $Output}
}
if($LASTEXITCODE -ne 0){throw "Full-data ${Stage} failed: $LASTEXITCODE"}

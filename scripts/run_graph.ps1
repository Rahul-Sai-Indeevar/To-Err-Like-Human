param([ValidateSet('features','probe','finetune','report','test')][string]$Stage='report')
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$env:PYTHONPATH = "$repo/src"
if (Test-Path -LiteralPath "$repo/work/cuda_runtime/torch") {
    $env:PYTHONPATH = "$repo/work/cuda_runtime;$repo/src"
}
$env:TORCH_HOME = "$repo/data/torch_cache"
$env:OMP_NUM_THREADS = '6'
$env:MKL_NUM_THREADS = '6'
$env:TEMP = "$repo/work/tmp"
$env:TMP = "$repo/work/tmp"
New-Item -ItemType Directory -Force -Path "$repo/work/tmp" | Out-Null
$py = "$repo/.venv/Scripts/python.exe"
switch ($Stage) {
    'features' { & $py -u -m ecc.graph_experiment --stage features }
    'probe' { & $py -u -m ecc.graph_experiment --stage probe --epochs 60 --seeds 0 1 2 --device cpu }
    'finetune' { & $py -u -m ecc.graph_experiment --stage finetune --epochs 6 --seeds 0 1 2 --batch-size 32 }
    'report' { & $py scripts/report_graph.py }
    'test' { & $py -m pytest -q --basetemp="$repo/work/pytest_final" }
}
if ($LASTEXITCODE -ne 0) { throw "Stage $Stage failed with exit code $LASTEXITCODE" }

param([switch]$Report)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$env:PYTHONPATH="$repo/work/cuda_runtime;$repo/src"
$env:TORCH_HOME="$repo/data/torch_cache"
$env:OMP_NUM_THREADS='3'
$env:MKL_NUM_THREADS='3'
$env:MPLCONFIGDIR="$repo/work/matplotlib"
$env:TEMP="$repo/work/tmp"
$env:TMP="$repo/work/tmp"
$py="$repo/.venv/Scripts/python.exe"
if($Report){
    & $py scripts/report_seed_evidence.py --out outputs/seed_evidence --bootstrap 1500
    if($LASTEXITCODE -ne 0){throw "Report failed: $LASTEXITCODE"}
}else{
    foreach($name in @('emoset','fi','artphoto')){
        $dataRoot=if($name -eq 'emoset'){'data/emoset_graph'}else{"data/seed_audit/$name"}
        & $py -u scripts/run_seed_evidence.py --data-root $dataRoot --out "outputs/seed_evidence/$name" --batch-size 64
        if($LASTEXITCODE -ne 0){throw "Study failed for ${name}: $LASTEXITCODE"}
    }
}

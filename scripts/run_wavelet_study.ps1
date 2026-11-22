param([ValidateSet('all','train','evaluate','benchmark','report','diagnostics')][string]$Stage='all',[string]$Output='outputs/wavelet_study')
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
# A temporary thread-level request prevents system sleep during this run only.
# It neither keeps the screen on nor changes the user's power plan.
Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class WaveletStudyPower { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
[void][WaveletStudyPower]::SetThreadExecutionState([uint32]2147483649)
try {
    if($Stage -eq 'report') {
        & $py -u scripts/report_wavelet_study.py --out $Output --bootstrap 1500
    } elseif($Stage -eq 'diagnostics') {
        & $py -u scripts/report_wavelet_diagnostics.py --out $Output
    } elseif($Stage -eq 'benchmark') {
        & $py -u scripts/run_wavelet_study.py --benchmark --out outputs/wavelet_benchmark
    } else {
        & $py -u scripts/run_wavelet_study.py --stage $Stage --out $Output
    }
    if($LASTEXITCODE -ne 0){throw "Wavelet ${Stage} failed: $LASTEXITCODE"}
    if($Stage -eq 'all') {
        & $py -u scripts/report_wavelet_diagnostics.py --out $Output
        if($LASTEXITCODE -ne 0){throw 'Independent condition audit failed'}
        & $py scripts/plot_wavelet_theory.py --out $Output
        if($LASTEXITCODE -ne 0){throw 'Theoretical figures failed'}
    }
} finally {
    [void][WaveletStudyPower]::SetThreadExecutionState([uint32]2147483648)
}

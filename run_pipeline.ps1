# ESM3 Nanobody CDR3 Optimization Pipeline
# Run with: .\run_pipeline.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ESM3 Nanobody CDR3 Optimization" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Determine Python executable
$Python = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".venv\Scripts\python.exe"
} elseif (Test-Path ".venv\bin\python") {
    $Python = ".venv\bin\python"
}

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    & python -m venv .venv
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r requirements.txt
}

# Run the pipeline
Write-Host "Running optimization pipeline..." -ForegroundColor Green
& $Python -m src.esm3_nanobody.cli run --config configs/default_config.json

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pipeline completed!" -ForegroundColor Green
Write-Host "Results saved to outputs/" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

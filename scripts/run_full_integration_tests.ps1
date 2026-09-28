$ErrorActionPreference = "Stop"

Write-Host "Starting 5x3 Integration Test..."
& "E:\EVOLVE\.venv\Scripts\python.exe" scripts/run_integration.py --levels 5
if ($LASTEXITCODE -ne 0) {
    Write-Host "5x3 Integration Test Failed!"
    exit $LASTEXITCODE
}

Write-Host "Starting 10x3 Integration Test..."
& "E:\EVOLVE\.venv\Scripts\python.exe" scripts/run_integration.py --levels 10
if ($LASTEXITCODE -ne 0) {
    Write-Host "10x3 Integration Test Failed!"
    exit $LASTEXITCODE
}

Write-Host "All tests completed successfully!"

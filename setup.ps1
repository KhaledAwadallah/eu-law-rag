# setup.ps1 - one-time environment setup for eulaw (Windows PowerShell)
# Run from the project root:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "[1/5] Checking Python..." -ForegroundColor Cyan
$py = $null
foreach ($candidate in @("py", "python")) {
    try {
        $v = & $candidate --version 2>&1
        if ($LASTEXITCODE -eq 0 -and "$v" -match "^Python 3") {
            $py = $candidate
            Write-Host "  Found: $v (command: $candidate)"
            break
        }
    } catch { }
}
if (-not $py) {
    Write-Host "  No working Python found." -ForegroundColor Red
    Write-Host "  Install it with:  winget install Python.Python.3.12" -ForegroundColor Yellow
    Write-Host "  (or from python.org - tick 'Add python.exe to PATH'), then re-run this script."
    exit 1
}

Write-Host "[2/5] Creating virtual environment (.venv)..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    & $py -m venv .venv
} else {
    Write-Host "  .venv already exists, skipping."
}

Write-Host "[3/5] Installing dependencies (PyTorch is ~2 GB - this takes a while)..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .

Write-Host "[4/5] Verifying installation..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -c "import chromadb, sentence_transformers; from eulaw import config; print('setup OK - chunk size:', config.CHUNK_SIZE)"
& .\.venv\Scripts\python.exe -m pytest -q

Write-Host "[5/5] Initializing git repository..." -ForegroundColor Cyan
if (-not (Test-Path ".git")) { git init | Out-Null }
$gitEmail = git config user.email
if (-not $gitEmail) {
    Write-Host "  Git identity not set. Run these once, then re-run setup.ps1:" -ForegroundColor Yellow
    Write-Host '    git config --global user.name  "Khaled Awadallah"'
    Write-Host '    git config --global user.email "your-email@example.com"'
} else {
    git add .
    git commit -m "Project scaffold: package structure, dependencies, config" | Out-Null
    Write-Host "  Initial commit created."
}

Write-Host ""
Write-Host "Done. In every new terminal, activate the environment with:" -ForegroundColor Green
Write-Host "  .venv\Scripts\activate"

# TOA Personal Copilot - Quick Start Script
# This script helps you start all required services

Write-Host "================================" -ForegroundColor Cyan
Write-Host "TOA Personal Copilot - Startup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check PostgreSQL
Write-Host "1. Checking PostgreSQL..." -ForegroundColor Yellow
try {
    $service = Get-Service postgresql-x64-15 -ErrorAction SilentlyContinue
    if ($service) {
        if ($service.Status -eq "Running") {
            Write-Host "   ✅ PostgreSQL is running" -ForegroundColor Green
        } else {
            Write-Host "   ⏳ Starting PostgreSQL..." -ForegroundColor Yellow
            Start-Service postgresql-x64-15
            Start-Sleep -Seconds 3
            Write-Host "   ✅ PostgreSQL started" -ForegroundColor Green
        }
    } else {
        Write-Host "   ⚠️  PostgreSQL service not found. Install PostgreSQL first." -ForegroundColor Red
        Write-Host "   Download from: https://www.postgresql.org/download/windows/" -ForegroundColor Magenta
        exit 1
    }
} catch {
    Write-Host "   ⚠️  Error checking PostgreSQL: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "2. Checking Python environment..." -ForegroundColor Yellow
$venvPath = "d:\JFF\TOA-Personal-Copilot\venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "   ✅ Virtual environment found" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Virtual environment not found" -ForegroundColor Red
    Write-Host "   Creating venv..." -ForegroundColor Yellow
    cd d:\JFF\TOA-Personal-Copilot
    python -m venv venv
    Write-Host "   ✅ Virtual environment created" -ForegroundColor Green
}

Write-Host ""
Write-Host "3. Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "   ✅ Node.js $nodeVersion found" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  Node.js not found. Download from: https://nodejs.org/" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "4. Checking Ollama..." -ForegroundColor Yellow
try {
    $ollamaVersion = ollama --version
    Write-Host "   ✅ Ollama found" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  Ollama not found. Download from: https://ollama.ai/" -ForegroundColor Red
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Ready to start services!" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📌 You need to open 4 separate PowerShell terminals and run:" -ForegroundColor Magenta
Write-Host ""
Write-Host "Terminal 1 (Ollama):" -ForegroundColor Yellow
Write-Host "  ollama serve" -ForegroundColor White
Write-Host ""
Write-Host "Terminal 2 (Backend):" -ForegroundColor Yellow
Write-Host "  cd d:\JFF\TOA-Personal-Copilot\backend" -ForegroundColor White
Write-Host "  ..\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  uvicorn main:app --reload --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "Terminal 3 (Frontend):" -ForegroundColor Yellow
Write-Host "  cd d:\JFF\TOA-Personal-Copilot\frontend" -ForegroundColor White
Write-Host "  npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "Terminal 4 (Database):" -ForegroundColor Yellow
Write-Host "  Optional - verify data with:" -ForegroundColor White
Write-Host "  psql -U fahad -d toa_copilot" -ForegroundColor White
Write-Host ""
Write-Host "Then open: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "💾 All data will be stored in PostgreSQL at:" -ForegroundColor Green
Write-Host "   Database: toa_copilot" -ForegroundColor White
Write-Host "   User: fahad" -ForegroundColor White
Write-Host "   Port: 5432" -ForegroundColor White
Write-Host ""

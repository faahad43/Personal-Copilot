#!/usr/bin/env pwsh
# TOA Personal Copilot - Local Setup Script
# This script sets up and runs the entire stack locally

param(
    [switch]$SkipVenv,
    [switch]$SkipNodeModules,
    [string]$BackendPort = "8000",
    [string]$FrontendPort = "3000"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TOA Personal Copilot - Local Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "[1] Checking prerequisites..." -ForegroundColor Yellow

$hasOllama = $null -ne (Get-Command ollama -ErrorAction SilentlyContinue)
$hasNode = $null -ne (Get-Command node -ErrorAction SilentlyContinue)
$hasPython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)

if (-not $hasOllama) {
    Write-Host "⚠️  Ollama not found. Please install from https://ollama.ai" -ForegroundColor Red
}

if (-not $hasNode) {
    Write-Host "⚠️  Node.js not found. Please install from https://nodejs.org" -ForegroundColor Red
}

if (-not $hasPython) {
    Write-Host "⚠️  Python not found. Please install from https://python.org" -ForegroundColor Red
}

if ($hasOllama -and $hasNode -and $hasPython) {
    Write-Host "✅ All prerequisites found" -ForegroundColor Green
} else {
    Write-Host "Please install missing prerequisites and try again." -ForegroundColor Red
    exit 1
}

# Backend Setup
Write-Host ""
Write-Host "[2] Setting up backend..." -ForegroundColor Yellow

Push-Location backend

if (-not $SkipVenv) {
    if (-not (Test-Path venv)) {
        Write-Host "Creating virtual environment..."
        python -m venv venv
    }
    
    & ".\venv\Scripts\Activate.ps1"
    
    Write-Host "Installing Python dependencies..."
    pip install -r requirements.txt --quiet
}

Pop-Location
Write-Host "✅ Backend setup complete" -ForegroundColor Green

# Frontend Setup
Write-Host ""
Write-Host "[3] Setting up frontend..." -ForegroundColor Yellow

Push-Location frontend

if (-not $SkipNodeModules) {
    if (-not (Test-Path node_modules)) {
        Write-Host "Installing Node dependencies..."
        npm install --silent
    }
}

Pop-Location
Write-Host "✅ Frontend setup complete" -ForegroundColor Green

# Download models
Write-Host ""
Write-Host "[4] Preparing Ollama models..." -ForegroundColor Yellow
Write-Host "This may take a few minutes on first run..."

& ollama pull llama3.2:3b 2>&1 | Out-Null
& ollama pull mxbai-embed-large 2>&1 | Out-Null

Write-Host "✅ Models ready" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete! Ready to start." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the application, open 3 terminals and run:" -ForegroundColor White
Write-Host ""
Write-Host "Terminal 1 (Ollama server):" -ForegroundColor Green
Write-Host "  ollama serve" -ForegroundColor Gray
Write-Host ""
Write-Host "Terminal 2 (FastAPI backend):" -ForegroundColor Green
Write-Host "  cd backend" -ForegroundColor Gray
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  uvicorn main:app --reload --port $BackendPort" -ForegroundColor Gray
Write-Host ""
Write-Host "Terminal 3 (Next.js frontend):" -ForegroundColor Green
Write-Host "  cd frontend" -ForegroundColor Gray
Write-Host "  npm run dev -- -p $FrontendPort" -ForegroundColor Gray
Write-Host ""
Write-Host "Then open: http://localhost:$FrontendPort" -ForegroundColor Cyan
Write-Host ""

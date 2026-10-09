# ForensIQ ML Backend Launcher
# Kills any existing process on port 8000, then starts api_server.py

Write-Host "Checking for existing process on port 8000..." -ForegroundColor Cyan

$existing = netstat -ano | Select-String ":8000 " | Select-String "LISTENING"
if ($existing) {
    $pid = ($existing -split '\s+')[-1].Trim()
    Write-Host "Killing existing process PID $pid..." -ForegroundColor Yellow
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    Write-Host "Process stopped." -ForegroundColor Green
} else {
    Write-Host "Port 8000 is free." -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting ForensIQ V2.1 API server..." -ForegroundColor Cyan
python api_server.py

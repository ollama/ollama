#!/usr/bin/env pwsh
# Validation script for Context MCP Integration
# Run this to verify your setup is correct

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "Context MCP Integration - Setup Validation" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check 1: .env file exists
Write-Host "[1/7] Checking .env file..." -NoNewline
if (Test-Path ".env") {
    Write-Host " ✓" -ForegroundColor Green
} else {
    Write-Host " ✗" -ForegroundColor Red
    Write-Host "      Missing .env file. Copy from .env.example" -ForegroundColor Yellow
    $allGood = $false
}

# Check 2: Backend files
Write-Host "[2/7] Checking backend files..." -NoNewline
if ((Test-Path "api-gateway/main.py") -and (Select-String -Path "api-gateway/main.py" -Pattern "build_library_docs_context" -Quiet)) {
    Write-Host " ✓" -ForegroundColor Green
} else {
    Write-Host " ✗" -ForegroundColor Red
    Write-Host "      api-gateway/main.py missing or incomplete" -ForegroundColor Yellow
    $allGood = $false
}

# Check 3: Docker compose config
Write-Host "[3/7] Checking docker-compose.yml..." -NoNewline
if ((Test-Path "docker-compose.yml") -and (Select-String -Path "docker-compose.yml" -Pattern "DOCS_CONTEXT_ENABLED" -Quiet)) {
    Write-Host " ✓" -ForegroundColor Green
} else {
    Write-Host " ✗" -ForegroundColor Red
    Write-Host "      docker-compose.yml missing DOCS_CONTEXT_ENABLED" -ForegroundColor Yellow
    $allGood = $false
}

# Check 4: Documentation files
Write-Host "[4/7] Checking documentation..." -NoNewline
if ((Test-Path "CONTEXT_MCP_INTEGRATION.md") -and (Test-Path "IMPLEMENTATION_SUMMARY.md")) {
    Write-Host " ✓" -ForegroundColor Green
} else {
    Write-Host " ✗" -ForegroundColor Red
    Write-Host "      Documentation files missing" -ForegroundColor Yellow
    $allGood = $false
}

# Check 5: Cursor MCP config
Write-Host "[5/7] Checking Cursor MCP config..." -NoNewline
$cursorMcpPath = "$env:USERPROFILE\.cursor\mcp.json"
if (Test-Path $cursorMcpPath) {
    if (Select-String -Path $cursorMcpPath -Pattern "context7" -Quiet) {
        Write-Host " ✓" -ForegroundColor Green
    } else {
        Write-Host " ⚠" -ForegroundColor Yellow
        Write-Host "      Cursor mcp.json exists but context7 not configured" -ForegroundColor Yellow
    }
} else {
    Write-Host " ⚠" -ForegroundColor Yellow
    Write-Host "      Cursor mcp.json not found (optional)" -ForegroundColor Yellow
}

# Check 6: Cursor rules
Write-Host "[6/7] Checking Cursor rules..." -NoNewline
if (Test-Path ".cursor\rules\documentation-lookup.md") {
    Write-Host " ✓" -ForegroundColor Green
} else {
    Write-Host " ⚠" -ForegroundColor Yellow
    Write-Host "      Cursor rule not found (optional)" -ForegroundColor Yellow
}

# Check 7: Environment variables in .env
Write-Host "[7/7] Checking .env configuration..." -NoNewline
if (Test-Path ".env") {
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "DOCS_CONTEXT_ENABLED") {
        Write-Host " ✓" -ForegroundColor Green
        
        # Show current settings
        Write-Host ""
        Write-Host "Current configuration:" -ForegroundColor Cyan
        
        if ($envContent -match "DOCS_CONTEXT_ENABLED=(\w+)") {
            $enabled = $matches[1]
            $color = if ($enabled -eq "true") { "Green" } else { "Yellow" }
            Write-Host "  - DOCS_CONTEXT_ENABLED: " -NoNewline
            Write-Host $enabled -ForegroundColor $color
        }
        
        if ($envContent -match "DOCS_CONTEXT_MAX_CHARS=(\d+)") {
            Write-Host "  - DOCS_CONTEXT_MAX_CHARS: $($matches[1])" -ForegroundColor Gray
        }
        
        if ($envContent -match "CONTEXT7_API_KEY=(.+)") {
            $key = $matches[1]
            if ($key -and $key -ne "") {
                Write-Host "  - CONTEXT7_API_KEY: " -NoNewline
                Write-Host "configured ✓" -ForegroundColor Green
            } else {
                Write-Host "  - CONTEXT7_API_KEY: " -NoNewline
                Write-Host "not set (optional)" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host " ✗" -ForegroundColor Red
        Write-Host "      Missing DOCS_CONTEXT configuration in .env" -ForegroundColor Yellow
        $allGood = $false
    }
}

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

if ($allGood) {
    Write-Host "Setup validation: " -NoNewline
    Write-Host "PASSED ✓" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Start the stack: docker compose up -d --build" -ForegroundColor Gray
    Write-Host "  2. Check health: curl.exe http://localhost:8080/api/health" -ForegroundColor Gray
    Write-Host "  3. Enable docs (optional): Edit .env and set DOCS_CONTEXT_ENABLED=true" -ForegroundColor Gray
    Write-Host "  4. See CONTEXT_MCP_INTEGRATION.md for usage examples" -ForegroundColor Gray
} else {
    Write-Host "Setup validation: " -NoNewline
    Write-Host "FAILED ✗" -ForegroundColor Red
    Write-Host ""
    Write-Host "Fix the issues above and run this script again." -ForegroundColor Yellow
}

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

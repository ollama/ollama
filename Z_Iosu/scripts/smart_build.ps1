#Requires -Version 5.1
<#
.SYNOPSIS
    Smart incremental build script for Ollama - detects changes and builds only what's necessary.

.DESCRIPTION
    Automatically detects which components have changed since last build and executes
    only the necessary build steps, saving 5-8 minutes on incremental builds.
    
    Detection logic:
    - CUDA files (*.cu) -> buildCUDA13
    - Vulkan files (ggml-vulkan/*.cpp) -> buildVulkan
    - GGML files (ggml/src/*.c, *.cpp, *.h) -> buildCPU
    - Go files (*.go) -> buildOllama
    - App files (app/*.go) -> buildApp
    - Always runs: gatherDependencies, buildInstaller (fast steps)

.PARAMETER Force
    Force full rebuild of all components, ignoring change detection.

.EXAMPLE
    .\smart_build.ps1
    Automatically detect changes and build only necessary components (verbose mode enabled by default).

.EXAMPLE
    .\smart_build.ps1 -Force
    Force full rebuild of all components.

.EXAMPLE
    .\smart_build.ps1 -Force
    Force full rebuild of all components.
#>

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Always enable verbose mode by default
$VerbosePreference = "Continue"

# ============================================================================
# Configuration
# ============================================================================

$script:BuildStateFile = "Z_Iosu\.build_state.json"
$script:BuildScriptPath = "Z_Iosu\scripts\build_windows.ps1"
$script:ProjectRoot = $PWD.Path

# File patterns for change detection
$script:ComponentPatterns = @{
    "CPU" = @(
        "ml/backend/ggml/ggml/src/*.c",
        "ml/backend/ggml/ggml/src/*.cpp",
        "ml/backend/ggml/ggml/src/*.h",
        "ml/backend/ggml/ggml/src/ggml-cpu/*.c",
        "ml/backend/ggml/ggml/src/ggml-cpu/*.cpp",
        "ml/backend/ggml/ggml/src/ggml-cpu/*.h"
    )
    "CUDA" = @(
        "ml/backend/ggml/ggml/src/ggml-cuda/*.cu",
        "ml/backend/ggml/ggml/src/ggml-cuda/*.cuh",
        "ml/backend/ggml/ggml/src/ggml-cuda/*.h"
    )
    "Vulkan" = @(
        "ml/backend/ggml/ggml/src/ggml-vulkan/*.cpp",
        "ml/backend/ggml/ggml/src/ggml-vulkan/*.h",
        "ml/backend/ggml/ggml/src/ggml-vulkan/*.hpp"
    )
    "Ollama" = @(
        "*.go",
        "cmd/*.go",
        "server/*.go",
        "api/*.go",
        "llm/*.go",
        "runner/*.go"
    )
    "App" = @(
        "app/*.go",
        "app/lifecycle/*.go",
        "app/tray/*.go"
    )
}

# ============================================================================
# Helper Functions
# ============================================================================

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-ColorOutput "===================================================================" "Cyan"
    Write-ColorOutput " $Title" "Cyan"
    Write-ColorOutput "===================================================================" "Cyan"
}

function Get-FileHash256 {
    param([string]$Path)
    
    if (-not (Test-Path $Path)) {
        return $null
    }
    
    $hash = Get-FileHash -Path $Path -Algorithm SHA256
    return $hash.Hash
}

function Get-FilesLastWriteTime {
    param([string[]]$Patterns)
    
    $maxTime = [DateTime]::MinValue
    $files = @()
    
    foreach ($pattern in $Patterns) {
        $matchedFiles = Get-ChildItem -Path $script:ProjectRoot -Filter (Split-Path $pattern -Leaf) -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*$($pattern.Replace('/', '\'))" }
        
        $files += $matchedFiles
        
        foreach ($file in $matchedFiles) {
            if ($file.LastWriteTime -gt $maxTime) {
                $maxTime = $file.LastWriteTime
            }
        }
    }
    
    return @{
        MaxTime = $maxTime
        Files = $files
        Count = $files.Count
    }
}

function Test-ComponentChanged {
    param(
        [string]$ComponentName,
        [DateTime]$LastBuildTime
    )
    
    $patterns = $script:ComponentPatterns[$ComponentName]
    if (-not $patterns) {
        return $false
    }
    
    $result = Get-FilesLastWriteTime -Patterns $patterns
    
    # Always show file information (verbose mode is always on)
    Write-ColorOutput ("  " + $ComponentName + ": " + $result.Count + " files, last modified: " + $result.MaxTime) "DarkGray"
    
    return ($result.MaxTime -gt $LastBuildTime)
}

function Get-BuildState {
    if (-not (Test-Path $script:BuildStateFile)) {
        return @{
            LastBuild = [DateTime]::MinValue
            Components = @{}
        }
    }
    
    try {
        $json = Get-Content $script:BuildStateFile -Raw | ConvertFrom-Json
        return @{
            LastBuild = [DateTime]::Parse($json.LastBuild)
            Components = $json.Components
        }
    }
    catch {
        Write-ColorOutput "Warning: Could not read build state, assuming full rebuild needed" "Yellow"
        return @{
            LastBuild = [DateTime]::MinValue
            Components = @{}
        }
    }
}

function Save-BuildState {
    param([hashtable]$Components)
    
    $state = @{
        LastBuild = (Get-Date).ToString("o")
        Components = $Components
    }
    
    $stateDir = Split-Path $script:BuildStateFile -Parent
    if (-not (Test-Path $stateDir)) {
        New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
    }
    
    $state | ConvertTo-Json -Depth 10 | Set-Content $script:BuildStateFile
}

function Invoke-BuildStep {
    param(
        [string]$StepName,
        [switch]$Required
    )
    
    Write-ColorOutput ("  [>] Executing: " + $StepName) "Green"
    
    $startTime = Get-Date
    
    # If verbose mode, show all output. Otherwise, capture and suppress
    if ($script:ShowVerbose) {
        Write-Host ""
        Write-Host "================================================================" -ForegroundColor DarkCyan
        Write-Host " BUILDING: $StepName" -ForegroundColor Cyan
        Write-Host "================================================================" -ForegroundColor DarkCyan
        Write-Host ""
        
        # Execute build script - output will be inherited from parent process
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "powershell.exe"
        $psi.Arguments = "-ExecutionPolicy Bypass -File `"$script:BuildScriptPath`" $StepName"
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $false
        $psi.RedirectStandardError = $false
        
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $psi
        $process.Start() | Out-Null
        $process.WaitForExit()
        $exitCode = $process.ExitCode
        
        Write-Host ""
        Write-Host "================================================================" -ForegroundColor DarkCyan
        Write-Host " FINISHED: $StepName (exit code: $exitCode)" -ForegroundColor Cyan
        Write-Host "================================================================" -ForegroundColor DarkCyan
        Write-Host ""
    }
    else {
        # Execute quietly, only show errors
        $output = & powershell -ExecutionPolicy Bypass -File $script:BuildScriptPath $StepName 2>&1
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -ne 0) {
            Write-Host ""
            Write-ColorOutput "Build output (last 50 lines):" "Yellow"
            $output | Select-Object -Last 50 | ForEach-Object { Write-Host $_ }
            Write-Host ""
        }
    }
    
    $duration = (Get-Date) - $startTime
    
    if ($exitCode -ne 0) {
        Write-ColorOutput ("  X Failed: " + $StepName + " (exit code: " + $exitCode + ")") "Red"
        if ($Required) {
            throw "Build step '$StepName' failed with exit code $exitCode"
        }
        return $false
    }
    else {
        $durationSeconds = [int]$duration.TotalSeconds
        Write-ColorOutput ("  OK Completed: " + $StepName + " (" + $durationSeconds + "s)") "Green"
        return $true
    }
}

# ============================================================================
# Main Build Logic
# ============================================================================

function Start-SmartBuild {
    Write-Section "Ollama Smart Incremental Build"
    
    # Always enable verbose mode
    $script:ShowVerbose = $true
    
    # Ensure VERSION environment variable is set
    if (-not $env:VERSION) {
        $env:VERSION = "0.12.7.99"
        Write-ColorOutput "VERSION not set, using default: $env:VERSION" "Yellow"
    }
    
    Write-ColorOutput "Version: $env:VERSION" "White"
    Write-ColorOutput "Project: $script:ProjectRoot" "White"
    Write-ColorOutput "Build Script: $script:BuildScriptPath" "White"
    Write-ColorOutput "Verbose Mode: ALWAYS ENABLED (showing all build output)" "Cyan"
    
    # Load previous build state
    $buildState = Get-BuildState
    $lastBuildTime = $buildState.LastBuild
    
    if ($Force) {
        Write-ColorOutput "`n⚠ Force mode enabled - full rebuild" "Yellow"
        $lastBuildTime = [DateTime]::MinValue
    }
    else {
        Write-ColorOutput "`nLast build: $lastBuildTime" "White"
    }
    
    # Detect changes
    Write-Section "Detecting Changes"
    
    $buildSteps = @()
    $changedComponents = @()
    
    # Check each component
    if ($Force -or (Test-ComponentChanged -ComponentName "CPU" -LastBuildTime $lastBuildTime)) {
        $buildSteps += "buildCPU"
        $changedComponents += "CPU (GGML)"
        Write-ColorOutput "  [+] CPU (GGML) - Changed" "Yellow"
    }
    else {
        Write-ColorOutput "  [ ] CPU (GGML) - No changes" "DarkGray"
    }
    
    if ($Force -or (Test-ComponentChanged -ComponentName "CUDA" -LastBuildTime $lastBuildTime)) {
        $buildSteps += "buildCUDA13"
        $changedComponents += "CUDA 13"
        Write-ColorOutput "  [+] CUDA 13 - Changed" "Yellow"
    }
    else {
        Write-ColorOutput "  [ ] CUDA 13 - No changes" "DarkGray"
    }
    
    if ($Force -or (Test-ComponentChanged -ComponentName "Vulkan" -LastBuildTime $lastBuildTime)) {
        $buildSteps += "buildVulkan"
        $changedComponents += "Vulkan"
        Write-ColorOutput "  [+] Vulkan - Changed" "Yellow"
    }
    else {
        Write-ColorOutput "  [ ] Vulkan - No changes" "DarkGray"
    }
    
    # Check if ollama.exe exists in distribution directory
    $ollamaExeExists = Test-Path (Join-Path $ProjectRoot "dist\windows-amd64\ollama.exe")
    
    if ($Force -or (Test-ComponentChanged -ComponentName "Ollama" -LastBuildTime $lastBuildTime) -or -not $ollamaExeExists) {
        $buildSteps += "buildOllama"
        $changedComponents += "Ollama (Go)"
        if (-not $ollamaExeExists) {
            Write-ColorOutput "  [+] Ollama (Go) - Required (ollama.exe missing)" "Yellow"
        } else {
            Write-ColorOutput "  [+] Ollama (Go) - Changed" "Yellow"
        }
    }
    else {
        Write-ColorOutput "  [ ] Ollama (Go) - No changes" "DarkGray"
    }
    
    # Check if app.exe exists in distribution directory
    $appExeExists = Test-Path (Join-Path $ProjectRoot "dist\windows-amd64-app.exe")
    
    if ($Force -or (Test-ComponentChanged -ComponentName "App" -LastBuildTime $lastBuildTime) -or -not $appExeExists) {
        $buildSteps += "buildApp"
        $changedComponents += "App (GUI)"
        if (-not $appExeExists) {
            Write-ColorOutput "  [+] App (GUI) - Required (app.exe missing)" "Yellow"
        } else {
            Write-ColorOutput "  [+] App (GUI) - Changed" "Yellow"
        }
    }
    else {
        Write-ColorOutput "  [ ] App (GUI) - No changes" "DarkGray"
    }
    
    # Always run these (fast steps)
    $buildSteps += "gatherDependencies"
    $buildSteps += "buildInstaller"
    
    # Summary
    Write-Section "Build Plan"
    
    if ($changedComponents.Count -eq 0) {
        Write-ColorOutput "  [i] No changes detected in source files" "Cyan"
        Write-ColorOutput "  [i] Will only update dependencies and rebuild installer" "Cyan"
    }
    else {
        Write-ColorOutput "  Changed components:" "White"
        foreach ($component in $changedComponents) {
            Write-ColorOutput ("    * " + $component) "Yellow"
        }
    }
    
    Write-ColorOutput "`n  Build steps to execute:" "White"
    foreach ($step in $buildSteps) {
        Write-ColorOutput ("    1. " + $step) "Cyan"
    }
    
    # Estimate time savings
    $allSteps = @("buildCPU", "buildCUDA13", "buildVulkan", "buildOllama", "buildApp")
    $skippedSteps = $allSteps | Where-Object { $_ -notin $buildSteps }
    
    if ($null -ne $skippedSteps -and @($skippedSteps).Count -gt 0) {
        $skippedCount = @($skippedSteps).Count
        $minSaved = $skippedCount * 2
        $maxSaved = $skippedCount * 3
        Write-ColorOutput ("`n  [!] Skipping " + $skippedCount + " unchanged components") "Green"
        Write-ColorOutput ("  [!] Estimated time saved: " + $minSaved + "-" + $maxSaved + " minutes") "Green"
    }
    
    # Execute build
    Write-Section "Building"
    
    $buildStartTime = Get-Date
    $successCount = 0
    $failCount = 0
    
    foreach ($step in $buildSteps) {
        $isRequired = $step -in @("buildOllama", "buildInstaller")
        $success = Invoke-BuildStep -StepName $step -Required:$isRequired
        
        if ($success) {
            $successCount++
        }
        else {
            $failCount++
        }
    }
    
    $buildDuration = (Get-Date) - $buildStartTime
    
    # Save build state
    $componentState = @{}
    foreach ($component in $script:ComponentPatterns.Keys) {
        $result = Get-FilesLastWriteTime -Patterns $script:ComponentPatterns[$component]
        $componentState[$component] = @{
            LastModified = $result.MaxTime.ToString("o")
            FileCount = $result.Count
        }
    }
    Save-BuildState -Components $componentState
    
    # Summary
    Write-Section "Build Summary"
    
    $totalMinutes = [int]$buildDuration.TotalMinutes
    $totalSeconds = $buildDuration.Seconds
    Write-ColorOutput ("  Total time: " + $totalMinutes + "m " + $totalSeconds + "s") "White"
    Write-ColorOutput ("  Successful steps: " + $successCount) "Green"
    
    if ($failCount -gt 0) {
        Write-ColorOutput ("  Failed steps: " + $failCount) "Red"
    }
    
    if ($successCount -eq $buildSteps.Count) {
        Write-ColorOutput "`n  [OK] Build completed successfully!" "Green"
        Write-ColorOutput "  [>>] Installer: dist\OllamaSetup.exe" "Cyan"
        return 0
    }
    else {
        Write-ColorOutput "`n  [!!] Build completed with errors" "Yellow"
        return 1
    }
}

# ============================================================================
# Entry Point
# ============================================================================

try {
    $exitCode = Start-SmartBuild
    exit $exitCode
}
catch {
    Write-ColorOutput "`n[ERROR] Build failed with error:" "Red"
    Write-ColorOutput $_.Exception.Message "Red"
    Write-ColorOutput $_.ScriptStackTrace "DarkGray"
    exit 1
}

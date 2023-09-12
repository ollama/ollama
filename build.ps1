# Define a function to print in red color
function Print-Red {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Red
}

# Check if 'go' is installed
$goCommand = Get-Command go -ErrorAction SilentlyContinue
if (-not $goCommand) {
    Print-Red "Error: 'go' is not installed."
    exit
}

# Check if 'cmake' is installed
$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmakeCommand) {
    Print-Red "Error: 'cmake' is not installed."
    exit
}

# If both are installed, execute the following commands:
try {
    go generate ./...
    go build -ldflags '-linkmode external -extldflags "-static"' .
    Write-Host "Successfully build ollama!" -ForegroundColor Green
} catch {
    Print-Red "Error: Failed to build ollama."
    exit
}

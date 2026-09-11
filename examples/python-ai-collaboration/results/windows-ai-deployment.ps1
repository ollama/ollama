[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Branch,

    [ValidateNotNullOrEmpty()]
    [string]$VolumeLabel = "Crucial X9",

    [ValidateNotNullOrEmpty()]
    [string]$ExpectedDiskModel = "Micron CT1000X9SSD9",

    [ValidateNotNullOrEmpty()]
    [string]$DestinationName = "ollama",

    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryUrl = "https://github.com/AKB0700/ollama.git"
$expectedRepository = "akb0700/ollama"

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed with exit code $LASTEXITCODE`: git $($Arguments -join ' ')"
    }
}

function Get-RepositoryIdentity {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    $identity = $Url.Trim()
    $identity = $identity -replace "^https://github\.com/", ""
    $identity = $identity -replace "^git@github\.com:", ""
    $identity = $identity.TrimEnd("/")
    $identity = $identity -replace "\.git$", ""
    return $identity.ToLowerInvariant()
}

try {
    Write-Host "Validating hardware..." -ForegroundColor Cyan

    $cpuNames = @(Get-CimInstance Win32_Processor | ForEach-Object { $_.Name.Trim() })
    if ($cpuNames.Count -eq 0) {
        throw "Windows did not report a CPU through Win32_Processor."
    }

    $computer = Get-CimInstance Win32_ComputerSystem
    $ramGB = [math]::Round($computer.TotalPhysicalMemory / 1GB, 2)
    Write-Host "CPU: $($cpuNames -join '; ')"
    Write-Host "RAM: $ramGB GB"

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        throw "nvidia-smi was not found. Install or repair the NVIDIA driver before deployment."
    }

    $gpuRows = @(
        & $nvidiaSmi.Source `
            "--query-gpu=name,memory.total,driver_version" `
            "--format=csv,noheader"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed with exit code $LASTEXITCODE. GPU acceleration could not be verified."
    }

    $gpuNames = @($gpuRows | ForEach-Object { ($_ -split ",", 2)[0].Trim() })
    if ($gpuNames -notcontains "NVIDIA GeForce RTX 5070 Ti") {
        throw "Required NVIDIA GeForce RTX 5070 Ti was not found. Detected: $($gpuNames -join '; ')"
    }
    Write-Host "GPU: $($gpuRows -join '; ')"

    Write-Host "Locating external SSD..." -ForegroundColor Cyan
    $volumes = @(
        Get-Volume | Where-Object { $_.FileSystemLabel -ceq $VolumeLabel }
    )
    if ($volumes.Count -eq 0) {
        throw "No volume has the exact label '$VolumeLabel'."
    }
    if ($volumes.Count -gt 1) {
        throw "Multiple volumes have the label '$VolumeLabel'. Assign unique labels before deployment."
    }

    $volume = $volumes[0]
    $driveLetter = [string]$volume.DriveLetter
    if ($driveLetter -notmatch "^[A-Za-z]$") {
        throw "Volume '$VolumeLabel' does not have a valid drive letter."
    }
    if ([string]$volume.HealthStatus -ne "Healthy") {
        throw "Volume '$VolumeLabel' is not healthy. Reported status: $($volume.HealthStatus)"
    }

    $driveRoot = "$($driveLetter.ToUpperInvariant()):\"
    $repositoryPath = Join-Path -Path $driveRoot -ChildPath $DestinationName
    $gitRepositoryArguments = @("-c", "safe.directory=$repositoryPath")
    Write-Host "SSD: $VolumeLabel at $driveRoot"

    $physicalDisks = @(
        Get-Partition -DriveLetter $driveLetter |
            Get-Disk
    )
    if ($physicalDisks.Count -ne 1) {
        throw "Expected one physical disk for $driveRoot, found $($physicalDisks.Count)."
    }

    $physicalDisk = $physicalDisks[0]
    if ([string]$physicalDisk.FriendlyName -cne $ExpectedDiskModel) {
        throw "Volume '$VolumeLabel' is on '$($physicalDisk.FriendlyName)', not '$ExpectedDiskModel'."
    }
    if ([string]$physicalDisk.BusType -ne "USB") {
        throw "Disk '$ExpectedDiskModel' is connected through '$($physicalDisk.BusType)', not USB."
    }
    if ([string]$physicalDisk.HealthStatus -ne "Healthy") {
        throw "Disk '$ExpectedDiskModel' is not healthy. Reported status: $($physicalDisk.HealthStatus)"
    }
    Write-Host "Physical disk: $($physicalDisk.FriendlyName) ($($physicalDisk.BusType))"

    $gitCommand = Get-Command git -ErrorAction SilentlyContinue
    if ($null -eq $gitCommand) {
        throw "Git was not found in PATH."
    }

    & git check-ref-format --branch $Branch *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "'$Branch' is not a valid Git branch name."
    }

    if ($ValidateOnly) {
        Write-Host ""
        Write-Host "Validation complete; deployment was not started." -ForegroundColor Green
        Write-Host "Drive: $driveRoot"
        Write-Host "Repository: $repositoryPath"
        Write-Host "Branch: $Branch"
        exit 0
    }

    Write-Host "Deploying repository..." -ForegroundColor Cyan
    if (Test-Path -LiteralPath $repositoryPath) {
        if (-not (Test-Path -LiteralPath $repositoryPath -PathType Container)) {
            throw "Destination exists but is not a directory: $repositoryPath"
        }

        $insideWorkTree = & git @gitRepositoryArguments -C $repositoryPath rev-parse --is-inside-work-tree 2>$null
        if ($LASTEXITCODE -ne 0 -or $insideWorkTree -ne "true") {
            throw "Destination is not a Git working tree: $repositoryPath"
        }

        $originUrl = & git @gitRepositoryArguments -C $repositoryPath config --get remote.origin.url
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($originUrl)) {
            throw "The existing repository has no origin remote."
        }
        if ((Get-RepositoryIdentity $originUrl) -ne $expectedRepository) {
            throw "Existing origin '$originUrl' is not $repositoryUrl."
        }

        $changes = @(
            git @gitRepositoryArguments -C $repositoryPath status --porcelain
        )
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to inspect the existing working tree."
        }
        if ($changes.Count -gt 0) {
            throw "The existing repository has uncommitted changes. Commit, stash, or remove them before deployment."
        }

        Invoke-Git ($gitRepositoryArguments + @("-C", $repositoryPath, "fetch", "--prune", "origin"))
        & git @gitRepositoryArguments -C $repositoryPath show-ref --verify --quiet "refs/remotes/origin/$Branch"
        if ($LASTEXITCODE -ne 0) {
            throw "Remote branch 'origin/$Branch' does not exist."
        }

        & git @gitRepositoryArguments -C $repositoryPath show-ref --verify --quiet "refs/heads/$Branch"
        if ($LASTEXITCODE -eq 0) {
            Invoke-Git ($gitRepositoryArguments + @("-C", $repositoryPath, "switch", $Branch))
            Invoke-Git ($gitRepositoryArguments + @("-C", $repositoryPath, "merge", "--ff-only", "origin/$Branch"))
        }
        else {
            Invoke-Git ($gitRepositoryArguments + @(
                "-C", $repositoryPath, "switch", "--track", "-c", $Branch,
                "origin/$Branch"
            ))
        }
    }
    else {
        Invoke-Git @(
            "clone", "--branch", $Branch, "--single-branch", "--",
            $repositoryUrl, $repositoryPath
        )
    }

    $commit = & git @gitRepositoryArguments -C $repositoryPath rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($commit)) {
        throw "Unable to read the deployed commit SHA."
    }

    Write-Host ""
    Write-Host "Deployment complete." -ForegroundColor Green
    Write-Host "Drive: $driveRoot"
    Write-Host "Repository: $repositoryPath"
    Write-Host "Branch: $Branch"
    Write-Host "Commit: $commit"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}

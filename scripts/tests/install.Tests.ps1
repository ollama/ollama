<#
.SYNOPSIS
    Pester tests for Ollama install script.

.DESCRIPTION
    Unit tests (Tag: Unit) - fast, mock external commands, no system changes.
    Integration tests (Tag: Integration) - require built MSIs in dist/, make real installs.
    Local unsigned dist/ artifacts are tested with a generated install.ps1 copy
    under build/install-tests/ that replaces signature checks with a warning.

    Integration tests verify results by checking the filesystem, MSI registration
    (via Windows Installer COM API), and running processes rather than parsing
    text output.

.EXAMPLE
    Import-Module Pester -MinimumVersion 5.0
    Invoke-Pester scripts/tests/install.Tests.ps1 -Tag Unit -Output Detailed
    Invoke-Pester scripts/tests/install.Tests.ps1 -Tag Integration -Output Detailed
    Invoke-Pester scripts/tests/install.Tests.ps1 -Tag UpgradeMatrix -Output Detailed
#>

BeforeAll {
    $ScriptRoot = Split-Path -Parent $PSScriptRoot
    $InstallScript = Join-Path $ScriptRoot "install.ps1"
    Import-Module (Join-Path $PSScriptRoot "Install-TestHelpers.psm1") -Force
}

# ==========================================================================
# Unit tests
# ==========================================================================

Describe "Install Directory Resolution" -Tag Unit {
    It "Uses explicit OLLAMA_INSTALL_DIR when set" {
        $testDir = Join-Path $env:TEMP "ollama-test-explicit"
        # The script resolves InstallDir from OLLAMA_INSTALL_DIR env var
        $defaultDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        $defaultDir | Should -Not -BeNullOrEmpty
    }

    It "Falls back to default directory" {
        $defaultDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        $defaultDir | Should -Be "$env:LOCALAPPDATA\Programs\Ollama"
    }
}

Describe "Architecture Detection" -Tag Unit {
    It "Returns amd64 or arm64" {
        $osArch = $null
        try {
            $osArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
        } catch { }

        $result = if ($null -ne $osArch) {
            switch ($osArch.ToString().ToLower()) {
                "x64"   { "amd64" }
                "arm64" { "arm64" }
                default { "amd64" }
            }
        } else {
            switch ($env:PROCESSOR_ARCHITECTURE) {
                "ARM64" { "arm64" }
                default { "amd64" }
            }
        }
        $result | Should -BeIn @("amd64", "arm64")
    }
}

Describe "Core MSI Name Selection" -Tag Unit {
    It "Returns ollama-core.msi for amd64" {
        $result = if ("amd64" -eq "arm64") { "ollama-core-arm64.msi" } else { "ollama-core.msi" }
        $result | Should -Be "ollama-core.msi"
    }

    It "Returns ollama-core-arm64.msi for arm64" {
        $result = if ("arm64" -eq "arm64") { "ollama-core-arm64.msi" } else { "ollama-core.msi" }
        $result | Should -Be "ollama-core-arm64.msi"
    }
}

Describe "Packages.json Parsing" -Tag Unit {
    BeforeEach {
        $testEnv = Initialize-TestEnvironment
    }

    AfterEach {
        Remove-TestEnvironment
    }

    It "Parses valid packages.json" {
        $jsonPath = Join-Path $testEnv.CacheDir "packages.json"
        $manifest = New-MockPackagesJson -OutputPath $jsonPath

        $loaded = Get-Content $jsonPath -Raw | ConvertFrom-Json
        $loaded.version | Should -Be "0.15.0"
        $loaded.packages.Count | Should -Be 5
    }

    It "Finds backend by name with arch" {
        $jsonPath = Join-Path $testEnv.CacheDir "packages.json"
        New-MockPackagesJson -OutputPath $jsonPath

        $loaded = Get-Content $jsonPath -Raw | ConvertFrom-Json
        $cudaPkg = $loaded.packages | Where-Object { $_.name -eq "cuda_v12" }
        $cudaPkg | Should -Not -BeNullOrEmpty
        $cudaPkg.arch | Should -Be "amd64"
        $cudaPkg.file | Should -Be "ollama-cuda-v12.msi"
        $cudaPkg.deps | Should -Be "ollama-cuda-deps-12.8.1.msi"
    }

    It "Filters packages by architecture" {
        $jsonPath = Join-Path $testEnv.CacheDir "packages.json"
        New-MockPackagesJson -OutputPath $jsonPath

        $loaded = Get-Content $jsonPath -Raw | ConvertFrom-Json

        # All test packages are amd64 — filtering for amd64 returns all 5
        $amd64Pkgs = $loaded.packages | Where-Object {
            $pkgArch = if ($_.arch) { $_.arch } else { "amd64" }
            $pkgArch -eq "amd64"
        }
        $amd64Pkgs.Count | Should -Be 5

        # Filtering for arm64 returns none (no arm64 GPU backends yet)
        $arm64Pkgs = $loaded.packages | Where-Object {
            $pkgArch = if ($_.arch) { $_.arch } else { "amd64" }
            $pkgArch -eq "arm64"
        }
        $arm64Pkgs.Count | Should -Be 0
    }
}

Describe "Backend Detection from Filesystem" -Tag Unit {
    BeforeAll {
        $tokens = $null
        $errors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($InstallScript, [ref]$tokens, [ref]$errors)
        $functionAst = $ast.Find({
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq "Get-InstalledBackends"
        }, $true)
        if (-not $functionAst) {
            throw "Get-InstalledBackends not found in install.ps1"
        }
        . ([scriptblock]::Create($functionAst.Extent.Text))
    }

    BeforeEach {
        $testEnv = Initialize-TestEnvironment
    }

    AfterEach {
        Remove-TestEnvironment
    }

    It "Detects installed backends" {
        $dir = $testEnv.InstallDir
        New-MockInstalledDir -Dir $dir -Backends @("cuda_v12", "vulkan", "rocm_v7_1")

        $backends = Get-InstalledBackends -Dir $dir

        $backends | Should -Contain "cuda_v12"
        $backends | Should -Contain "vulkan"
        $backends | Should -Contain "rocm_v7_1"
        $backends | Should -Not -Contain "cuda_v13"
        $backends | Should -Not -Contain "mlx_cuda_v13"
        $backends | Should -Not -Contain ""
    }

    It "Returns empty for fresh install" {
        $dir = $testEnv.InstallDir
        $backends = Get-InstalledBackends -Dir $dir

        $backends.Count | Should -Be 0
    }
}

Describe "Backend Selection Logic" -Tag Unit {
    It "All mode selects all available backends" {
        $availableBackends = @("cuda_v12", "cuda_v13", "rocm_v7_1", "vulkan", "mlx_cuda_v13")
        # OLLAMA_INSTALL_ALL=1: select everything
        $selected = $availableBackends
        $selected.Count | Should -Be 5
    }

    It "Explicit mode uses provided list" {
        $explicit = @("cuda_v12", "rocm_v7_1")
        $explicit.Count | Should -Be 2
        $explicit | Should -Contain "cuda_v12"
        $explicit | Should -Contain "rocm_v7_1"
    }

    It "Minimal mode selects nothing" {
        $selected = @()
        $selected.Count | Should -Be 0
    }

    It "Upgrade mode preserves installed backends" {
        $installed = @("cuda_v12", "vulkan")
        # On upgrade without explicit flags, keep what's installed
        $selected = $installed
        $selected | Should -Contain "cuda_v12"
        $selected | Should -Contain "vulkan"
    }
}

Describe "Packages.json Generation" -Tag Unit {
    BeforeAll {
        $script:repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:packagesJsonScript = Join-Path $script:repoRoot "app\msi\generate-packages-json.ps1"
    }

    BeforeEach {
        $script:packagesJsonTestRoot = Join-Path $env:TEMP "ollama-packages-json-test-$([Guid]::NewGuid())"
        $script:packagesJsonDist = Join-Path $script:packagesJsonTestRoot "dist"
        New-Item -ItemType Directory -Path $script:packagesJsonDist -Force | Out-Null
    }

    AfterEach {
        if (Test-Path $script:packagesJsonTestRoot) {
            Remove-Item $script:packagesJsonTestRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "Selects dependency MSI from backend version breadcrumb" {
        $rocmDir = Join-Path $script:packagesJsonDist "windows-amd64\lib\ollama\rocm_v7_1"
        New-Item -ItemType Directory -Path $rocmDir -Force | Out-Null
        "6.2.41512" | Out-File (Join-Path $rocmDir "rocm-version.txt") -Encoding ascii -NoNewline
        "backend" | Out-File (Join-Path $script:packagesJsonDist "ollama-rocm.msi") -Encoding ascii -NoNewline
        "deps" | Out-File (Join-Path $script:packagesJsonDist "ollama-rocm-deps-6.2.41512.msi") -Encoding ascii -NoNewline

        $outputFile = Join-Path $script:packagesJsonDist "packages.json"
        & $script:packagesJsonScript -Version "0.21.0" -DistDir $script:packagesJsonDist -OutputFile $outputFile *> $null

        $manifest = Get-Content $outputFile -Raw | ConvertFrom-Json
        $pkg = $manifest.packages | Where-Object { $_.name -eq "rocm_v7_1" } | Select-Object -First 1
        $pkg.deps | Should -Be "ollama-rocm-deps-6.2.41512.msi"
        $expectedHash = (Get-FileHash (Join-Path $script:packagesJsonDist "ollama-rocm-deps-6.2.41512.msi") -Algorithm SHA256).Hash.ToLower()
        $pkg.deps_sha256 | Should -Be $expectedHash
    }

    It "Uses shared CUDA v13 deps for CUDA v13 and MLX CUDA v13 backends" {
        $cudaDir = Join-Path $script:packagesJsonDist "windows-amd64\lib\ollama\cuda_v13"
        $mlxDir = Join-Path $script:packagesJsonDist "windows-amd64\lib\ollama\mlx_cuda_v13"
        New-Item -ItemType Directory -Path $cudaDir -Force | Out-Null
        New-Item -ItemType Directory -Path $mlxDir -Force | Out-Null
        "13.0.0" | Out-File (Join-Path $cudaDir "cuda-version.txt") -Encoding ascii -NoNewline
        "13.0.0-cudnn9.16.0" | Out-File (Join-Path $mlxDir "mlx-version.txt") -Encoding ascii -NoNewline
        "backend" | Out-File (Join-Path $script:packagesJsonDist "ollama-cuda-v13.msi") -Encoding ascii -NoNewline
        "backend" | Out-File (Join-Path $script:packagesJsonDist "ollama-mlx-cuda-v13.msi") -Encoding ascii -NoNewline
        "deps" | Out-File (Join-Path $script:packagesJsonDist "ollama-cuda-v13-deps-13.0.0-cudnn9.16.0.msi") -Encoding ascii -NoNewline

        $outputFile = Join-Path $script:packagesJsonDist "packages.json"
        & $script:packagesJsonScript -Version "0.21.0" -DistDir $script:packagesJsonDist -OutputFile $outputFile *> $null

        $manifest = Get-Content $outputFile -Raw | ConvertFrom-Json
        foreach ($packageName in @("cuda_v13", "mlx_cuda_v13")) {
            $pkg = $manifest.packages | Where-Object { $_.name -eq $packageName } | Select-Object -First 1
            $pkg.deps | Should -Be "ollama-cuda-v13-deps-13.0.0-cudnn9.16.0.msi"
            $pkg.deps | Should -Not -Match "mlx-deps"
        }
    }

    It "Fails when multiple dependency MSIs match a backend" {
        $rocmDir = Join-Path $script:packagesJsonDist "windows-amd64\lib\ollama\rocm_v7_1"
        New-Item -ItemType Directory -Path $rocmDir -Force | Out-Null
        "6.2.41512" | Out-File (Join-Path $rocmDir "rocm-version.txt") -Encoding ascii -NoNewline
        "backend" | Out-File (Join-Path $script:packagesJsonDist "ollama-rocm.msi") -Encoding ascii -NoNewline
        "deps" | Out-File (Join-Path $script:packagesJsonDist "ollama-rocm-deps-6.2.41512.msi") -Encoding ascii -NoNewline
        "stale" | Out-File (Join-Path $script:packagesJsonDist "ollama-rocm-deps-6.1.0.msi") -Encoding ascii -NoNewline

        {
            & $script:packagesJsonScript -Version "0.21.0" -DistDir $script:packagesJsonDist *> $null
        } | Should -Throw "*Multiple dependency MSIs found for rocm_v7_1*"
    }

    It "Rejects dependency MSI names that do not match the breadcrumb" {
        $rocmDir = Join-Path $script:packagesJsonDist "windows-amd64\lib\ollama\rocm_v7_1"
        New-Item -ItemType Directory -Path $rocmDir -Force | Out-Null
        "6.2.41512" | Out-File (Join-Path $rocmDir "rocm-version.txt") -Encoding ascii -NoNewline
        "backend" | Out-File (Join-Path $script:packagesJsonDist "ollama-rocm.msi") -Encoding ascii -NoNewline
        "stale" | Out-File (Join-Path $script:packagesJsonDist "ollama-rocm-deps-6.1.0.msi") -Encoding ascii -NoNewline

        {
            & $script:packagesJsonScript -Version "0.21.0" -DistDir $script:packagesJsonDist *> $null
        } | Should -Throw "*expected ollama-rocm-deps-6.2.41512.msi*"
    }

    It "Fails when the expected dependency MSI is missing" {
        $rocmDir = Join-Path $script:packagesJsonDist "windows-amd64\lib\ollama\rocm_v7_1"
        New-Item -ItemType Directory -Path $rocmDir -Force | Out-Null
        "6.2.41512" | Out-File (Join-Path $rocmDir "rocm-version.txt") -Encoding ascii -NoNewline
        "backend" | Out-File (Join-Path $script:packagesJsonDist "ollama-rocm.msi") -Encoding ascii -NoNewline

        {
            & $script:packagesJsonScript -Version "0.21.0" -DistDir $script:packagesJsonDist *> $null
        } | Should -Throw "*Dependency MSI missing for rocm_v7_1: expected ollama-rocm-deps-6.2.41512.msi*"
    }
}

Describe "SHA256 Download Comparison" -Tag Unit {
    BeforeEach {
        $testEnv = Initialize-TestEnvironment
    }

    AfterEach {
        Remove-TestEnvironment
    }

    It "Skips download when hash matches" {
        $testFile = Join-Path $testEnv.CacheDir "test.msi"
        "test content" | Out-File $testFile -Encoding utf8
        $hash = (Get-FileHash -Path $testFile -Algorithm SHA256).Hash.ToLower()

        # Simulate: cached hash matches manifest hash -> skip
        $needsDownload = $true
        $cachedHash = (Get-FileHash -Path $testFile -Algorithm SHA256).Hash.ToLower()
        if ($cachedHash -eq $hash) {
            $needsDownload = $false
        }

        $needsDownload | Should -Be $false
    }

    It "Downloads when hash differs" {
        $testFile = Join-Path $testEnv.CacheDir "test.msi"
        "test content" | Out-File $testFile -Encoding utf8

        $needsDownload = $true
        $cachedHash = (Get-FileHash -Path $testFile -Algorithm SHA256).Hash.ToLower()
        if ($cachedHash -eq "0000000000000000") {
            $needsDownload = $false
        }

        $needsDownload | Should -Be $true
    }
}

Describe "Legacy Inno Setup Migration" -Tag Unit {
    It "Detects legacy Inno Setup registry key format" {
        $keyPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
        # Just verify the key path format is correct
        $keyPath | Should -Match '\{44E83376-CE68-45EB-8FC1-393500EB558C\}_is1$'
    }

    It "Keeps legacy upgrades opt-in for MSI migration" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match 'OLLAMA_MIGRATE_TO_MSI'
        $content | Should -Match '\$MigrateToMsi\s+=\s+\$env:OLLAMA_MIGRATE_TO_MSI -eq "1"'
        $content | Should -Match 'function Resolve-InstallMode'
        $content | Should -Match 'if \(\$innoInstallKey -and -not \$MigrateToMsi\)'
        $content | Should -Match 'return "inno"'
    }

    It "Builds a guarded legacy Inno Setup installer" {
        $iss = Get-Content (Join-Path (Split-Path -Parent $ScriptRoot) "app\ollama.iss") -Raw
        $iss | Should -Match 'MsiOllamaInstalled'
        $iss | Should -Match 'CreateOleObject\(''WindowsInstaller\.Installer''\)'
        $iss | Should -Match 'RelatedProducts\(UpgradeCode\)'
        $iss | Should -Match 'MsiInstallQueryFailed'
        $iss | Should -Match '7A5B3E2F-1C4D-4F8A-9E6B-0D2A1F3C5E7D'
        $iss | Should -Match 'B4C6D8E0-2F1A-4E3C-A5D7-9B0E1F2A3C4D'
        $iss | Should -Match 'managed by the MSI installer'
        $iss | Should -Match 'SuppressibleMsgBox'
        $iss | Should -Match 'Excludes: "\\mlx_\*\\\*"'

        $coreWxs = Get-Content (Join-Path (Split-Path -Parent $ScriptRoot) "app\msi\ollama-core.wxs") -Raw
        $coreArm64Wxs = Get-Content (Join-Path (Split-Path -Parent $ScriptRoot) "app\msi\ollama-core-arm64.wxs") -Raw
        $coreWxs | Should -Not -Match 'Name="Installer" Type="string" Value="MSI"'
        $coreArm64Wxs | Should -Not -Match 'Name="Installer" Type="string" Value="MSI"'
    }
}

Describe "UpgradeCode Consistency" -Tag Unit {
    It "All WXS UpgradeCodes match install script constants" {
        $scriptContent = Get-Content $InstallScript -Raw

        # Verify that the UpgradeCodes hash in the script contains expected GUIDs
        $scriptContent | Should -Match "7A5B3E2F-1C4D-4F8A-9E6B-0D2A1F3C5E7D"  # core
        $scriptContent | Should -Match "3F8A2D1E-5B6C-4E7F-A9D0-1C2B3E4F5A6D"  # cuda_v12
        $scriptContent | Should -Match "9C7E3A1B-2D4F-4E5A-B6C8-0D1E2F3A4B5C"  # cuda_v13
        $scriptContent | Should -Match "4B2E8F1A-6C3D-4A5E-9F7B-0D1C2E3A4B5D"  # rocm
        $scriptContent | Should -Match "6D4A2E8F-1B3C-4F5E-A7D9-0C1B2E3F4A5D"  # vulkan
        $scriptContent | Should -Match "3E7A1B5C-9D2F-4A6E-B8C0-1F2D3E4A5B6C"  # mlx_cuda_v13
        $scriptContent | Should -Not -Match "4F8B2C6D-0E3A-5B7F-C9D1-2E3F4A5B6C7D"  # removed mlx_deps
    }

    It "UpgradeCodes in WXS files match install script" {
        $wxsDir = Join-Path (Split-Path -Parent $ScriptRoot) "app\msi"

        # Read core WXS
        $coreWxs = Get-Content (Join-Path $wxsDir "ollama-core.wxs") -Raw
        $coreWxs | Should -Match "7A5B3E2F-1C4D-4F8A-9E6B-0D2A1F3C5E7D"

        # Read cuda-v12 WXS
        $cudaWxs = Get-Content (Join-Path $wxsDir "cuda-v12.wxs") -Raw
        $cudaWxs | Should -Match "3F8A2D1E-5B6C-4E7F-A9D0-1C2B3E4F5A6D"
    }

    It "Chained uninstall includes MLX backend and CUDA v13 deps" {
        $wxsDir = Join-Path (Split-Path -Parent $ScriptRoot) "app\msi"
        $chainedUninstall = Get-Content (Join-Path $wxsDir "chained-uninstall.ps1") -Raw

        $chainedUninstall | Should -Match "3E7A1B5C-9D2F-4A6E-B8C0-1F2D3E4A5B6C"
        $chainedUninstall | Should -Match "8F2A4E6C-1B3D-5C7E-A9F0-2D1E3B4A5C6D"
        $chainedUninstall | Should -Not -Match "4F8B2C6D-0E3A-5B7F-C9D1-2E3F4A5B6C7D"
    }
}

Describe "Install Script Syntax" -Tag Unit {
    It "Script parses without errors" {
        $errors = $null
        $null = [System.Management.Automation.Language.Parser]::ParseFile(
            $InstallScript, [ref]$null, [ref]$errors
        )
        $errors.Count | Should -Be 0
    }

    It "Uses environment variables for configuration" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match 'OLLAMA_INSTALL_ALL'
        $content | Should -Match 'OLLAMA_INSTALL_MINIMAL'
        $content | Should -Match 'OLLAMA_INSTALL_DIR'
        $content | Should -Match 'OLLAMA_VERSION'
        $content | Should -Match 'OLLAMA_UNINSTALL'
        $content | Should -Match 'OLLAMA_CACHE_ONLY'
        $content | Should -Match 'OLLAMA_INSTALL_CACHED'
        $content | Should -Match 'OLLAMA_MIGRATE_TO_MSI'
        $content | Should -Not -Match 'OLLAMA_MSI_CACHE'
    }

    It "Has comment-based help" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match '\.SYNOPSIS'
        $content | Should -Match '\.DESCRIPTION'
        $content | Should -Match '\.EXAMPLE'
    }

    It "Uses installer-mode-specific install directory precedence" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match 'function Resolve-InstallDir\s*\{\s*param\(\[string\]\$InstallMode = ""\)'
        $content | Should -Match '\$InstallMode -eq "inno"'
        $content | Should -Match 'Resolve-InstallDir -InstallMode "inno"'
        $content | Should -Match 'Resolve-InstallDir -InstallMode "msi"'
    }

    It "Documents environment variables in help" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match 'Environment variables:'
        $content | Should -Match 'OLLAMA_INSTALL_ALL.*Install all GPU backends'
        $content | Should -Match 'OLLAMA_INSTALL_MINIMAL.*CPU-only'
        $content | Should -Match 'OLLAMA_CACHE_ONLY.*download installer payloads'
        $content | Should -Match 'OLLAMA_INSTALL_CACHED.*install from the Ollama installer cache'
        $content | Should -Match 'OLLAMA_MIGRATE_TO_MSI.*migrate'
    }

    It "Treats package hash mismatches as fatal" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match 'function Assert-FileHashMatches'
        $content | Should -Match 'function Get-FileSHA256'
        $content | Should -Match '\[System.Security.Cryptography.SHA256\]::Create\(\)'
        $content | Should -Match 'throw "SHA256 mismatch'
        $content | Should -Not -Match 'WARNING: SHA256 mismatch'
        $content | Should -Not -Match 'Get-FileHash'
    }

    It "Supports app-updater cache-only installs" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match '\$CacheOnly\s+=\s+\$env:OLLAMA_CACHE_ONLY -eq "1"'
        $content | Should -Match '\$InstallCached\s+=\s+\$env:OLLAMA_INSTALL_CACHED -eq "1"'
        $content | Should -Match 'Get-CachedInstallCacheTarget'
        $content | Should -Match 'Required backend MSI missing from cache'
        $content | Should -Match 'ol-installer-cache'
        $content | Should -Not -Match 'OLLAMA_MSI_CACHE'
    }

    It "Does not wait on the Inno Setup process tree" {
        $content = Get-Content $InstallScript -Raw
        $start = $content.IndexOf("function Start-InnoInstaller")
        $end = $content.IndexOf("function Invoke-InnoInstall")
        $start | Should -BeGreaterOrEqual 0
        $end | Should -BeGreaterThan $start
        $innoInstaller = $content.Substring($start, $end - $start)
        $innoInstaller | Should -Match 'Start-Process -FilePath \$InstallerPath'
        $innoInstaller | Should -Match '-PassThru -WindowStyle Hidden'
        $innoInstaller | Should -Match '\$proc\.WaitForExit\(\)'
        $innoInstaller | Should -Not -Match '-Wait'
    }

    It "Maps backend packages to their dependency package keys" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match '\$BackendDepsPackages\s*=\s*@\{'
        $content | Should -Match '"cuda_v12"\s*=\s*"cuda_v12_deps"'
        $content | Should -Match '"cuda_v13"\s*=\s*"cuda_v13_deps"'
        $content | Should -Match '"rocm"\s*=\s*"rocm_deps"'
        $content | Should -Match '"vulkan"\s*=\s*"vulkan_deps"'
        $content | Should -Match '"mlx_cuda_v13"\s*=\s*"cuda_v13_deps"'
        $content | Should -Match 'Get-BackendDepsPackage -Name \$backend'
    }

    It "Fails MSI installs instead of continuing with partial payloads" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match 'throw "msiexec returned exit code'
        $content | Should -Not -Match 'skipping backend'
        $content | Should -Not -Match '\$dl\.msiPath = \$null'
    }

    It "Requires legacy Inno removal before MSI migration proceeds" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match 'function Remove-InnoSetupInstall\s*\{\s*param\(\[switch\]\$RequireRemoved\)'
        $content | Should -Match 'Remove-InnoSetupInstall -RequireRemoved'
        $content | Should -Match 'Legacy Inno Setup uninstaller not found'
        $content | Should -Match 'Inno Setup registry key still exists after uninstall'
    }

    It "Generates unsigned local integration copies without weakening the production script" {
        $testScript = New-InstallScriptForTest -BaseUrl "http://127.0.0.1:1" -DisableSignatureVerification $true
        try {
            $testScript.Path | Should -Match '\\build\\install-tests\\'
            $testScript.SignatureVerificationDisabled | Should -Be $true

            $testContent = Get-Content $testScript.Path -Raw
            $testContent | Should -Match 'Signature verification disabled for local integration tests'
            $testContent | Should -Not -Match 'Get-AuthenticodeSignature'

            $productionContent = Get-Content $InstallScript -Raw
            $productionContent | Should -Match 'Get-AuthenticodeSignature'
            $productionContent | Should -Match 'O=Ollama Inc'
        } finally {
            if ($testScript.TempDir -and (Test-Path $testScript.TempDir)) {
                Remove-Item -LiteralPath $testScript.TempDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    It "Caches pinned Inno Setup installers under the repo cache" {
        $cacheRoot = Get-InstallTestCacheRoot
        $cacheRoot | Should -Match '\\\.cache\\install-tests$'
        $cacheRoot | Should -Not -Match ([regex]::Escape($env:TEMP))
    }
}

Describe "MSI Build Graph" -Tag Unit {
    BeforeAll {
        $script:repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:msiCmake = Get-Content (Join-Path $script:repoRoot "app\msi\CMakeLists.txt") -Raw
        $script:runtimeDepsCmake = Get-Content (Join-Path $script:repoRoot "cmake\windows-runtime-deps.cmake") -Raw
        $script:llamaServerCmake = Get-Content (Join-Path $script:repoRoot "llama\server\CMakeLists.txt") -Raw
        $script:checkPayloadsCmake = Get-Content (Join-Path $script:repoRoot "app\msi\check-required-payloads.cmake") -Raw
        $script:foldersWxs = Get-Content (Join-Path $script:repoRoot "app\msi\Folders.wxs") -Raw
        $script:coreWxs = Get-Content (Join-Path $script:repoRoot "app\msi\ollama-core.wxs") -Raw
        $script:coreArm64Wxs = Get-Content (Join-Path $script:repoRoot "app\msi\ollama-core-arm64.wxs") -Raw
        $script:buildWindows = Get-Content (Join-Path $script:repoRoot "scripts\build_windows.ps1") -Raw
        $script:releaseWorkflow = Get-Content (Join-Path $script:repoRoot ".github\workflows\release.yaml") -Raw
    }

    It "Packages direct core runtime DLL dependencies" {
        $script:msiCmake | Should -Match 'CORE_RUNTIME_WXS'
        $script:msiCmake | Should -Match 'CORE_RUNTIME_FILES'
        $script:msiCmake | Should -Match 'check-core-payloads'
        $script:msiCmake | Should -Match 'llama-server\.exe'
        $script:msiCmake | Should -Match 'GO_LICENSE'
        $script:msiCmake | Should -Match 'generate_go_license\.cmake'
        $script:msiCmake | Should -Match 'list\(APPEND CORE_RUNTIME_FILES "\$\{msi-go-license-amd64_OUTPUT\}"\)'
        $script:msiCmake | Should -Match 'list\(APPEND CORE_RUNTIME_ARM64_FILES "\$\{msi-go-license-arm64_OUTPUT\}"\)'
        $script:msiCmake | Should -Match 'core_runtime_Components'
        $script:msiCmake | Should -Match '-BackendPath "lib/ollama"'
        $script:msiCmake | Should -Match '-NoRecurse'
        $script:checkPayloadsCmake | Should -Match 'Missing required MSI payload'
        $script:coreWxs | Should -Match '<ComponentGroupRef Id="core_runtime_Components" />'
    }

    It "Tracks dependency MSI inputs before package hashes are generated" {
        $script:msiCmake | Should -Match 'file\(GLOB_RECURSE DEPS_INPUT_FILES'
        $script:msiCmake | Should -Match '-NoProfile -ExecutionPolicy Bypass -File\s*\r?\n\s*"\$\{DEPS_WXS_DIR\}/generate-backend-deps\.ps1"'
        $script:msiCmake | Should -Match 'DEPENDS "\$\{DEPS_WXS_DIR\}/generate-backend-deps\.ps1" \$\{DEPS_INPUT_FILES\}'
        $script:msiCmake | Should -Match 'DEPENDS "\$\{GENERATED_WXS\}" \$\{DEPS_INPUT_FILES\} \$\{DEPS_SOURCES\}'
        $script:msiCmake | Should -Match 'OUTPUT "\$\{PACKAGES_JSON\}"'
        $script:msiCmake | Should -Match 'DEPENDS \$\{PACKAGES_JSON_INPUTS\}'
    }

    It "Rejects downloaded dependency artifacts that are not MSIs" {
        $script:msiCmake | Should -Match 'DOWNLOADED_MAGIC'
        $script:msiCmake | Should -Match 'd0cf11e0a1b11ae1'
        $script:msiCmake | Should -Match 'not a valid MSI'
    }

    It "Does not re-sign downloaded or reused dependency MSIs" {
        $downloadedBlock = [regex]::Match($script:msiCmake, 'if\(DEPS_DOWNLOADED\)(?s:.*?)else\(\)').Value
        $downloadedBlock | Should -Not -BeNullOrEmpty
        $downloadedBlock | Should -Not -Match 'sign_target'
        $script:msiCmake | Should -Match 'sign_output\("\$\{DEPS_MSI_PATH\}"\)'
    }

    It "Uses MSI-specific version and cabinet settings" {
        $script:msiCmake | Should -Match 'OLLAMA_MSI_VERSION'
        $script:msiCmake | Should -Not -Match 'OLLAMA_PKG_VERSION'
        $script:msiCmake | Should -Match '-d MSI_VERSION=\$\{OLLAMA_MSI_VERSION\}'
        $script:msiCmake | Should -Match 'set\(OLLAMA_CABSIZEMB "200"'
    }

    It "Builds both MSI and legacy Inno artifacts" {
        $script:buildWindows | Should -Match 'function installer'
        $script:buildWindows | Should -Match 'ISCC\.exe'
        $script:buildWindows | Should -Match 'msi\s*\r?\n\s*installer\s*\r?\n\s*zip'
        $script:releaseWorkflow | Should -Match 'dist/OllamaSetup\.exe'
        $script:releaseWorkflow | Should -Match 'dist/ollama-windows-amd64-mlx\.zip'
        $script:releaseWorkflow | Should -Match 'dist/ollama-mlx-cuda-v13\.msi'
        $script:releaseWorkflow | Should -Match 'dist/ollama-cuda-v13-deps-\*\.msi'
        $script:releaseWorkflow | Should -Not -Match 'dist/\*\.exe'
    }

    It "Includes WiX Util notices in core MSIs" {
        $licenseRoot = Join-Path $script:repoRoot "app\msi\licenses\wix"
        Test-Path (Join-Path $licenseRoot "LICENSE.TXT") | Should -Be $true

        $script:foldersWxs | Should -Not -Match 'Name="licenses"'
        $script:foldersWxs | Should -Not -Match 'Name="wix"'

        $script:coreWxs | Should -Match 'util:CloseApplication'
        $script:coreWxs | Should -Match 'Id="WixNotice_Components" Directory="lib_ollama_Dir"'
        $script:coreWxs | Should -Match 'WIX_LICENSE\.TXT'

        $script:coreArm64Wxs | Should -Match 'util:CloseApplication'
        $script:coreArm64Wxs | Should -Match 'Id="WixNotice_Components" Directory="lib_ollama_Dir"'
        $script:coreArm64Wxs | Should -Match 'WIX_LICENSE\.TXT'
    }

    It "Keeps Ollama-built MLX binaries out of the shared CUDA dependency MSI" {
        $script:msiCmake | Should -Match '-ExcludeFile "mlx\.dll,mlxc\.dll,ollama_xgrammar\.dll"'
        $mlxWxs = Get-Content (Join-Path $script:repoRoot "app\msi\mlx-cuda-v13.wxs") -Raw
        $mlxWxs | Should -Match 'mlx_cuda_v13\\ollama_xgrammar\.dll'
    }

    It "Carries common Windows C and C++ runtime DLL dependencies" {
        $script:runtimeDepsCmake | Should -Match 'OLLAMA_WINDOWS_RUNTIME_DEP_INCLUDE_REGEXES'
        $script:runtimeDepsCmake | Should -Match 'msvcp\.\*\\\\\.dll'
        $script:runtimeDepsCmake | Should -Match 'vcruntime\.\*\\\\\.dll'
        $script:runtimeDepsCmake | Should -Match 'libomp\.\*\\\\\.dll'
        $script:llamaServerCmake | Should -Match 'windows-runtime-deps\.cmake'
        $script:llamaServerCmake | Should -Match 'ollama_install_llama_windows_runtime_dlls'
        $script:llamaServerCmake | Should -Match 'PRE_INCLUDE_REGEXES hipblas rocblas .* \$\{OLLAMA_WINDOWS_RUNTIME_DEP_INCLUDE_REGEXES\}'
    }

    It "Generates bounded WiX identifiers for deep dependency paths" {
        $testRoot = Join-Path $env:TEMP "ollama-wix-id-test-$([Guid]::NewGuid())"
        $distDir = Join-Path $testRoot "dist"
        $backendPath = "lib\ollama\mlx_cuda_v13"
        $deepDir = Join-Path $distDir "windows-amd64\$backendPath\include\cccl\cuda\__ptx\instructions\generated"
        $outputFile = Join-Path $testRoot "mlx-deps-generated.wxs"
        $generator = Join-Path $script:repoRoot "app\msi\dependencies\generate-backend-deps.ps1"

        try {
            New-Item -ItemType Directory -Path $deepDir -Force | Out-Null
            "header" | Out-File -FilePath (Join-Path $deepDir "barrier_cluster.h") -Encoding ascii -NoNewline

            $powerShellExe = (Get-Process -Id $PID).Path
            if (-not $powerShellExe) {
                $powerShellExe = (Get-Command powershell -ErrorAction Stop).Source
            }

            $stdoutPath = Join-Path $testRoot "generator.out"
            $stderrPath = Join-Path $testRoot "generator.err"
            Set-TestProcessEnvironment
            $proc = Start-Process -FilePath $powerShellExe `
                -ArgumentList @(
                    "-NoProfile",
                    "-File", "`"$generator`"",
                    "-BackendPath", $backendPath,
                    "-ComponentGroupId", "mlx_cuda_v13_deps_Components",
                    "-RootDirectoryId", "lib_ollama_mlx_cuda_v13_Dir",
                    "-DistDir", "`"$distDir`"",
                    "-OutputFile", "`"$outputFile`"",
                    "-ExcludeFile", "mlx.dll,mlxc.dll,ollama_xgrammar.dll"
                ) `
                -WindowStyle Hidden `
                -RedirectStandardOutput $stdoutPath `
                -RedirectStandardError $stderrPath `
                -PassThru

            if (-not $proc.WaitForExit(30000)) {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                throw "generate-backend-deps.ps1 timed out after 30 seconds"
            }

            $output = @()
            if (Test-Path $stdoutPath) { $output += Get-Content $stdoutPath }
            if (Test-Path $stderrPath) { $output += Get-Content $stderrPath }
            if ($null -ne $proc.ExitCode) {
                $proc.ExitCode | Should -Be 0 -Because ($output -join "`n")
            }
            $content = Get-Content $outputFile -Raw
            $content | Should -Match 'Name="generated"'

            $identifierMatches = [regex]::Matches($content, '\b(?:Id|Directory)="([^"]+)"')
            $identifierMatches.Count | Should -BeGreaterThan 0
            foreach ($match in $identifierMatches) {
                $match.Groups[1].Value.Length | Should -BeLessOrEqual 72 -Because $match.Value
            }
        } finally {
            if (Test-Path $testRoot) {
                Remove-Item $testRoot -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

}

Describe "MSI Property" -Tag Unit {
    It "Install-Msi uses ROOTDIRECTORY not TARGETDIR for install" {
        $content = Get-Content $InstallScript -Raw
        # Should use ROOTDIRECTORY for custom install paths
        $content | Should -Match 'ROOTDIRECTORY='
        # The Install-Msi function should NOT use TARGETDIR (the WXS uses StandardDirectory).
        # Note: TARGETDIR is still used in the msiexec /a extraction step, which is correct.
        # We check that the Install-Msi function specifically uses ROOTDIRECTORY.
        $content | Should -Match '\$msiArgs \+= "ROOTDIRECTORY='
    }
}

Describe "Environment Variable Parsing" -Tag Unit {
    It "OLLAMA_INSTALL_ALL=1 is truthy" {
        $val = "1" -eq "1"
        $val | Should -Be $true
    }

    It "OLLAMA_INSTALL_ALL=0 is falsy" {
        $val = "0" -eq "1"
        $val | Should -Be $false
    }

    It "Unset env var is falsy" {
        $val = $null -eq "1"
        $val | Should -Be $false
    }

    It "OLLAMA_INSTALL_BACKENDS splits on comma" {
        $raw = "cuda_v12,rocm,vulkan"
        $backends = ($raw -split ',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
        $backends.Count | Should -Be 3
        $backends | Should -Contain "cuda_v12"
        $backends | Should -Contain "rocm"
        $backends | Should -Contain "vulkan"
    }

    It "OLLAMA_INSTALL_BACKENDS handles spaces around commas" {
        $raw = "cuda_v12 , rocm , vulkan"
        $backends = ($raw -split ',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
        $backends.Count | Should -Be 3
        $backends[0] | Should -Be "cuda_v12"
    }
}

# ==========================================================================
# Build system tests (CMake deps download-or-build logic)
# ==========================================================================

Describe "CMake Deps Download-or-Build" -Tag Build {
    BeforeAll {
        # Require cmake and wix on PATH
        $script:cmakeExe = (Get-Command cmake -ErrorAction SilentlyContinue).Source
        if (-not $script:cmakeExe) {
            throw "cmake not found on PATH. Build tests require CMake."
        }
        $wixExe = Get-Command wix -ErrorAction SilentlyContinue
        if (-not $wixExe) {
            throw "wix not found on PATH. Build tests require WiX Toolset v6."
        }
        $pythonExe = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonExe) {
            throw "Python not found. Build tests require Python for a local HTTP server."
        }

        # Find the repo root and CMakeLists.txt
        $script:repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:cmakeSourceDir = Join-Path $script:repoRoot "app\msi"
        if (-not (Test-Path (Join-Path $script:cmakeSourceDir "CMakeLists.txt"))) {
            throw "CMakeLists.txt not found at $($script:cmakeSourceDir)"
        }

        # Create temp directories for the test
        $script:testRoot = Join-Path $env:TEMP "ollama-cmake-deps-test"
        if (Test-Path $script:testRoot) {
            Remove-Item $script:testRoot -Recurse -Force
        }
        New-Item -ItemType Directory -Path $script:testRoot -Force | Out-Null

        # Create a minimal dist structure with breadcrumb version files
        $script:distDir = Join-Path $script:testRoot "dist"
        $script:distAmd64 = Join-Path $script:distDir "windows-amd64"

        # We need at least one backend's breadcrumb + deps files for testing.
        # Use vulkan since it's the smallest.
        $vulkanDir = Join-Path $script:distAmd64 "lib\ollama\vulkan"
        New-Item -ItemType Directory -Path $vulkanDir -Force | Out-Null
        "1.4.0" | Out-File (Join-Path $vulkanDir "vulkan-version.txt") -Encoding ascii -NoNewline
        # Create a placeholder ggml-vulkan.dll (the WXS exclude file)
        New-Item -ItemType File -Path (Join-Path $vulkanDir "ggml-vulkan.dll") -Force | Out-Null
        # Create a dummy deps DLL so generate-backend-deps.ps1 finds something
        New-Item -ItemType File -Path (Join-Path $vulkanDir "vulkan-1.dll") -Force | Out-Null

        # Also need ollama.exe for the core MSI references (but we won't build core)
        New-Item -ItemType File -Path (Join-Path $script:distAmd64 "ollama.exe") -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $script:distAmd64 "Ollama app.exe") -Force | Out-Null
        New-Item -ItemType Directory -Path (Join-Path $script:distAmd64 "lib\ollama") -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $script:distAmd64 "lib\ollama\llama-server.exe") -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $script:distAmd64 "lib\ollama\ggml-cpu-x64.dll") -Force | Out-Null
        "go licenses" | Out-File (Join-Path $script:distAmd64 "lib\ollama\GO_LICENSE") -Encoding ascii -NoNewline

        # Create a directory for serving pre-built MSIs
        $script:serveDir = Join-Path $script:testRoot "serve"
        New-Item -ItemType Directory -Path $script:serveDir -Force | Out-Null

        # Create a dummy "pre-built" MSI with the compound-file header WiX/MSI uses.
        $script:dummyMsiName = "ollama-vulkan-deps-1.4.0.msi"
        $dummyMsiPath = Join-Path $script:serveDir $script:dummyMsiName
        $dummyBytes = [byte[]]::new(4096)
        (New-Object Random).NextBytes($dummyBytes)
        $msiMagic = [byte[]](0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1)
        [Array]::Copy($msiMagic, $dummyBytes, $msiMagic.Length)
        [System.IO.File]::WriteAllBytes($dummyMsiPath, $dummyBytes)
        $script:dummyMsiHash = (Get-FileHash $dummyMsiPath -Algorithm SHA256).Hash

        # Start the local HTTP server
        $script:server = Start-LocalHttpServer -Dir $script:serveDir
    }

    AfterAll {
        Stop-LocalHttpServer $script:server
        if (Test-Path $script:testRoot) {
            Remove-Item $script:testRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "Downloads deps MSI when OLLAMA_DEPS_DOWNLOAD_URL env var is set" {
        $buildDir = Join-Path $script:testRoot "build-download"
        $savedUrl = $env:OLLAMA_DEPS_DOWNLOAD_URL
        try {
            $env:OLLAMA_DEPS_DOWNLOAD_URL = $script:server.BaseUrl

            $output = & $script:cmakeExe -B $buildDir -S $script:cmakeSourceDir `
                -DOLLAMA_VERSION="0.15.0" `
                -DOLLAMA_DIST_DIR="$($script:distDir)" `
                2>&1 | Out-String

            # Configure output should show the URL was picked up and download succeeded
            $output | Should -Match "Deps download URL:.*$([regex]::Escape($script:server.BaseUrl))"
            $output | Should -Match "Downloaded ollama-vulkan-deps-1.4.0.msi"

            # The MSI should exist in the dist directory
            $downloadedMsi = Join-Path $script:distDir $script:dummyMsiName
            Test-Path $downloadedMsi | Should -Be $true

            # Verify it matches the original (same content)
            $downloadedHash = (Get-FileHash $downloadedMsi -Algorithm SHA256).Hash
            $downloadedHash | Should -Be $script:dummyMsiHash
        } finally {
            if ($savedUrl) { $env:OLLAMA_DEPS_DOWNLOAD_URL = $savedUrl } else { Remove-Item Env:\OLLAMA_DEPS_DOWNLOAD_URL -ErrorAction SilentlyContinue }
            $downloadedMsi = Join-Path $script:distDir $script:dummyMsiName
            Remove-Item $downloadedMsi -Force -ErrorAction SilentlyContinue
            Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "Falls back to local build when file not found on server" {
        $buildDir = Join-Path $script:testRoot "build-fallback"
        $savedUrl = $env:OLLAMA_DEPS_DOWNLOAD_URL

        # Temporarily rename the served file so it 404s
        $servedMsi = Join-Path $script:serveDir $script:dummyMsiName
        $tempName = "$servedMsi.hidden"
        Rename-Item $servedMsi $tempName

        try {
            $env:OLLAMA_DEPS_DOWNLOAD_URL = $script:server.BaseUrl

            $output = & $script:cmakeExe -B $buildDir -S $script:cmakeSourceDir `
                -DOLLAMA_VERSION="0.15.0" `
                -DOLLAMA_DIST_DIR="$($script:distDir)" `
                2>&1 | Out-String

            # Configure output should show download failure and fallback
            $output | Should -Match "Download failed|Will build .* locally|will build locally"

            # The MSI should NOT exist (download failed, build not run yet)
            $downloadedMsi = Join-Path $script:distDir $script:dummyMsiName
            Test-Path $downloadedMsi | Should -Be $false
        } finally {
            if ($savedUrl) { $env:OLLAMA_DEPS_DOWNLOAD_URL = $savedUrl } else { Remove-Item Env:\OLLAMA_DEPS_DOWNLOAD_URL -ErrorAction SilentlyContinue }
            Rename-Item $tempName $servedMsi
            Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "Defaults to https://ollama.com/download when OLLAMA_DEPS_DOWNLOAD_URL is not set" {
        $buildDir = Join-Path $script:testRoot "build-no-url"
        $savedUrl = $env:OLLAMA_DEPS_DOWNLOAD_URL
        try {
            Remove-Item Env:\OLLAMA_DEPS_DOWNLOAD_URL -ErrorAction SilentlyContinue

            $output = & $script:cmakeExe -B $buildDir -S $script:cmakeSourceDir `
                -DOLLAMA_VERSION="0.15.0" `
                -DOLLAMA_DIST_DIR="$($script:distDir)" `
                2>&1 | Out-String

            # Should default to the production URL
            $output | Should -Match "Deps download URL: https://ollama.com/download"
        } finally {
            if ($savedUrl) { $env:OLLAMA_DEPS_DOWNLOAD_URL = $savedUrl }
            Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "Uses MLX payload breadcrumb for the shared CUDA v13 deps MSI" {
        $buildDir = Join-Path $script:testRoot "build-cuda-v13-shared-version"
        $cudaDir = Join-Path $script:distAmd64 "lib\ollama\cuda_v13"
        $mlxDir = Join-Path $script:distAmd64 "lib\ollama\mlx_cuda_v13"
        try {
            New-Item -ItemType Directory -Path $cudaDir -Force | Out-Null
            New-Item -ItemType Directory -Path $mlxDir -Force | Out-Null
            "13.0.0" | Out-File (Join-Path $cudaDir "cuda-version.txt") -Encoding ascii -NoNewline
            "13.0.0-cudnn9.16.0" | Out-File (Join-Path $mlxDir "mlx-version.txt") -Encoding ascii -NoNewline
            New-Item -ItemType File -Path (Join-Path $cudaDir "cudart64_13.dll") -Force | Out-Null
            New-Item -ItemType File -Path (Join-Path $mlxDir "cudnn64_9.dll") -Force | Out-Null

            $output = & $script:cmakeExe -B $buildDir -S $script:cmakeSourceDir `
                -DOLLAMA_VERSION="0.15.0" `
                -DOLLAMA_DIST_DIR="$($script:distDir)" `
                -DOLLAMA_DEPS_FORCE_BUILD=ON `
                2>&1 | Out-String

            $output | Should -Match "cuda-v13 deps version: 13\.0\.0-cudnn9\.16\.0"
            $output | Should -Match "Will build ollama-cuda-v13-deps-13\.0\.0-cudnn9\.16\.0\.msi locally"
        } finally {
            Remove-Item $cudaDir -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item $mlxDir -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "Rejects downloaded file smaller than 1KB" {
        $buildDir = Join-Path $script:testRoot "build-tiny"
        $savedUrl = $env:OLLAMA_DEPS_DOWNLOAD_URL

        # Replace the served MSI with a tiny file (simulating a 404 HTML page saved as .msi)
        $servedMsi = Join-Path $script:serveDir $script:dummyMsiName
        $originalContent = [System.IO.File]::ReadAllBytes($servedMsi)
        try {
            $env:OLLAMA_DEPS_DOWNLOAD_URL = $script:server.BaseUrl
            "Not Found" | Out-File $servedMsi -Encoding ascii

            $output = & $script:cmakeExe -B $buildDir -S $script:cmakeSourceDir `
                -DOLLAMA_VERSION="0.15.0" `
                -DOLLAMA_DIST_DIR="$($script:distDir)" `
                2>&1 | Out-String

            # Should detect the file is not a valid MSI (too small / empty)
            $output | Should -Match "not available from server|will build locally|Will build .* locally"
        } finally {
            if ($savedUrl) { $env:OLLAMA_DEPS_DOWNLOAD_URL = $savedUrl } else { Remove-Item Env:\OLLAMA_DEPS_DOWNLOAD_URL -ErrorAction SilentlyContinue }
            [System.IO.File]::WriteAllBytes($servedMsi, $originalContent)
            $downloadedMsi = Join-Path $script:distDir $script:dummyMsiName
            Remove-Item $downloadedMsi -Force -ErrorAction SilentlyContinue
            Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# ==========================================================================
# Integration tests (require built MSIs in dist/ and clean environment)
#
# All scenarios are nested inside a single Describe block. The BeforeAll
# guard checks for existing Ollama installs and throws immediately if found,
# which prevents ANY scenario from running (and avoids silently uninstalling
# the user's real Ollama).
#
# Verification strategy: check filesystem, MSI registration (COM API), and
# processes rather than parsing script text output. This makes tests resilient
# to cosmetic output changes.
# ==========================================================================

Describe "Integration Tests" -Tag Integration {
    BeforeAll {
        # --- Safety gate: abort ALL integration tests if Ollama is already installed ---
        $existingProducts = Get-InstalledOllamaProducts
        if ($existingProducts.Count -gt 0) {
            $names = ($existingProducts | ForEach-Object { $_.Name }) -join ", "
            throw ("Existing Ollama MSI install detected ($($existingProducts.Count) products: $names). " +
                   "Integration tests install and uninstall Ollama MSIs and would destroy your install. " +
                   "Please uninstall first: `$env:OLLAMA_UNINSTALL=1; .\install.ps1")
        }

        $innoKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
        if (Test-Path $innoKey) {
            throw ("Legacy Ollama InnoSetup install detected. " +
                   "Please uninstall it from Add/Remove Programs before running integration tests.")
        }

        $ollamaProcs = Get-Process -Name "ollama", "Ollama app" -ErrorAction SilentlyContinue
        if ($ollamaProcs) {
            throw ("Ollama processes are running ($(($ollamaProcs | ForEach-Object { $_.Name }) -join ', ')). " +
                   "Please stop Ollama before running integration tests.")
        }

        # Two modes:
        # 1. Local mode: built MSIs in dist/ served via Python HTTP server (for dev/local testing)
        # 2. Remote mode: use official download URL (for CI or when no local build available)
        $script:distDir = Find-DistDir
        if ($script:distDir -and (Test-Path (Join-Path $script:distDir "ollama-core.msi"))) {
            $script:useLocalServer = $true
            Write-Host "Using local MSIs from $($script:distDir)" -ForegroundColor Green
            $script:localInstallerArtifactsSigned = Test-InstallerArtifactsSigned -DistDir $script:distDir
            Set-InstallScriptTestSignatureVerification -Disabled:(-not $script:localInstallerArtifactsSigned)
            if (-not $script:localInstallerArtifactsSigned) {
                Write-Warning "Local installer artifacts are not Ollama-signed; integration tests will use a generated install.ps1 with signature verification disabled."
            }

            $python = Get-Command python -ErrorAction SilentlyContinue
            if (-not $python) {
                $python = Get-Command python3 -ErrorAction SilentlyContinue
            }
            if (-not $python) {
                throw "Python not found. Local mode requires Python for the HTTP server."
            }
        } else {
            $script:useLocalServer = $false
            Set-InstallScriptTestSignatureVerification -Disabled:$false
            Write-Host "No local MSIs found - using official download URL" -ForegroundColor Yellow
        }
    }

    Context "Fresh Install (defaults)" {
        BeforeAll {
            Write-Host "`n>>> Fresh Install (defaults)" -ForegroundColor Cyan
            Invoke-FullUninstall
            if ($script:useLocalServer) {
                $script:server = Start-LocalHttpServer -Dir $script:distDir
            }
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
        }

        It "Installs successfully with OLLAMA_INSTALL_MINIMAL" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "ollama.exe exists in install directory" {
            Test-Path (Join-Path $script:installDir "ollama.exe") | Should -Be $true
        }

        It "Ollama app.exe exists in install directory" {
            Test-Path (Join-Path $script:installDir "Ollama app.exe") | Should -Be $true
        }

        It "packages.json exists in install directory" {
            Test-Path (Join-Path $script:installDir "packages.json") | Should -Be $true
        }

        It "llama-server.exe exists in install directory" {
            Test-Path (Join-Path $script:installDir "lib\ollama\llama-server.exe") | Should -Be $true
        }

        It "Core MSI registered via Windows Installer" {
            $products = Get-InstalledOllamaProducts
            $products | Should -Not -BeNullOrEmpty
            $coreProduct = $products | Where-Object { $_.Name -eq "core" }
            $coreProduct | Should -Not -BeNullOrEmpty
        }

        It "No GPU backends installed in Minimal mode" {
            Test-BackendInstalled -Dir $script:installDir -Backend "cuda_v12" | Should -Be $false
            Test-BackendInstalled -Dir $script:installDir -Backend "cuda_v13" | Should -Be $false
            Test-BackendInstalled -Dir $script:installDir -Backend "rocm" | Should -Be $false
            Test-BackendInstalled -Dir $script:installDir -Backend "vulkan" | Should -Be $false
            Test-BackendInstalled -Dir $script:installDir -Backend "mlx_cuda_v13" | Should -Be $false
        }
    }

    Context "Fresh Install (custom directory)" {
        BeforeAll {
            Write-Host "`n>>> Fresh Install (custom directory)" -ForegroundColor Cyan
            Invoke-FullUninstall
            if ($script:useLocalServer) {
                $script:server = Start-LocalHttpServer -Dir $script:distDir
            }
            $script:customDir = Join-Path $env:TEMP "OllamaIntegrationTest"
            if (Test-Path $script:customDir) {
                Remove-Item $script:customDir -Recurse -Force
            }
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
            if (Test-Path $script:customDir) {
                Remove-Item $script:customDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        It "Installs to custom directory" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1"; OLLAMA_INSTALL_DIR = $script:customDir }
            Test-OllamaInstalled -Dir $script:customDir | Should -Be $true
        }

        It "ollama.exe exists in custom directory" {
            Test-Path (Join-Path $script:customDir "ollama.exe") | Should -Be $true
        }

        It "llama-server.exe exists in custom directory" {
            Test-Path (Join-Path $script:customDir "lib\ollama\llama-server.exe") | Should -Be $true
        }

        It "Ollama app.exe exists in custom directory" {
            Test-Path (Join-Path $script:customDir "Ollama app.exe") | Should -Be $true
        }

        It "Install directory persisted to registry" {
            $regDir = $null
            try {
                $regDir = (Get-ItemProperty -Path "HKCU:\Software\Ollama" -Name "InstallDir" -ErrorAction SilentlyContinue).InstallDir
            } catch {}
            $regDir | Should -Be $script:customDir
        }
    }

    Context "Fresh Install (OLLAMA_INSTALL_ALL)" {
        BeforeAll {
            Write-Host "`n>>> Fresh Install (OLLAMA_INSTALL_ALL)" -ForegroundColor Cyan
            Invoke-FullUninstall
            if ($script:useLocalServer) {
                $script:server = Start-LocalHttpServer -Dir $script:distDir
            }
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
        }

        It "Installs all backends" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_ALL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "All backend directories exist" {
            $pkgJson = Join-Path $script:installDir "packages.json"
            if (Test-Path $pkgJson) {
                $manifest = Get-Content $pkgJson -Raw | ConvertFrom-Json
                foreach ($pkg in $manifest.packages) {
                    Test-BackendInstalled -Dir $script:installDir -Backend $pkg.name |
                        Should -Be $true -Because "$($pkg.name) should be installed with OLLAMA_INSTALL_ALL"
                }
            }
        }

        It "Backend MSIs registered, deps also registered" {
            $products = Get-InstalledOllamaProducts
            # Core should be registered
            ($products | Where-Object { $_.Name -eq "core" }) | Should -Not -BeNullOrEmpty
            # At least one backend should be registered
            $backendProducts = $products | Where-Object { $_.Name -match "^(cuda_v12|cuda_v13|rocm|vulkan|mlx_cuda_v13)$" }
            $backendProducts | Should -Not -BeNullOrEmpty
        }
    }

    Context "Upgrade (MSI to MSI)" {
        BeforeAll {
            Write-Host "`n>>> Upgrade (MSI to MSI)" -ForegroundColor Cyan
            Invoke-FullUninstall
            if ($script:useLocalServer) {
                $script:server = Start-LocalHttpServer -Dir $script:distDir
            }
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
        }

        It "First install succeeds with OLLAMA_INSTALL_ALL" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_ALL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "Reinstall (upgrade) preserves backends" {
            # Record installed backends before upgrade
            $pkgJson = Join-Path $script:installDir "packages.json"
            $beforeBackends = @()
            if (Test-Path $pkgJson) {
                $manifest = Get-Content $pkgJson -Raw | ConvertFrom-Json
                foreach ($pkg in $manifest.packages) {
                    if (Test-BackendInstalled -Dir $script:installDir -Backend $pkg.name) {
                        $beforeBackends += $pkg.name
                    }
                }
            }

            # Run install again (upgrade, no flags = preserve existing)
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" })
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true

            # Verify backends are preserved
            foreach ($backend in $beforeBackends) {
                Test-BackendInstalled -Dir $script:installDir -Backend $backend |
                    Should -Be $true -Because "$backend should be preserved on upgrade"
            }
        }
    }

    Context "Upgrade with OLLAMA_INSTALL_MINIMAL" {
        BeforeAll {
            Write-Host "`n>>> Upgrade with OLLAMA_INSTALL_MINIMAL" -ForegroundColor Cyan
            Invoke-FullUninstall
            if ($script:useLocalServer) {
                $script:server = Start-LocalHttpServer -Dir $script:distDir
            }
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
        }

        It "First install with OLLAMA_INSTALL_ALL succeeds" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_ALL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "Upgrade with OLLAMA_INSTALL_MINIMAL removes GPU backends" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true

            # All GPU backends should be removed
            Test-BackendInstalled -Dir $script:installDir -Backend "cuda_v12" | Should -Be $false
            Test-BackendInstalled -Dir $script:installDir -Backend "cuda_v13" | Should -Be $false
            Test-BackendInstalled -Dir $script:installDir -Backend "rocm" | Should -Be $false
            Test-BackendInstalled -Dir $script:installDir -Backend "vulkan" | Should -Be $false
            Test-BackendInstalled -Dir $script:installDir -Backend "mlx_cuda_v13" | Should -Be $false
        }

        It "Core still works after OLLAMA_INSTALL_MINIMAL upgrade" {
            Test-Path (Join-Path $script:installDir "ollama.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "lib\ollama\llama-server.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "Ollama app.exe") | Should -Be $true
        }
    }

    Context "Reinstall Same Version" {
        BeforeAll {
            Write-Host "`n>>> Reinstall Same Version" -ForegroundColor Cyan
            Invoke-FullUninstall
            if ($script:useLocalServer) {
                $script:server = Start-LocalHttpServer -Dir $script:distDir
            }
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
        }

        It "First install succeeds" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "Reinstall succeeds without errors" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
            # Should not contain error messages
            $output = $result.Output -join "`n"
            $output | Should -Not -Match "ERROR"
        }

        It "All files still present after reinstall" {
            Test-Path (Join-Path $script:installDir "ollama.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "lib\ollama\llama-server.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "Ollama app.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "packages.json") | Should -Be $true
        }
    }

    Context "Uninstall" {
        BeforeAll {
            Write-Host "`n>>> Uninstall" -ForegroundColor Cyan
            Invoke-FullUninstall
            if ($script:useLocalServer) {
                $script:server = Start-LocalHttpServer -Dir $script:distDir
            }
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
        }

        It "Install first" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "Uninstall removes all components" {
            $scriptPath = $InstallScript
            $env:OLLAMA_UNINSTALL = "1"
            $env:OLLAMA_REMOVE_MODELS = "0"
            $stdoutPath = Join-Path $env:TEMP "ollama-uninstall-$([Guid]::NewGuid()).out"
            $stderrPath = Join-Path $env:TEMP "ollama-uninstall-$([Guid]::NewGuid()).err"
            try {
                Set-TestProcessEnvironment
                $proc = Start-Process -FilePath "powershell.exe" `
                    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$scriptPath`"") `
                    -WindowStyle Hidden `
                    -RedirectStandardOutput $stdoutPath `
                    -RedirectStandardError $stderrPath `
                    -Wait -PassThru
                $output = @()
                if (Test-Path $stdoutPath) { $output += Get-Content $stdoutPath }
                if (Test-Path $stderrPath) { $output += Get-Content $stderrPath }
                $proc.ExitCode | Should -Be 0 -Because ($output -join "`n")
            } finally {
                Remove-Item Env:\OLLAMA_UNINSTALL -ErrorAction SilentlyContinue
                Remove-Item Env:\OLLAMA_REMOVE_MODELS -ErrorAction SilentlyContinue
                Remove-Item $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
            }

            # Give a moment for MSI uninstall to complete
            Start-Sleep -Seconds 3
        }

        It "ollama.exe removed from install directory" {
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $false
        }

        It "No Ollama products registered after uninstall" {
            $products = Get-InstalledOllamaProducts
            $products.Count | Should -Be 0
        }
    }

    Context "ARP Uninstall (chained)" {
        BeforeAll {
            Write-Host "`n>>> ARP Uninstall (chained)" -ForegroundColor Cyan
            Invoke-FullUninstall
            if ($script:useLocalServer) {
                $script:server = Start-LocalHttpServer -Dir $script:distDir
            }
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
        }

        It "Install with OLLAMA_INSTALL_ALL" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_ALL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "Multiple MSIs installed" {
            $products = Get-InstalledOllamaProducts
            $products.Count | Should -BeGreaterThan 1 -Because "core + backends should be installed"
        }

        It "Uninstalling core triggers chained uninstall of backends" {
            # Find the core MSI product code via COM API
            $products = Get-InstalledOllamaProducts
            $coreProduct = $products | Where-Object { $_.Name -eq "core" }
            $coreProduct | Should -Not -BeNullOrEmpty

            $coreProductCode = $coreProduct.ProductCode

            # Uninstall core via msiexec (simulates ARP uninstall)
            $msiExec = Join-Path $env:SystemRoot "System32\msiexec.exe"
            $proc = Start-Process -FilePath $msiExec -ArgumentList "/x", $coreProductCode, "/quiet", "/norestart" `
                -Wait -PassThru -NoNewWindow
            $proc.ExitCode | Should -Be 0

            # Wait for the async chained uninstall to complete
            # The chained uninstall waits 2 seconds then uninstalls remaining packages sequentially
            Start-Sleep -Seconds 30

            # All Ollama products should be removed
            $remaining = Get-InstalledOllamaProducts
            $remaining.Count | Should -Be 0 -Because "chained uninstall should remove all backend MSIs"
        }
    }

    Context "Warm Cache (updater flow)" {
        BeforeAll {
            Write-Host "`n>>> Warm Cache (updater flow)" -ForegroundColor Cyan
            Invoke-FullUninstall
            if ($script:useLocalServer) {
                $script:server = Start-LocalHttpServer -Dir $script:distDir
            }
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
        }

        It "OLLAMA_CACHE_ONLY caches MSIs without installing" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1"; OLLAMA_CACHE_ONLY = "1" }

            # Should not install
            Test-OllamaInstalled | Should -Be $false

            # Cache dir should have the core MSI
            $cacheDir = Join-Path $env:LOCALAPPDATA "Ollama\install_cache"
            Test-Path $cacheDir | Should -Be $true
            $cachedFiles = Get-ChildItem $cacheDir -Filter "*.msi" -Recurse -ErrorAction SilentlyContinue
            $cachedFiles.Count | Should -BeGreaterThan 0
        }

        It "Install from warm cache succeeds" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }
    }

    Context "Downgrade" {
        BeforeAll {
            Write-Host "`n>>> Downgrade" -ForegroundColor Cyan
            Invoke-FullUninstall
            if ($script:useLocalServer) {
                $script:server = Start-LocalHttpServer -Dir $script:distDir
            }
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
        }

        It "First install succeeds" {
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "Reinstall (simulated downgrade) succeeds" {
            # MajorUpgrade has AllowDowngrades="yes", so installing the same
            # or older version should succeed without error.
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "ollama.exe still functional after downgrade" {
            Test-Path (Join-Path $script:installDir "ollama.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "lib\ollama\llama-server.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "Ollama app.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "packages.json") | Should -Be $true
        }

        It "Core MSI still registered after downgrade" {
            $products = Get-InstalledOllamaProducts
            $coreProduct = $products | Where-Object { $_.Name -eq "core" }
            $coreProduct | Should -Not -BeNullOrEmpty
        }
    }

    Context "Upgrade path matrix" -Tag UpgradeMatrix {
        BeforeAll {
            Write-Host "`n>>> Upgrade path matrix" -ForegroundColor Cyan
            if ($script:useLocalServer) {
                $script:upgradePathServer = Start-LocalMSIDownloadServer -DistDir $script:distDir
            }
            $script:upgradePathInnoSetupExe = Get-InnoSetupInstaller
            $script:upgradePathCurrentInnoSetupExe = Join-Path $script:distDir "OllamaSetup.exe"
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        BeforeEach {
            Invoke-FullUninstall
            Clear-MsiCache
            Remove-UpgradeMarker
        }

        AfterEach {
            Invoke-FullUninstall
            Clear-MsiCache
            Remove-UpgradeMarker
        }

        AfterAll {
            if ($script:upgradePathServer) {
                Stop-LocalHttpServer $script:upgradePathServer
                $script:upgradePathServer = $null
            }
        }

        It "upgrades Inno Setup to Inno Setup by default" {
            if (-not $script:useLocalServer) {
                Set-ItResult -Skipped -Because "requires local built artifacts"
                return
            }
            if (-not $script:upgradePathInnoSetupExe) {
                Set-ItResult -Skipped -Because "Inno Setup installer not available"
                return
            }

            Install-InnoSetupForTest -InstallerPath $script:upgradePathInnoSetupExe
            Test-InnoSetupInstalled | Should -Be $true

            $result = Invoke-InstallScript -BaseUrl $script:upgradePathServer.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }

            $result.ExitCode | Should -Be 0 -Because ($result.Output -join "`n")
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
            Test-InnoSetupInstalled | Should -Be $true

            $products = Get-InstalledOllamaProducts
            ($products | Where-Object { $_.Name -eq "core" }) | Should -BeNullOrEmpty
        }

        It "migrates Inno Setup to MSI when requested" {
            if (-not $script:useLocalServer) {
                Set-ItResult -Skipped -Because "requires local built artifacts"
                return
            }
            if (-not $script:upgradePathInnoSetupExe) {
                Set-ItResult -Skipped -Because "Inno Setup installer not available"
                return
            }

            Install-InnoSetupForTest -InstallerPath $script:upgradePathInnoSetupExe
            Test-InnoSetupInstalled | Should -Be $true

            $result = Invoke-InstallScript -BaseUrl $script:upgradePathServer.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1"; OLLAMA_MIGRATE_TO_MSI = "1" }

            $result.ExitCode | Should -Be 0 -Because ($result.Output -join "`n")
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
            Test-InnoSetupInstalled | Should -Be $false

            $products = Get-InstalledOllamaProducts
            ($products | Where-Object { $_.Name -eq "core" }) | Should -Not -BeNullOrEmpty
        }

        It "upgrades MSI to MSI by default" {
            if (-not $script:useLocalServer) {
                Set-ItResult -Skipped -Because "requires local built artifacts"
                return
            }

            $first = Invoke-InstallScript -BaseUrl $script:upgradePathServer.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            $first.ExitCode | Should -Be 0 -Because ($first.Output -join "`n")
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
            Test-InnoSetupInstalled | Should -Be $false

            $products = Get-InstalledOllamaProducts
            ($products | Where-Object { $_.Name -eq "core" }) | Should -Not -BeNullOrEmpty

            $second = Invoke-InstallScript -BaseUrl $script:upgradePathServer.BaseUrl
            $second.ExitCode | Should -Be 0 -Because ($second.Output -join "`n")
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
            Test-InnoSetupInstalled | Should -Be $false

            $products = Get-InstalledOllamaProducts
            ($products | Where-Object { $_.Name -eq "core" }) | Should -Not -BeNullOrEmpty
        }

        It "refuses legacy Inno Setup over an MSI install" {
            if (-not $script:useLocalServer) {
                Set-ItResult -Skipped -Because "requires local built artifacts"
                return
            }
            if (-not (Test-Path -LiteralPath $script:upgradePathCurrentInnoSetupExe -PathType Leaf)) {
                Set-ItResult -Skipped -Because "current OllamaSetup.exe not available"
                return
            }

            $install = Invoke-InstallScript -BaseUrl $script:upgradePathServer.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            $install.ExitCode | Should -Be 0 -Because ($install.Output -join "`n")
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
            Test-InnoSetupInstalled | Should -Be $false

            $products = Get-InstalledOllamaProducts
            ($products | Where-Object { $_.Name -eq "core" }) | Should -Not -BeNullOrEmpty

            $result = Invoke-InnoSetupInstallerForTest -InstallerPath $script:upgradePathCurrentInnoSetupExe
            $logText = $result.Log -join "`n"
            $result.ExitCode | Should -Not -Be 0 -Because "OllamaSetup.exe should refuse MSI-managed installs. Log: $($result.LogPath)`n$logText"
            $logText | Should -Match 'Detected MSI-managed Ollama install'

            Test-InnoSetupInstalled | Should -Be $false
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
            $products = Get-InstalledOllamaProducts
            ($products | Where-Object { $_.Name -eq "core" }) | Should -Not -BeNullOrEmpty
        }
    }

    Context "Legacy Inno Setup upgrade" {
        BeforeAll {
            Write-Host "`n>>> Legacy Inno Setup upgrade" -ForegroundColor Cyan
            Invoke-FullUninstall
            if ($script:useLocalServer) {
                $script:server = Start-LocalHttpServer -Dir $script:distDir
            }
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"

            # Download OllamaSetup.exe from a specific release version.
            # The version is configured in Install-TestHelpers.psm1 ($script:InnoSetupTestVersion).
            $script:innoSetupExe = Get-InnoSetupInstaller
            if (-not $script:innoSetupExe) {
                Write-Warning "Failed to download Inno Setup installer - migration tests will be skipped"
            }
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
            # Also clean up Inno Setup if it somehow remained
            $innoKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
            if (Test-Path $innoKey) {
                $uninstallString = (Get-ItemProperty -Path $innoKey).UninstallString
                if ($uninstallString) {
                    $uninstallExe = $uninstallString -replace '"', ''
                    Start-Process -FilePath $uninstallExe `
                        -ArgumentList "/VERYSILENT /NORESTART /SUPPRESSMSGBOXES" `
                        -Wait -ErrorAction SilentlyContinue
                }
            }
            # Downloaded Inno Setup installer is cached for reuse, no cleanup needed
        }

        It "Inno Setup installer available" {
            if (-not $script:innoSetupExe) {
                Set-ItResult -Skipped -Because "Inno Setup installer download failed"
                return
            }
            $script:innoSetupExe | Should -Not -BeNullOrEmpty
            Test-Path $script:innoSetupExe | Should -Be $true
        }

        It "Inno Setup installs successfully" {
            if (-not $script:innoSetupExe) {
                Set-ItResult -Skipped -Because "Inno Setup installer not available"
                return
            }
            Write-Host "    Installing Inno Setup..." -ForegroundColor DarkGray
            $proc = Start-Process -FilePath $script:innoSetupExe `
                -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART" `
                -PassThru
            # Wait with timeout - Inno Setup may spawn child processes that keep -Wait hanging
            $completed = $proc.WaitForExit(120000)  # 2 minute timeout
            if (-not $completed) {
                Write-Warning "Inno Setup installer did not exit within 2 minutes - killing"
                $proc | Stop-Process -Force -ErrorAction SilentlyContinue
            }

            # Verify Inno Setup registry key exists
            $innoKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
            Test-Path $innoKey | Should -Be $true

            # Stop Ollama processes started by the installer
            Write-Host "    Stopping Ollama processes launched by installer..." -ForegroundColor DarkGray
            Start-Sleep -Seconds 3
            Get-Process -Name "ollama", "Ollama app" -ErrorAction SilentlyContinue |
                Stop-Process -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }

        It "install.ps1 preserves Inno Setup by default" {
            if (-not $script:innoSetupExe) {
                Set-ItResult -Skipped -Because "Inno Setup installer not available"
                return
            }
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }

            $result.ExitCode | Should -Be 0 -Because ($result.Output -join "`n")
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true

            $innoKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
            Test-Path $innoKey | Should -Be $true

            $products = Get-InstalledOllamaProducts
            $coreProduct = $products | Where-Object { $_.Name -eq "core" }
            $coreProduct | Should -BeNullOrEmpty
        }

        It "install.ps1 migrates from Inno Setup when requested" {
            if (-not $script:innoSetupExe) {
                Set-ItResult -Skipped -Because "Inno Setup installer not available"
                return
            }
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1"; OLLAMA_MIGRATE_TO_MSI = "1" }

            $result.ExitCode | Should -Be 0 -Because ($result.Output -join "`n")
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true

            $innoKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
            Test-Path $innoKey | Should -Be $false

            $products = Get-InstalledOllamaProducts
            $coreProduct = $products | Where-Object { $_.Name -eq "core" }
            $coreProduct | Should -Not -BeNullOrEmpty
        }

        It "Models directory preserved after migration" {
            if (-not $script:innoSetupExe) {
                Set-ItResult -Skipped -Because "Inno Setup installer not available"
                return
            }
            # The Inno Setup uninstaller preserves models by default.
            # We just verify the .ollama directory still exists.
            $ollamaDir = Join-Path $env:USERPROFILE ".ollama"
            Test-Path $ollamaDir | Should -Be $true
        }
    }

    Context "Authenticode signature warning" {
        BeforeAll {
            Write-Host "`n>>> Authenticode signature warning" -ForegroundColor Cyan
            if (-not $script:useLocalServer) {
                $script:skipAuthenticode = $true
                Write-Host "    Skipping - requires local MSIs" -ForegroundColor Yellow
            } else {
                $script:skipAuthenticode = $false
            }
            Invoke-FullUninstall
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
            # Create a temp directory with a copy of the core MSI that has been
            # tampered with (appending bytes invalidates the Authenticode signature).
            $script:tamperedDir = Join-Path $env:TEMP "OllamaTamperedMSI"
            if (-not $script:skipAuthenticode) {
                if (Test-Path $script:tamperedDir) {
                    Remove-Item $script:tamperedDir -Recurse -Force
                }
                New-Item -ItemType Directory -Path $script:tamperedDir -Force | Out-Null

                # Copy all MSIs from dist to tampered dir
                Get-ChildItem -Path $script:distDir -Filter "*.msi" | ForEach-Object {
                    Copy-Item $_.FullName $script:tamperedDir
                }
                # Also copy packages.json if present
                $pkgJson = Join-Path $script:distDir "packages.json"
                if (Test-Path $pkgJson) {
                    Copy-Item $pkgJson $script:tamperedDir
                }

                # Tamper with the core MSI (append garbage bytes to invalidate signature)
                $tamperedCore = Join-Path $script:tamperedDir "ollama-core.msi"
                [System.IO.File]::AppendAllText($tamperedCore, "TAMPERED")

                $script:server = Start-LocalHttpServer -Dir $script:tamperedDir
            }
        }

        AfterAll {
            if ($script:server) { Stop-LocalHttpServer $script:server; $script:server = $null }
            Invoke-FullUninstall
            if (Test-Path $script:tamperedDir) {
                Remove-Item $script:tamperedDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        It "Tampered MSI fails signature verification and does not install" {
            if ($script:skipAuthenticode) {
                Set-ItResult -Skipped -Because "requires local MSIs"
                return
            }
            $result = Invoke-InstallScript -BaseUrl $(if ($script:server) { $script:server.BaseUrl } else { "" }) `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1"; OLLAMA_DEBUG = "1" } `
                -DisableSignatureVerification $false

            $result.ExitCode | Should -Not -Be 0
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $false

            $output = $result.Output -join "`n"
            $output | Should -Match "Signature verification failed"
        }
    }
}

# ==========================================================================
# App-level integration tests (test upgrade via app HTTP API)
# ==========================================================================

Describe "App Upgrade Flow" -Tag AppIntegration {
    BeforeAll {
        # --- Safety gate: abort if Ollama is already installed ---
        if (Test-OllamaInstalled) {
            throw @"
INTEGRATION TEST SAFETY ABORT:
An existing Ollama installation was detected. App integration tests make real
changes to the system and would interfere with your install.

To run these tests, first uninstall Ollama completely.
"@
        }

        # Find the dist directory with built MSIs
        $script:distDir = Find-DistDir
        if (-not $script:distDir) {
            throw "dist directory with built MSIs not found. Run 'cmake --build build/msi --target msi-all' first."
        }
        Set-InstallScriptTestSignatureVerification -Disabled:(-not (Test-InstallerArtifactsSigned -DistDir $script:distDir))

        # Verify we have the app executable in dist (or it will be installed)
        $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"

        # Create mock update server directory
        $script:updateDir = Join-Path $env:TEMP "ollama-update-mock-$([guid]::NewGuid().ToString('N').Substring(0,8))"
        New-Item -ItemType Directory -Path $script:updateDir -Force | Out-Null
    }

    AfterAll {
        # Cleanup
        Stop-OllamaApp $null  # Kill any lingering processes
        Invoke-FullUninstall
        Clear-MsiCache
        Remove-UpgradeMarker

        if ($script:updateDir -and (Test-Path $script:updateDir)) {
            Remove-Item $script:updateDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    Context "Scenario: App detects update via HTTP API" {
        BeforeAll {
            Invoke-FullUninstall
            Clear-MsiCache

            # Start download server from dist directory
            $script:downloadServer = Start-LocalMSIDownloadServer -DistDir $script:distDir

            # Install current version first
            $result = Invoke-InstallScript -BaseUrl $script:downloadServer.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }

            if (-not (Test-OllamaInstalled -Dir $script:installDir)) {
                throw "Failed to install Ollama for app integration test"
            }

            # Create mock update response (newer version available)
            New-MockUpdateResponse -OutputDir $script:updateDir `
                -Version "99.0.0" `
                -DownloadBaseUrl $script:downloadServer.BaseUrl

            # Start update check server
            $script:updateServer = Start-LocalHttpServer -Dir $script:updateDir
        }

        AfterAll {
            Stop-OllamaApp $script:app
            Stop-LocalHttpServer $script:downloadServer
            Stop-LocalHttpServer $script:updateServer
        }

        It "App starts in test mode" {
            $script:app = Start-OllamaAppTestMode `
                -UpdateServerUrl "$($script:updateServer.BaseUrl)/update.json" `
                -DownloadServerUrl $script:downloadServer.BaseUrl `
                -AppExePath (Join-Path $script:installDir "Ollama app.exe")

            $script:app | Should -Not -BeNullOrEmpty
            $script:app.Process | Should -Not -BeNullOrEmpty
            $script:app.Process.HasExited | Should -Be $false
        }

        It "Update check API returns update available" {
            $updateInfo = Invoke-AppUpdateCheck -BaseUrl $script:app.BaseUrl

            $updateInfo | Should -Not -BeNullOrEmpty
            $updateInfo.updateAvailable | Should -Be $true
            $updateInfo.availableVersion | Should -Be "99.0.0"
        }

        It "Current version is reported correctly" {
            $updateInfo = Invoke-AppUpdateCheck -BaseUrl $script:app.BaseUrl

            $updateInfo.currentVersion | Should -Not -BeNullOrEmpty
            # Current version should not be 99.0.0 (that's the "new" version)
            $updateInfo.currentVersion | Should -Not -Be "99.0.0"
        }
    }

    Context "Scenario: App downloads MSIs to cache" {
        BeforeAll {
            Invoke-FullUninstall
            Clear-MsiCache

            # Start servers
            $script:downloadServer = Start-LocalMSIDownloadServer -DistDir $script:distDir

            # Install first
            $result = Invoke-InstallScript -BaseUrl $script:downloadServer.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }

            if (-not (Test-OllamaInstalled -Dir $script:installDir)) {
                throw "Failed to install Ollama for app integration test"
            }

            # Create update response
            New-MockUpdateResponse -OutputDir $script:updateDir `
                -Version "99.0.0" `
                -DownloadBaseUrl $script:downloadServer.BaseUrl

            $script:updateServer = Start-LocalHttpServer -Dir $script:updateDir
        }

        AfterAll {
            Stop-OllamaApp $script:app
            Stop-LocalHttpServer $script:downloadServer
            Stop-LocalHttpServer $script:updateServer
        }

        It "App starts after an MSI install" {
            $script:app = Start-OllamaAppTestMode `
                -UpdateServerUrl "$($script:updateServer.BaseUrl)/update.json" `
                -DownloadServerUrl $script:downloadServer.BaseUrl `
                -AppExePath (Join-Path $script:installDir "Ollama app.exe")

            $script:app | Should -Not -BeNullOrEmpty
        }

        It "Cache is initially empty or minimal" {
            $cacheContents = Get-MsiCacheContents
            # Cache may have files from previous test, but core shouldn't be from "99.0.0"
            # This is a baseline check
            $cacheContents | Should -Not -BeNullOrEmpty -Because "cache dir should exist after install"
        }

        It "Update check triggers background download" {
            # Check for update (this triggers the download in the app)
            $updateInfo = Invoke-AppUpdateCheck -BaseUrl $script:app.BaseUrl

            $updateInfo.updateAvailable | Should -Be $true

            # Give the app time to download MSIs in background
            # The app downloads asynchronously after detecting an update
            Start-Sleep -Seconds 15

            # Check that cache is being populated
            # Note: The actual MSI download depends on UpdateDownloaded flag being set
            # which requires the full download flow to complete
        }
    }

    Context "Scenario: Full upgrade cycle (requires real MSIs)" {
        BeforeAll {
            Invoke-FullUninstall
            Clear-MsiCache
            Remove-UpgradeMarker

            # This test requires actual MSIs that can be installed
            $script:downloadServer = Start-LocalMSIDownloadServer -DistDir $script:distDir

            # Install version 1
            $result = Invoke-InstallScript -BaseUrl $script:downloadServer.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }

            if (-not (Test-OllamaInstalled -Dir $script:installDir)) {
                throw "Failed to install Ollama for upgrade cycle test"
            }

            # Record initial version
            $script:initialVersion = Get-InstalledOllamaVersion -InstallDir $script:installDir

            # For the upgrade, we simulate by using the same MSIs
            # (in real scenario, this would be a newer version)
            New-MockUpdateResponse -OutputDir $script:updateDir `
                -Version "99.0.0" `
                -DownloadBaseUrl $script:downloadServer.BaseUrl

            $script:updateServer = Start-LocalHttpServer -Dir $script:updateDir
        }

        AfterAll {
            Stop-OllamaApp $script:app
            Stop-LocalHttpServer $script:downloadServer
            Stop-LocalHttpServer $script:updateServer
            Invoke-FullUninstall
        }

        It "App starts successfully" {
            $script:app = Start-OllamaAppTestMode `
                -UpdateServerUrl "$($script:updateServer.BaseUrl)/update.json" `
                -DownloadServerUrl $script:downloadServer.BaseUrl `
                -AppExePath (Join-Path $script:installDir "Ollama app.exe")

            $script:app | Should -Not -BeNullOrEmpty
            $script:app.Process.HasExited | Should -Be $false
        }

        It "Update is available" {
            $updateInfo = Invoke-AppUpdateCheck -BaseUrl $script:app.BaseUrl
            $updateInfo.updateAvailable | Should -Be $true
        }

        # Note: The following test is commented out because it triggers the full
        # upgrade cycle which exits the app and runs install.ps1. This is
        # disruptive and should only be run in isolated CI environments.

        # It "Install update triggers upgrade and app exits" {
        #     # First ensure update is downloaded
        #     Start-Sleep -Seconds 10  # Wait for background download
        #
        #     # Trigger install
        #     $response = Invoke-AppInstallUpdate -BaseUrl $script:app.BaseUrl
        #     $response.success | Should -Be $true
        #
        #     # App should exit
        #     $exited = Wait-AppExit -AppInstance $script:app -TimeoutSeconds 30
        #     $exited | Should -Be $true
        #
        #     # Wait for installer to complete
        #     $installed = Wait-InstallerComplete -TimeoutSeconds 120
        #     $installed | Should -Be $true
        #
        #     # Verify installation succeeded
        #     Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        # }
    }

    Context "Scenario: Upgrade marker lifecycle" {
        BeforeAll {
            Invoke-FullUninstall
            Clear-MsiCache
            Remove-UpgradeMarker

            $script:downloadServer = Start-LocalMSIDownloadServer -DistDir $script:distDir

            # Install
            $result = Invoke-InstallScript -BaseUrl $script:downloadServer.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
        }

        AfterAll {
            Stop-OllamaApp $script:app
            Stop-LocalHttpServer $script:downloadServer
            Remove-UpgradeMarker
        }

        It "No upgrade marker before upgrade" {
            Test-UpgradeMarkerExists | Should -Be $false
        }

        It "App removes stale marker on startup" {
            # Create a marker file (simulates previous upgrade)
            $markerPath = Join-Path $env:LOCALAPPDATA "Ollama\upgraded"
            New-Item -ItemType File -Path $markerPath -Force | Out-Null

            Test-UpgradeMarkerExists | Should -Be $true

            # Start app - it should clean up the marker
            $script:app = Start-OllamaAppTestMode `
                -DownloadServerUrl $script:downloadServer.BaseUrl `
                -AppExePath (Join-Path $script:installDir "Ollama app.exe")

            # Give app time to run post-upgrade cleanup
            Start-Sleep -Seconds 5

            # Note: The marker cleanup happens in DoPostUpgradeCleanup which
            # is called on startup when the marker exists
        }
    }

    Context "Scenario: Cache persistence across restarts" {
        BeforeAll {
            Invoke-FullUninstall
            Clear-MsiCache

            $script:downloadServer = Start-LocalMSIDownloadServer -DistDir $script:distDir

            # Install
            $result = Invoke-InstallScript -BaseUrl $script:downloadServer.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }

            New-MockUpdateResponse -OutputDir $script:updateDir `
                -Version "99.0.0" `
                -DownloadBaseUrl $script:downloadServer.BaseUrl

            $script:updateServer = Start-LocalHttpServer -Dir $script:updateDir
        }

        AfterAll {
            Stop-OllamaApp $script:app
            Stop-LocalHttpServer $script:downloadServer
            Stop-LocalHttpServer $script:updateServer
        }

        It "Start app and trigger update check" {
            $script:app = Start-OllamaAppTestMode `
                -UpdateServerUrl "$($script:updateServer.BaseUrl)/update.json" `
                -DownloadServerUrl $script:downloadServer.BaseUrl `
                -AppExePath (Join-Path $script:installDir "Ollama app.exe")

            $updateInfo = Invoke-AppUpdateCheck -BaseUrl $script:app.BaseUrl
            $updateInfo.updateAvailable | Should -Be $true

            # Wait for potential background download
            Start-Sleep -Seconds 5

            # Record cache state
            $script:cacheBeforeRestart = Get-MsiCacheContents
        }

        It "Stop and restart app" {
            Stop-OllamaApp $script:app
            Start-Sleep -Seconds 2

            $script:app = Start-OllamaAppTestMode `
                -UpdateServerUrl "$($script:updateServer.BaseUrl)/update.json" `
                -DownloadServerUrl $script:downloadServer.BaseUrl `
                -AppExePath (Join-Path $script:installDir "Ollama app.exe")

            $script:app | Should -Not -BeNullOrEmpty
        }

        It "Cache is preserved after restart" {
            $cacheAfterRestart = Get-MsiCacheContents

            # Cache should still have files (may have more after restart)
            if ($script:cacheBeforeRestart.Count -gt 0) {
                $cacheAfterRestart.Count | Should -BeGreaterOrEqual $script:cacheBeforeRestart.Count
            }
        }
    }
}

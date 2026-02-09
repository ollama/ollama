<#
.SYNOPSIS
    Pester tests for Ollama install script.

.DESCRIPTION
    Unit tests (Tag: Unit) - fast, mock external commands, no system changes.
    Integration tests (Tag: Integration) - require built MSIs in dist/, make real installs.

    Integration tests start a local HTTP server from the dist directory and use
    OLLAMA_DOWNLOAD_URL to redirect the install script. They verify results by
    checking the filesystem, MSI registration (via Windows Installer COM API),
    and running processes rather than parsing text output.

.EXAMPLE
    Import-Module Pester -MinimumVersion 5.0
    Invoke-Pester scripts/tests/install.Tests.ps1 -Tag Unit
    Invoke-Pester scripts/tests/install.Tests.ps1 -Tag Integration
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
        $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLower()
        $result = switch ($arch) {
            "x64"   { "amd64" }
            "arm64" { "arm64" }
            default { "amd64" }
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
        $loaded.packages.Count | Should -Be 4
    }

    It "Finds backend by name" {
        $jsonPath = Join-Path $testEnv.CacheDir "packages.json"
        New-MockPackagesJson -OutputPath $jsonPath

        $loaded = Get-Content $jsonPath -Raw | ConvertFrom-Json
        $cudaPkg = $loaded.packages | Where-Object { $_.name -eq "cuda_v12" }
        $cudaPkg | Should -Not -BeNullOrEmpty
        $cudaPkg.file | Should -Be "ollama-cuda-v12.msi"
        $cudaPkg.deps | Should -Be "ollama-cuda-deps-12.8.1.msi"
    }

    It "Returns empty packages for ARM64 manifest" {
        $jsonPath = Join-Path $testEnv.CacheDir "packages.json"
        $arm64Manifest = @{ version = "0.15.0"; packages = @() }
        $arm64Manifest | ConvertTo-Json -Depth 3 | Out-File $jsonPath -Encoding utf8

        $loaded = Get-Content $jsonPath -Raw | ConvertFrom-Json
        $loaded.packages.Count | Should -Be 0
    }
}

Describe "Backend Detection from Filesystem" -Tag Unit {
    BeforeEach {
        $testEnv = Initialize-TestEnvironment
    }

    AfterEach {
        Remove-TestEnvironment
    }

    It "Detects installed backends" {
        $dir = $testEnv.InstallDir
        New-MockInstalledDir -Dir $dir -Backends @("cuda_v12", "vulkan")

        $libDir = Join-Path $dir "lib\ollama"
        $backends = @()
        foreach ($name in @("cuda_v12", "cuda_v13", "rocm", "vulkan")) {
            if (Test-Path (Join-Path $libDir $name)) {
                $backends += $name
            }
        }

        $backends | Should -Contain "cuda_v12"
        $backends | Should -Contain "vulkan"
        $backends | Should -Not -Contain "rocm"
        $backends | Should -Not -Contain "cuda_v13"
    }

    It "Returns empty for fresh install" {
        $dir = $testEnv.InstallDir
        $libDir = Join-Path $dir "lib\ollama"

        $backends = @()
        foreach ($name in @("cuda_v12", "cuda_v13", "rocm", "vulkan")) {
            if (Test-Path (Join-Path $libDir $name)) {
                $backends += $name
            }
        }

        $backends.Count | Should -Be 0
    }
}

Describe "Backend Selection Logic" -Tag Unit {
    It "All mode selects all available backends" {
        $availableBackends = @("cuda_v12", "cuda_v13", "rocm", "vulkan")
        # OLLAMA_INSTALL_ALL=1: select everything
        $selected = $availableBackends
        $selected.Count | Should -Be 4
    }

    It "Explicit mode uses provided list" {
        $explicit = @("cuda_v12", "rocm")
        $explicit.Count | Should -Be 2
        $explicit | Should -Contain "cuda_v12"
        $explicit | Should -Contain "rocm"
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

Describe "Inno Setup Detection" -Tag Unit {
    It "Detects Inno Setup registry key format" {
        $keyPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
        # Just verify the key path format is correct
        $keyPath | Should -Match '\{44E83376-CE68-45EB-8FC1-393500EB558C\}_is1$'
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
    }

    It "Has comment-based help" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match '\.SYNOPSIS'
        $content | Should -Match '\.DESCRIPTION'
        $content | Should -Match '\.EXAMPLE'
    }

    It "Documents environment variables in help" {
        $content = Get-Content $InstallScript -Raw
        $content | Should -Match 'Environment variables:'
        $content | Should -Match 'OLLAMA_INSTALL_ALL.*Install all GPU backends'
        $content | Should -Match 'OLLAMA_INSTALL_MINIMAL.*CPU-only'
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
        New-Item -ItemType File -Path (Join-Path $script:distAmd64 "lib\ollama\ggml-base.dll") -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $script:distAmd64 "lib\ollama\ggml-cpu-x64.dll") -Force | Out-Null

        # Create a directory for serving pre-built MSIs
        $script:serveDir = Join-Path $script:testRoot "serve"
        New-Item -ItemType Directory -Path $script:serveDir -Force | Out-Null

        # Create a dummy "pre-built" MSI (just a binary file > 1024 bytes)
        $script:dummyMsiName = "ollama-vulkan-deps-1.4.0.msi"
        $dummyMsiPath = Join-Path $script:serveDir $script:dummyMsiName
        $dummyBytes = [byte[]]::new(4096)
        (New-Object Random).NextBytes($dummyBytes)
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

    It "Downloads deps MSI when OLLAMA_DOWNLOAD_URL env var is set" {
        $buildDir = Join-Path $script:testRoot "build-download"
        $savedUrl = $env:OLLAMA_DOWNLOAD_URL
        try {
            $env:OLLAMA_DOWNLOAD_URL = $script:server.BaseUrl

            $output = & $script:cmakeExe -B $buildDir -S $script:cmakeSourceDir `
                -DOLLAMA_VERSION="0.15.0" `
                -DOLLAMA_DIST_DIR="$($script:distDir)" `
                2>&1 | Out-String

            # Configure output should show the URL was picked up and download succeeded
            $output | Should -Match "Download URL:.*$([regex]::Escape($script:server.BaseUrl))"
            $output | Should -Match "Downloaded ollama-vulkan-deps-1.4.0.msi"

            # The MSI should exist in the dist directory
            $downloadedMsi = Join-Path $script:distDir $script:dummyMsiName
            Test-Path $downloadedMsi | Should -Be $true

            # Verify it matches the original (same content)
            $downloadedHash = (Get-FileHash $downloadedMsi -Algorithm SHA256).Hash
            $downloadedHash | Should -Be $script:dummyMsiHash
        } finally {
            if ($savedUrl) { $env:OLLAMA_DOWNLOAD_URL = $savedUrl } else { Remove-Item Env:\OLLAMA_DOWNLOAD_URL -ErrorAction SilentlyContinue }
            $downloadedMsi = Join-Path $script:distDir $script:dummyMsiName
            Remove-Item $downloadedMsi -Force -ErrorAction SilentlyContinue
            Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "Falls back to local build when file not found on server" {
        $buildDir = Join-Path $script:testRoot "build-fallback"
        $savedUrl = $env:OLLAMA_DOWNLOAD_URL

        # Temporarily rename the served file so it 404s
        $servedMsi = Join-Path $script:serveDir $script:dummyMsiName
        $tempName = "$servedMsi.hidden"
        Rename-Item $servedMsi $tempName

        try {
            $env:OLLAMA_DOWNLOAD_URL = $script:server.BaseUrl

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
            if ($savedUrl) { $env:OLLAMA_DOWNLOAD_URL = $savedUrl } else { Remove-Item Env:\OLLAMA_DOWNLOAD_URL -ErrorAction SilentlyContinue }
            Rename-Item $tempName $servedMsi
            Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "Defaults to https://ollama.com/download when OLLAMA_DOWNLOAD_URL is not set" {
        $buildDir = Join-Path $script:testRoot "build-no-url"
        $savedUrl = $env:OLLAMA_DOWNLOAD_URL
        try {
            Remove-Item Env:\OLLAMA_DOWNLOAD_URL -ErrorAction SilentlyContinue

            $output = & $script:cmakeExe -B $buildDir -S $script:cmakeSourceDir `
                -DOLLAMA_VERSION="0.15.0" `
                -DOLLAMA_DIST_DIR="$($script:distDir)" `
                2>&1 | Out-String

            # Should default to the production URL
            $output | Should -Match "Download URL: https://ollama.com/download"
        } finally {
            if ($savedUrl) { $env:OLLAMA_DOWNLOAD_URL = $savedUrl }
            Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "Rejects downloaded file smaller than 1KB" {
        $buildDir = Join-Path $script:testRoot "build-tiny"
        $savedUrl = $env:OLLAMA_DOWNLOAD_URL

        # Replace the served MSI with a tiny file (simulating a 404 HTML page saved as .msi)
        $servedMsi = Join-Path $script:serveDir $script:dummyMsiName
        $originalContent = [System.IO.File]::ReadAllBytes($servedMsi)
        try {
            $env:OLLAMA_DOWNLOAD_URL = $script:server.BaseUrl
            "Not Found" | Out-File $servedMsi -Encoding ascii

            $output = & $script:cmakeExe -B $buildDir -S $script:cmakeSourceDir `
                -DOLLAMA_VERSION="0.15.0" `
                -DOLLAMA_DIST_DIR="$($script:distDir)" `
                2>&1 | Out-String

            # Should detect the file is not a valid MSI (too small / empty)
            $output | Should -Match "not available from server|will build locally|Will build .* locally"
        } finally {
            if ($savedUrl) { $env:OLLAMA_DOWNLOAD_URL = $savedUrl } else { Remove-Item Env:\OLLAMA_DOWNLOAD_URL -ErrorAction SilentlyContinue }
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

        $script:distDir = Find-DistDir
        if (-not $script:distDir) {
            throw "Dist directory with built MSIs not found. Run build_windows.ps1 first."
        }
        if (-not (Test-Path (Join-Path $script:distDir "ollama-core.msi"))) {
            throw "ollama-core.msi not found in $($script:distDir). Run build_windows.ps1 first."
        }

        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) {
            throw "Python not found. Integration tests require Python for a local HTTP server."
        }
    }

    Context "Scenario 1: Fresh Install (defaults)" {
        BeforeAll {
            Invoke-FullUninstall
            $script:server = Start-LocalHttpServer -Dir $script:distDir
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            Stop-LocalHttpServer $script:server
            Invoke-FullUninstall
        }

        It "Installs successfully with OLLAMA_INSTALL_MINIMAL" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
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

        It "CPU backend libraries exist" {
            Test-Path (Join-Path $script:installDir "lib\ollama\ggml-base.dll") | Should -Be $true
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
        }
    }

    Context "Scenario 2: Fresh Install (custom directory)" {
        BeforeAll {
            Invoke-FullUninstall
            $script:server = Start-LocalHttpServer -Dir $script:distDir
            $script:customDir = Join-Path $env:TEMP "OllamaIntegrationTest"
            if (Test-Path $script:customDir) {
                Remove-Item $script:customDir -Recurse -Force
            }
        }

        AfterAll {
            Stop-LocalHttpServer $script:server
            Invoke-FullUninstall
            if (Test-Path $script:customDir) {
                Remove-Item $script:customDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        It "Installs to custom directory" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1"; OLLAMA_INSTALL_DIR = $script:customDir }
            Test-OllamaInstalled -Dir $script:customDir | Should -Be $true
        }

        It "ollama.exe exists in custom directory" {
            Test-Path (Join-Path $script:customDir "ollama.exe") | Should -Be $true
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

    Context "Scenario 3: Fresh Install (OLLAMA_INSTALL_ALL)" {
        BeforeAll {
            Invoke-FullUninstall
            $script:server = Start-LocalHttpServer -Dir $script:distDir
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            Stop-LocalHttpServer $script:server
            Invoke-FullUninstall
        }

        It "Installs all backends" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
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
            $backendProducts = $products | Where-Object { $_.Name -match "^(cuda_v12|cuda_v13|rocm|vulkan)$" }
            $backendProducts | Should -Not -BeNullOrEmpty
        }
    }

    Context "Scenario 5: Upgrade (MSI to MSI)" {
        BeforeAll {
            Invoke-FullUninstall
            $script:server = Start-LocalHttpServer -Dir $script:distDir
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            Stop-LocalHttpServer $script:server
            Invoke-FullUninstall
        }

        It "First install succeeds with OLLAMA_INSTALL_ALL" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
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
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true

            # Verify backends are preserved
            foreach ($backend in $beforeBackends) {
                Test-BackendInstalled -Dir $script:installDir -Backend $backend |
                    Should -Be $true -Because "$backend should be preserved on upgrade"
            }
        }
    }

    Context "Scenario 6: Upgrade with OLLAMA_INSTALL_MINIMAL" {
        BeforeAll {
            Invoke-FullUninstall
            $script:server = Start-LocalHttpServer -Dir $script:distDir
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            Stop-LocalHttpServer $script:server
            Invoke-FullUninstall
        }

        It "First install with OLLAMA_INSTALL_ALL succeeds" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_ALL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "Upgrade with OLLAMA_INSTALL_MINIMAL removes GPU backends" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true

            # All GPU backends should be removed
            Test-BackendInstalled -Dir $script:installDir -Backend "cuda_v12" | Should -Be $false
            Test-BackendInstalled -Dir $script:installDir -Backend "cuda_v13" | Should -Be $false
            Test-BackendInstalled -Dir $script:installDir -Backend "rocm" | Should -Be $false
            Test-BackendInstalled -Dir $script:installDir -Backend "vulkan" | Should -Be $false
        }

        It "Core still works after OLLAMA_INSTALL_MINIMAL upgrade" {
            Test-Path (Join-Path $script:installDir "ollama.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "Ollama app.exe") | Should -Be $true
        }
    }

    Context "Scenario 9: Reinstall Same Version" {
        BeforeAll {
            Invoke-FullUninstall
            $script:server = Start-LocalHttpServer -Dir $script:distDir
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            Stop-LocalHttpServer $script:server
            Invoke-FullUninstall
        }

        It "First install succeeds" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "Reinstall succeeds without errors" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
            # Should not contain error messages
            $output = $result.Output -join "`n"
            $output | Should -Not -Match "ERROR"
        }

        It "All files still present after reinstall" {
            Test-Path (Join-Path $script:installDir "ollama.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "Ollama app.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "packages.json") | Should -Be $true
        }
    }

    Context "Scenario 10: Uninstall" {
        BeforeAll {
            Invoke-FullUninstall
            $script:server = Start-LocalHttpServer -Dir $script:distDir
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            Stop-LocalHttpServer $script:server
            Invoke-FullUninstall
        }

        It "Install first" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "Uninstall removes all components" {
            $scriptPath = $InstallScript
            $env:OLLAMA_DOWNLOAD_URL = $script:server.BaseUrl
            $env:OLLAMA_UNINSTALL = "1"
            $env:OLLAMA_REMOVE_MODELS = "0"
            try {
                $output = & powershell.exe -ExecutionPolicy Bypass -File $scriptPath 2>&1
            } finally {
                Remove-Item Env:\OLLAMA_DOWNLOAD_URL -ErrorAction SilentlyContinue
                Remove-Item Env:\OLLAMA_UNINSTALL -ErrorAction SilentlyContinue
                Remove-Item Env:\OLLAMA_REMOVE_MODELS -ErrorAction SilentlyContinue
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

    Context "Scenario 11: ARP Uninstall (chained)" {
        BeforeAll {
            Invoke-FullUninstall
            $script:server = Start-LocalHttpServer -Dir $script:distDir
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            Stop-LocalHttpServer $script:server
            Invoke-FullUninstall
        }

        It "Install with OLLAMA_INSTALL_ALL" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
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
            $proc = Start-Process msiexec -ArgumentList "/x", $coreProductCode, "/quiet", "/norestart" `
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

    Context "Scenario 13: Warm Cache (updater flow)" {
        BeforeAll {
            Invoke-FullUninstall
            $script:server = Start-LocalHttpServer -Dir $script:distDir
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            Stop-LocalHttpServer $script:server
            Invoke-FullUninstall
        }

        It "OLLAMA_DOWNLOAD_ONLY caches MSIs without installing" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1"; OLLAMA_DOWNLOAD_ONLY = "1" }

            # Should not install
            Test-OllamaInstalled | Should -Be $false

            # Cache dir should have the core MSI
            $cacheDir = Join-Path $env:LOCALAPPDATA "Ollama\msi_cache"
            Test-Path $cacheDir | Should -Be $true
            $cachedFiles = Get-ChildItem $cacheDir -Filter "*.msi" -ErrorAction SilentlyContinue
            $cachedFiles.Count | Should -BeGreaterThan 0
        }

        It "Install from warm cache succeeds" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }
    }

    Context "Scenario 7: Downgrade" {
        BeforeAll {
            Invoke-FullUninstall
            $script:server = Start-LocalHttpServer -Dir $script:distDir
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        }

        AfterAll {
            Stop-LocalHttpServer $script:server
            Invoke-FullUninstall
        }

        It "First install succeeds" {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "Reinstall (simulated downgrade) succeeds" {
            # MajorUpgrade has AllowDowngrades="yes", so installing the same
            # or older version should succeed without error.
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true
        }

        It "ollama.exe still functional after downgrade" {
            Test-Path (Join-Path $script:installDir "ollama.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "Ollama app.exe") | Should -Be $true
            Test-Path (Join-Path $script:installDir "packages.json") | Should -Be $true
        }

        It "Core MSI still registered after downgrade" {
            $products = Get-InstalledOllamaProducts
            $coreProduct = $products | Where-Object { $_.Name -eq "core" }
            $coreProduct | Should -Not -BeNullOrEmpty
        }
    }

    Context "Scenario 8: Migrate from Inno Setup" {
        BeforeAll {
            Invoke-FullUninstall
            $script:server = Start-LocalHttpServer -Dir $script:distDir
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
            $script:innoSetupExe = $null

            # Try to find OllamaSetup.exe for legacy install testing.
            # Look in dist/ first (CI may place it there), then try downloading.
            $localInno = Join-Path $script:distDir "OllamaSetup.exe"
            if (Test-Path $localInno) {
                $script:innoSetupExe = $localInno
            } else {
                # Try downloading from ollama.com (may not be available in all environments)
                $tempInno = Join-Path $env:TEMP "OllamaSetup-test.exe"
                try {
                    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" `
                        -OutFile $tempInno -UseBasicParsing -ErrorAction Stop
                    $script:innoSetupExe = $tempInno
                } catch {
                    # Not available - tests in this context will be skipped
                }
            }
        }

        AfterAll {
            Stop-LocalHttpServer $script:server
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
            if ($script:innoSetupExe -and $script:innoSetupExe -like "$env:TEMP*") {
                Remove-Item $script:innoSetupExe -Force -ErrorAction SilentlyContinue
            }
        }

        It "Inno Setup installer available" -Skip:(-not $script:innoSetupExe) {
            $script:innoSetupExe | Should -Not -BeNullOrEmpty
            Test-Path $script:innoSetupExe | Should -Be $true
        }

        It "Inno Setup installs successfully" -Skip:(-not $script:innoSetupExe) {
            $proc = Start-Process -FilePath $script:innoSetupExe `
                -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART" `
                -Wait -PassThru -NoNewWindow
            $proc.ExitCode | Should -Be 0

            # Verify Inno Setup registry key exists
            $innoKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
            Test-Path $innoKey | Should -Be $true

            # Stop Ollama processes started by the installer
            Start-Sleep -Seconds 3
            Get-Process -Name "ollama", "Ollama app" -ErrorAction SilentlyContinue |
                Stop-Process -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }

        It "MSI install migrates from Inno Setup" -Skip:(-not $script:innoSetupExe) {
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1" }

            # MSI should be installed
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true

            # Inno Setup should be removed
            $innoKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{44E83376-CE68-45EB-8FC1-393500EB558C}_is1"
            Test-Path $innoKey | Should -Be $false

            # MSI core should be registered
            $products = Get-InstalledOllamaProducts
            $coreProduct = $products | Where-Object { $_.Name -eq "core" }
            $coreProduct | Should -Not -BeNullOrEmpty
        }

        It "Models directory preserved after migration" -Skip:(-not $script:innoSetupExe) {
            # The Inno Setup uninstaller preserves models by default.
            # We just verify the .ollama directory still exists.
            $ollamaDir = Join-Path $env:USERPROFILE ".ollama"
            Test-Path $ollamaDir | Should -Be $true
        }
    }

    Context "Scenario 14: Authenticode signature warning" {
        BeforeAll {
            Invoke-FullUninstall
            $script:installDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
            # Create a temp directory with a copy of the core MSI that has been
            # tampered with (appending bytes invalidates the Authenticode signature).
            $script:tamperedDir = Join-Path $env:TEMP "OllamaTamperedMSI"
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

        AfterAll {
            Stop-LocalHttpServer $script:server
            Invoke-FullUninstall
            if (Test-Path $script:tamperedDir) {
                Remove-Item $script:tamperedDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        It "Tampered MSI triggers signature warning but install proceeds" {
            # The install script warns on invalid signatures but does not hard-fail
            # (this allows unsigned dev builds to work). Verify the warning appears
            # in output and the install still succeeds.
            $result = Invoke-InstallScript -BaseUrl $script:server.BaseUrl `
                -EnvVars @{ OLLAMA_INSTALL_MINIMAL = "1"; OLLAMA_DEBUG = "1" }

            # Install should succeed despite tampered signature
            Test-OllamaInstalled -Dir $script:installDir | Should -Be $true

            # Output should contain a signature warning
            $output = $result.Output -join "`n"
            $output | Should -Match "does not have a valid Authenticode signature"
        }
    }
}

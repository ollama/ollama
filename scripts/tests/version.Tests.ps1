<#
.SYNOPSIS
    Pester tests for git describe version normalization.

.EXAMPLE
    Import-Module Pester -MinimumVersion 5.0
    Invoke-Pester scripts/tests/version.Tests.ps1 -Tag Unit -Output Detailed
#>

BeforeAll {
    $ScriptRoot = Split-Path -Parent $PSScriptRoot
    . (Join-Path $ScriptRoot "version.ps1")
}

Describe "Git describe version normalization" -Tag Unit {
    It "Normalizes <Input> to <Expected>" -ForEach @(
        @{ Input = "v0.30.10-0-ge1f7f9c"; Expected = "0.30.11-0.0.ge1f7f9c" },
        @{ Input = "v0.30.10-12-gabcdef0"; Expected = "0.30.11-0.12.gabcdef0" },
        @{ Input = "v0.30.10-12-gabcdef0-dirty"; Expected = "0.30.11-0.12.gabcdef0.dirty" },
        @{ Input = "v0.30.11-rc0-0-ge11eeb3"; Expected = "0.30.11-0.rc0.0.ge11eeb3" },
        @{ Input = "v0.30.11-rc0-5-gd9075ca"; Expected = "0.30.11-0.rc0.5.gd9075ca" },
        @{ Input = "v0.30.11-rc1-5-gABCDEF0-dirty"; Expected = "0.30.11-0.rc1.5.gABCDEF0.dirty" },
        @{ Input = "v0.30.11"; Expected = "0.30.11" },
        @{ Input = "d26a585"; Expected = "d26a585" },
        @{ Input = "d26a585-dirty"; Expected = "d26a585-dirty" }
    ) {
        Convert-GitDescribeVersion $Input | Should -Be $Expected
    }
}

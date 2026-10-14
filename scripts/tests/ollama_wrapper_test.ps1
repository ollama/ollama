$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$wrapper = Join-Path $root 'scripts/ollama.ps1'
$temp = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temp | Out-Null

function Assert-Equal([string]$Expected, [string]$Actual, [string]$Name) {
    if ($Expected -cne $Actual) { throw "$Name`: expected [$Expected], got [$Actual]" }
}

try {
    $fakeSource = Join-Path $temp 'fake.go'
    $fakeBinary = Join-Path $temp 'ollama-fake.exe'
    @'
package main
import (
  "fmt"
  "net/http"
  "os"
  "strings"
)
func main() {
  if len(os.Args) > 1 && os.Args[1] == "serve" {
    http.HandleFunc("/api/version", func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, `{"version":"test"}`) })
    http.ListenAndServe(os.Getenv("OLLAMA_TEST_LISTEN_ADDR"), nil)
    return
  }
  if len(os.Args) > 1 && os.Args[1] == "list" {
    fmt.Print(os.Getenv("OLLAMA_TEST_LIST"))
    os.Exit(0)
  }
  if len(os.Args) > 2 && os.Args[1] == "pull" {
    f, _ := os.OpenFile(os.Getenv("OLLAMA_TEST_PULLS"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
    fmt.Fprintln(f, os.Args[2])
    f.Close()
    if os.Args[2] == os.Getenv("OLLAMA_TEST_FAIL_PULL") { os.Exit(1) }
  }
  fmt.Print(strings.Join(os.Args[1:], "\n"))
  if len(os.Args) > 1 && os.Args[1] == "fail" { os.Exit(23) }
}
'@ | Set-Content -Encoding UTF8 $fakeSource
    & go build -o $fakeBinary $fakeSource
    if ($LASTEXITCODE -ne 0) { throw 'failed to build fake Ollama executable' }

    $env:OLLAMA_WRAPPER_BIN = $fakeBinary
    $env:OLLAMA_WRAPPER_STATE_DIR = Join-Path $temp 'state'
    $env:OLLAMA_TEST_PULLS = Join-Path $temp 'pulls.txt'
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = ([Net.IPEndPoint]$listener.LocalEndpoint).Port
    $listener.Stop()
    $env:OLLAMA_HOST = "127.0.0.1:$port"
    $env:OLLAMA_TEST_LISTEN_ADDR = $env:OLLAMA_HOST

    $env:OLLAMA_CONTEXT_LENGTH = '16384'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $wrapper start | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'managed start should succeed' }
    $result = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $wrapper run 'model:tag' 'prompt with spaces'
    Assert-Equal "run`nmodel:tag`nprompt with spaces" (($result -join "`n").Trim()) 'argument forwarding'

    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $wrapper stop 'model:tag' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'stop MODEL must be passed through' }

    $env:OLLAMA_TEST_LIST = "NAME ID SIZE MODIFIED`nalpha/model:latest a 1GB now`nbroken/model:latest b 1GB now`nomega/model:latest c 1GB now"
    $env:OLLAMA_TEST_FAIL_PULL = 'broken/model:latest'
    $env:OLLAMA_TEST_PULLS = Join-Path $temp 'pulls.txt'
    $env:OLLAMA_WRAPPER_TEST_HEALTH = '1'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $wrapper update all | Out-Null
    Assert-Equal '1' "$LASTEXITCODE" 'update all aggregate failure'
    Assert-Equal "alpha/model:latest`nbroken/model:latest`nomega/model:latest" (Get-Content $env:OLLAMA_TEST_PULLS -Raw).Trim() 'update all continuation'

    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $wrapper status | Out-Null
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $wrapper stop | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'managed stop should succeed' }
    if (Test-Path (Join-Path $env:OLLAMA_WRAPPER_STATE_DIR "$($env:OLLAMA_HOST -replace '[:\[\]]', '_')/server.pid")) { throw 'stop should remove PID state' }

    Write-Output 'PASS: Windows wrapper routing and update-all'
} finally {
    Remove-Item Env:\OLLAMA_WRAPPER_BIN -ErrorAction SilentlyContinue
    Remove-Item Env:\OLLAMA_WRAPPER_STATE_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\OLLAMA_TEST_PULLS -ErrorAction SilentlyContinue
    Remove-Item Env:\OLLAMA_TEST_LIST -ErrorAction SilentlyContinue
    Remove-Item Env:\OLLAMA_TEST_FAIL_PULL -ErrorAction SilentlyContinue
    Remove-Item Env:\OLLAMA_WRAPPER_TEST_HEALTH -ErrorAction SilentlyContinue
    Remove-Item Env:\OLLAMA_CONTEXT_LENGTH -ErrorAction SilentlyContinue
    Remove-Item Env:\OLLAMA_HOST -ErrorAction SilentlyContinue
    Remove-Item $temp -Recurse -Force
}

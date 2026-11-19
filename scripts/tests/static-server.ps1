[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Root,

    [Parameter(Mandatory = $true)]
    [int]$Port,

    [string]$ListenAddress = "127.0.0.1",

    [string]$OverlayRoot = ""
)

$ErrorActionPreference = "Stop"
$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path.TrimEnd('\')
$resolvedOverlayRoot = if ($OverlayRoot) {
    (Resolve-Path -LiteralPath $OverlayRoot).Path.TrimEnd('\')
} else {
    ""
}
$address = [System.Net.IPAddress]::Parse($ListenAddress)
$listener = [System.Net.Sockets.TcpListener]::new($address, $Port)

function Write-HttpResponse {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.Stream]$Stream,

        [Parameter(Mandatory = $true)]
        [int]$StatusCode,

        [Parameter(Mandatory = $true)]
        [string]$Reason,

        [long]$ContentLength = 0,
        [string]$ETag = "",
        [string]$Method = "HEAD",
        [string]$FilePath = ""
    )

    $headers = @(
        "HTTP/1.1 $StatusCode $Reason",
        "Content-Length: $ContentLength",
        "Content-Type: application/octet-stream",
        "Connection: close"
    )
    if ($ETag) {
        $headers += "ETag: $ETag"
    }
    $headerBytes = [System.Text.Encoding]::ASCII.GetBytes(($headers -join "`r`n") + "`r`n`r`n")
    $Stream.Write($headerBytes, 0, $headerBytes.Length)

    if ($Method -eq "GET" -and $FilePath) {
        $file = [System.IO.File]::OpenRead($FilePath)
        try {
            $file.CopyTo($Stream)
        } finally {
            $file.Dispose()
        }
    }
    $Stream.Flush()
}

$listener.Start()
Write-Output "Listening on http://${ListenAddress}:$Port from $resolvedRoot"

try {
    while ($true) {
        $client = $listener.AcceptTcpClient()
        try {
            $stream = $client.GetStream()
            $reader = [System.IO.StreamReader]::new(
                $stream,
                [System.Text.Encoding]::ASCII,
                $false,
                4096,
                $true
            )
            try {
                $requestLine = $reader.ReadLine()
                while ($reader.ReadLine()) { }
            } finally {
                $reader.Dispose()
            }

            if ($requestLine -notmatch '^(GET|HEAD)\s+([^\s]+)\s+HTTP/1\.[01]$') {
                Write-HttpResponse -Stream $stream -StatusCode 400 -Reason "Bad Request"
                continue
            }

            $method = $matches[1]
            $requestPath = [System.Uri]::UnescapeDataString(($matches[2] -split '\?', 2)[0])
            if ($requestPath.StartsWith('/download/', [System.StringComparison]::OrdinalIgnoreCase)) {
                $requestPath = $requestPath.Substring('/download/'.Length)
            } else {
                $requestPath = $requestPath.TrimStart('/')
            }

            $relativePath = $requestPath.Replace('/', '\')
            $candidate = ""
            foreach ($candidateRoot in @($resolvedOverlayRoot, $resolvedRoot) | Where-Object { $_ }) {
                $candidatePath = [System.IO.Path]::GetFullPath((Join-Path $candidateRoot $relativePath))
                if (-not $candidatePath.StartsWith($candidateRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
                    Write-HttpResponse -Stream $stream -StatusCode 403 -Reason "Forbidden"
                    $candidate = $null
                    break
                }
                if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
                    $candidate = $candidatePath
                    break
                }
            }
            if ($null -eq $candidate) {
                continue
            }
            if (-not $candidate) {
                Write-HttpResponse -Stream $stream -StatusCode 404 -Reason "Not Found"
                continue
            }

            $item = Get-Item -LiteralPath $candidate
            $etagPath = "$candidate.etag"
            $etag = if (Test-Path -LiteralPath $etagPath -PathType Leaf) {
                (Get-Content -LiteralPath $etagPath -Raw).Trim()
            } else {
                '"{0:x}-{1:x}"' -f $item.Length, $item.LastWriteTimeUtc.Ticks
            }
            $length = $item.Length
            Write-HttpResponse -Stream $stream -StatusCode 200 -Reason "OK" `
                -ContentLength $length -ETag $etag -Method $method -FilePath $candidate
        } catch {
            # A client that disconnects mid-transfer (for example an app under
            # test being stopped while it downloads) must not take the shared
            # server down with it. Write-Error would be terminating here because
            # the script runs with ErrorActionPreference = Stop.
            [Console]::Error.WriteLine("request failed: $($_.Exception.Message)")
        } finally {
            $client.Dispose()
        }
    }
} finally {
    $listener.Stop()
}

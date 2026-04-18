# Builds minimal LGPL FFmpeg for Windows using pre-built libraries
# License: LGPL 2.1+ only (no GPL components)

param(
    [switch]$Ollama
)

$ErrorActionPreference = "Stop"

$FFMPEG_VERSION = "7.1"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$INSTALL_PREFIX = Join-Path $SCRIPT_DIR "install"
$DOWNLOAD_DIR = Join-Path $SCRIPT_DIR "download"

Write-Host "Setting up FFmpeg $FFMPEG_VERSION for Windows..."
Write-Host "Install prefix: $INSTALL_PREFIX"

# Create directories
New-Item -ItemType Directory -Force -Path $DOWNLOAD_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$INSTALL_PREFIX\lib" | Out-Null
New-Item -ItemType Directory -Force -Path "$INSTALL_PREFIX\include" | Out-Null

# Download pre-built FFmpeg dev package (LGPL, shared libs)
$FFMPEG_URL = "https://github.com/GyanD/codexffmpeg/releases/download/$FFMPEG_VERSION/ffmpeg-$FFMPEG_VERSION-full_build-shared.7z"
$FFMPEG_ARCHIVE = Join-Path $DOWNLOAD_DIR "ffmpeg-$FFMPEG_VERSION-shared.7z"

if (-not (Test-Path $FFMPEG_ARCHIVE)) {
    Write-Host "Downloading FFmpeg $FFMPEG_VERSION..."
    Invoke-WebRequest -Uri $FFMPEG_URL -OutFile $FFMPEG_ARCHIVE
}

# Extract using 7z (available on GitHub Actions)
Write-Host "Extracting FFmpeg..."
$EXTRACT_DIR = Join-Path $DOWNLOAD_DIR "ffmpeg-extract"
if (Test-Path $EXTRACT_DIR) {
    Remove-Item -Recurse -Force $EXTRACT_DIR
}
7z x $FFMPEG_ARCHIVE -o"$EXTRACT_DIR" -y | Out-Null

# Find the extracted directory
$FFMPEG_DIR = Get-ChildItem -Path $EXTRACT_DIR -Directory | Select-Object -First 1

# Copy dev files (headers and import libs)
Write-Host "Installing FFmpeg libraries..."
Copy-Item -Recurse -Force "$($FFMPEG_DIR.FullName)\include\*" "$INSTALL_PREFIX\include\"
Copy-Item -Force "$($FFMPEG_DIR.FullName)\lib\*.lib" "$INSTALL_PREFIX\lib\"
Copy-Item -Force "$($FFMPEG_DIR.FullName)\bin\*.dll" "$INSTALL_PREFIX\lib\"

# Create pkg-config files for Windows
$PKG_CONFIG_DIR = Join-Path $INSTALL_PREFIX "lib\pkgconfig"
New-Item -ItemType Directory -Force -Path $PKG_CONFIG_DIR | Out-Null

$PC_FILES = @(
    @{Name="libavcodec"; Libs="-lavcodec"; Requires="libavutil"}
    @{Name="libavformat"; Libs="-lavformat"; Requires="libavcodec libavutil"}
    @{Name="libavutil"; Libs="-lavutil"; Requires=""}
    @{Name="libswscale"; Libs="-lswscale"; Requires="libavutil"}
    @{Name="libavdevice"; Libs="-lavdevice"; Requires="libavformat libavcodec libavutil"}
    @{Name="libavfilter"; Libs="-lavfilter"; Requires="libavutil"}
    @{Name="libswresample"; Libs="-lswresample"; Requires="libavutil"}
)

foreach ($pc in $PC_FILES) {
    $PC_FILE = Join-Path $PKG_CONFIG_DIR "$($pc.Name).pc"
    $REQUIRES_LINE = if ($pc.Requires) { "Requires: $($pc.Requires)" } else { "" }

    @"
prefix=$($INSTALL_PREFIX -replace '\\','/')
exec_prefix=`${prefix}
libdir=`${prefix}/lib
includedir=`${prefix}/include

Name: $($pc.Name)
Description: FFmpeg library
Version: $FFMPEG_VERSION
$REQUIRES_LINE
Libs: -L`${libdir} $($pc.Libs)
Cflags: -I`${includedir}
"@ | Set-Content -Path $PC_FILE
}

Write-Host ""
Write-Host "✅ FFmpeg setup complete!"
Write-Host "Library location: $INSTALL_PREFIX\lib"
Write-Host "Header location: $INSTALL_PREFIX\include"
Write-Host ""
Write-Host "Supported formats:"
Write-Host "  - MP4 (H.264)"
Write-Host "  - WebM (VP9)"
Write-Host "  - MKV (H.265/H.264)"
Write-Host "  - AVI (H.264)"
Write-Host ""
Write-Host "License: LGPL 2.1+ (shared libraries)"

# If -Ollama flag is passed, build Ollama with embedded FFmpeg
if ($Ollama) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Building Ollama with embedded FFmpeg"
    Write-Host "========================================"
    Write-Host ""

    $PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)
    Set-Location $PROJECT_ROOT

    $env:PKG_CONFIG_PATH = $PKG_CONFIG_DIR
    $env:CGO_ENABLED = "1"

    Write-Host "PKG_CONFIG_PATH: $env:PKG_CONFIG_PATH"
    Write-Host ""

    go build -tags ffmpeg,cgo -o ollama.exe .

    Write-Host ""
    Write-Host "✅ Ollama build complete: ollama.exe"
    Write-Host "   - Embedded FFmpeg: YES (dynamically linked)"
    Write-Host "   - Fallback: System ffmpeg"
}

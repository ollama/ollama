#!/bin/bash
# Script to fork and rebrand Ollama to KC-Riff

echo "Starting KC-Riff fork and rebrand process..."

# Step 1: Remove any existing build directories
rm -rf kcriff-build

# Step 2: Create build directory
mkdir -p kcriff-build

# Step 3: Set branding variables
BRAND_NAME="KC-Riff"
BRAND_LOWERCASE="kcriff"
VERSION="0.1.0"
DESCRIPTION="Enhanced Ollama fork with enterprise features"

# Step 4: Copy minimal server
cp minimal_server.go kcriff-build/

# Step 5: Copy CLI client
cp cli_client.py kcriff-build/

# Step 6: Add PyQt interface
mkdir -p kcriff-build/pyqt_interface
cp -r pyqt_interface/* kcriff-build/pyqt_interface/

# Step 7: Create launcher scripts
cp run_desktop_app.py kcriff-build/
cp run_desktop_app_pyqt6.py kcriff-build/

# Step 8: Copy README
cp README.md kcriff-build/

echo "KC-Riff fork and rebrand complete!"
echo "Build files are available in the kcriff-build directory."

#!/bin/sh
# This script uninstalls Ollama.

set -eu

# Validate script is running on Linux
if [ "$(uname -s)" != "Linux" ]; then
    echo 'This script is intended to run on Linux only.'
    exit 1
fi

# Make sure script is run as root
# Else exit with message
if [ "$(id -u)" -ne 0 ]; then
    echo "Please run this script as root"
    exit 1
fi

# Confirmation prompt
echo "Are you sure you want to uninstall Ollama? (y/N): "
read -r REPLY
case $REPLY in
    [Yy]* ) 
        ;;
    * ) 
        echo "Uninstallation cancelled."
        exit 1
        ;;
esac

# Remove stop and remove any systemd services if they exist
if [ -f '/etc/systemd/system/ollama.service' ]; then
    # Stop and disable the ollama service if its running
    systemctl is-enabled --quiet ollama && systemctl disable --now --quiet ollama 

    # Delete the ollama unit file
    echo "Deleting the ollama service file..."
    rm -f /etc/systemd/system/ollama.service

    # Validate system is up to date
    systemctl daemon-reload

    # Inform user ollama is stopped and disabled
    echo "Ollama service stopped and removed."
fi

# Delete the ollama binary if it exists
if command -v ollama > /dev/null; then
    echo "Deleting the ollama binary..."
    rm -f "$(command -v ollama)"
fi

# Remove any locally installed models if they exist
if [ -d '/usr/share/ollama' ]; then
    echo "Deleting locally installed Ollama models..."
    rm -rf /usr/share/ollama/
fi

# Remove the ollama group and user from the system if they exist
if getent passwd ollama > /dev/null; then
    echo "Deleting the ollama user..."
    userdel ollama > /dev/null 2>&1
fi

if getent group ollama > /dev/null; then
    echo "Deleting the ollama group..."
    groupdel ollama > /dev/null 2>&1
fi
# Inform user ollama is uninstall
echo "Ollama has been successfully uninstalled from the system."

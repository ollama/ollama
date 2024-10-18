#!/bin/sh
# This script uninstalls Ollama.

set -eu

error() { echo "ERROR $*"; exit 1; }
available() { command -v "$1" >/dev/null; }

[ "$(uname -s)" = "Linux" ] || error 'This script is intended to run on Linux only.'

SUDO=
if [ "$(id -u)" -ne 0 ]; then
    # Running as root, no need for sudo
    if ! available sudo; then
        error "Please re-run this script as root"
    fi

    SUDO="sudo"
fi

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

# clean up ollama

if [ -f '/etc/systemd/system/ollama.service' ]; then
    echo "Stopping and disabling ollama service at system startup..."
    $SUDO systemctl is-enabled --quiet ollama && $SUDO systemctl disable --now --quiet ollama 

    echo "Deleting the ollama service file..."
    $SUDO rm -f /etc/systemd/system/ollama.service

    $SUDO systemctl daemon-reload
    echo "Ollama service stopped and removed."
fi

if available ollama; then
    echo "Deleting the ollama binary..."
    $SUDO rm -f "$(which ollama)"
fi

if [ -d '/usr/share/ollama' ]; then
    echo "Deleting locally installed Ollama models..."
    $SUDO rm -rf /usr/share/ollama/
fi

if getent passwd ollama > /dev/null; then
    echo "Deleting the ollama user..."
    $SUDO userdel ollama > /dev/null 2>&1
fi

if getent group ollama > /dev/null; then
    echo "Deleting the ollama group..."
    $SUDO groupdel ollama > /dev/null 2>&1
fi

echo "Ollama has been successfully uninstalled from the system."

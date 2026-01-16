#!/bin/sh
# This script uninstalls Ollama on Linux.
set -eu

EUID=$(id -u)
if [ "${EUID}" -ne 0 ]; then
    echo 2>&1 "[ERR] $0 must be run as root."
    exit 1
fi

if [ -f /etc/systemd/system/ollama.service ]; then
    SYSTEMCTL_RUNNING=$(systemctl is-system-running || true)
    case "${SYSTEMCTL_RUNNING}" in
    running | degraded)
        systemctl stop ollama
        systemctl disable ollama
        rm /etc/systemd/system/ollama.service
        systemctl daemon-reload
        ;;
    *)
        rm /etc/systemd/system/ollama.service
        echo 2>&1 "[WRN] systemd is not running."
        ;;
    esac
fi

# https://github.com/ollama/ollama/blob/main/scripts/install.sh#L69
for BINDIR in /usr/local/bin /usr/bin /bin; do
    echo "${PATH}" | grep -q "${BINDIR}" && break
done
OLLAMA_INSTALL_DIR=$(dirname "${BINDIR}")

rm -rf "${OLLAMA_INSTALL_DIR}/bin/ollama"
rm -rf "${OLLAMA_INSTALL_DIR}/lib/ollama"
rm -rf "/usr/share/ollama"

if getent passwd ollama >/dev/null 2>&1; then
    userdel ollama
fi

if getent group ollama >/dev/null 2>&1; then
    groupdel ollama
fi

if [ -n "${SUDO_USER}" ]; then
    user_home=$(getent passwd "${SUDO_USER}" | cut -d: -f6)
    rm -rf "${user_home}/.ollama"
fi

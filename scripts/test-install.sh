#!/usr/bin/env bash
# Exercise the installer with fake hardware and no access to real package managers.
set -euo pipefail

installer=${1:-"$(dirname "$0")/install.sh"}
test_dir=$(mktemp -d)
trap 'rm -rf "$test_dir"' EXIT

mkdir -p "$test_dir/bin" "$test_dir/etc/apt/sources.list.d"
export COMMAND_LOG="$test_dir/commands"
printf '%s\n' 'deb http://example.invalid/debian stable main' >"$test_dir/etc/apt/sources.list"
sed "s|/etc/|$test_dir/etc/|g" "$installer" >"$test_dir/install.sh"

cat >"$test_dir/mock" <<'SH'
#!/bin/sh
command_name=${0##*/}
printf '%s %s\n' "$command_name" "$*" >>"$COMMAND_LOG"

case "$command_name" in
    uname)
        case "$1" in
            -s) echo Linux ;;
            -m) echo x86_64 ;;
            -r) echo 6.8.0-test ;;
        esac ;;
    id) echo 0 ;;
    curl)
        case "$*" in
            *ollama.com/download/*.tar.zst*) exit 22 ;;
            *ollama.com/download/*.tgz*) exit 0 ;;
            *) exit 1 ;;
        esac ;;
    lspci)
        case "$*" in
            *10de:*) printf '%s\n' "$GPUS" | grep NVIDIA ;;
            *1002:*) printf '%s\n' "$GPUS" | grep AMD ;;
        esac ;;
    lshw) printf '%s\n' "$GPUS" ;;
    nvidia-smi) printf '%s\n' 'GPU 0: NVIDIA GeForce RTX 3070' ;;
    install|tar|rm|ln) ;;
    *) exit 1 ;;
esac
SH
chmod +x "$test_dir/mock"

for tool in sh awk cut dirname grep mktemp sed xargs; do
    ln -s "$(command -v "$tool")" "$test_dir/bin/$tool"
done
for tool in uname id curl lspci lshw install tar rm ln apt-get dpkg dkms lsmod modprobe tee; do
    ln -s "$test_dir/mock" "$test_dir/bin/$tool"
done

fail() {
    printf 'FAIL %s\n' "$1" >&2
    cat "$test_dir/output" "$COMMAND_LOG" >&2
    exit 1
}

run_case() {
    local name=$1 distro=$2 version=$3 manager=$4 gpus=$5 expected=$6
    local has_driver=${7:-false}

    printf 'ID=%s\nVERSION_ID=%s\n' "$distro" "$version" >"$test_dir/etc/os-release"
    rm -f "$test_dir/bin/dnf" "$test_dir/bin/yum" "$test_dir/bin/yum-config-manager" "$test_dir/bin/nvidia-smi"
    case "$manager" in
        dnf) ln -s "$test_dir/mock" "$test_dir/bin/dnf" ;;
        yum)
            ln -s "$test_dir/mock" "$test_dir/bin/yum"
            ln -s "$test_dir/mock" "$test_dir/bin/yum-config-manager"
            ;;
    esac
    if [ "$has_driver" = true ]; then
        ln -s "$test_dir/mock" "$test_dir/bin/nvidia-smi"
    fi

    : >"$COMMAND_LOG"
    GPUS="$gpus" PATH="$test_dir/bin" TERM=dumb sh "$test_dir/install.sh" >"$test_dir/output" 2>&1 || fail "$name"
    grep -Fq 'Install complete.' "$test_dir/output" || fail "$name did not complete"
    grep -Fq "$expected" "$test_dir/output" || fail "$name did not report the expected result"

    if grep -Eq '^(apt-get|dpkg|dnf|yum|yum-config-manager|dkms|modprobe|tee) |^curl .*developer\.download\.nvidia\.com' "$COMMAND_LOG"; then
        fail "$name touched the NVIDIA driver installation path"
    fi
    if [ "$expected" = 'AMD GPU ready.' ]; then
        grep -Eq '^curl .*ollama-linux-amd64-rocm\.tgz' "$COMMAND_LOG" || fail "$name did not download ROCm"
    elif grep -q 'ollama-linux-amd64-rocm' "$COMMAND_LOG"; then
        fail "$name unexpectedly tried to download ROCm"
    fi
    printf 'PASS %s\n' "$name"
}

legacy='01:00.0 VGA compatible controller: NVIDIA Corporation GP104 [GeForce GTX 1080]'
modern='02:00.0 3D controller: NVIDIA Corporation GA104 [GeForce RTX 3070]'
unknown='01:00.0 VGA compatible controller: NVIDIA Corporation Device ffff'
amd='01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 21'
guidance='Install a driver that supports all of your NVIDIA GPUs'

run_case 'legacy NVIDIA on Ubuntu' ubuntu 24.04 apt "$legacy" "$guidance"
run_case 'legacy NVIDIA on Debian' debian 12 apt "$legacy" "$guidance"
run_case 'legacy NVIDIA on Fedora' fedora 41 dnf "$legacy" "$guidance"
run_case 'legacy NVIDIA on RHEL' rhel 9.5 dnf "$legacy" "$guidance"
run_case 'legacy NVIDIA on CentOS' centos 7.9 yum "$legacy" "$guidance"
run_case 'legacy NVIDIA on Rocky' rocky 9.5 dnf "$legacy" "$guidance"
run_case 'legacy NVIDIA on Amazon Linux' amzn 2023 dnf "$legacy" "$guidance"
run_case 'modern NVIDIA without verified driver' ubuntu 24.04 apt "$modern" "$guidance"
run_case 'existing NVIDIA driver' ubuntu 24.04 apt "$modern" 'NVIDIA GPU installed.' true
run_case 'multiple NVIDIA generations' ubuntu 24.04 apt "$modern"$'\n'"$legacy" "$guidance"
run_case 'unknown NVIDIA architecture' ubuntu 24.04 apt "$unknown" "$guidance"
run_case 'CPU only' ubuntu 24.04 apt '' 'Ollama will run in CPU-only mode.'
run_case 'AMD' ubuntu 24.04 apt "$amd" 'AMD GPU ready.'

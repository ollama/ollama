#!/bin/sh

# Script for common Dockerfile dependency installation in redhat linux based images

set -ex
MACHINE=$(uname -m)

if grep -i "rocky" /etc/system-release >/dev/null; then
    # Temporary workaround until rocky 8 AppStream ships GCC 10.4 (10.3 is incompatible with NVCC)
    cat << EOF > /etc/yum.repos.d/Rocky-Vault.repo
[vault]
name=Rocky Vault
baseurl=https://dl.rockylinux.org/vault/rocky/8.5/AppStream/\$basearch/os/
gpgcheck=1
enabled=1
countme=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-rockyofficial
EOF
    if [ -z "${SKIP_COMPILER}" ]; then
        dnf install -y git \
            gcc-toolset-10-gcc-10.2.1-8.2.el8 \
            gcc-toolset-10-gcc-c++-10.2.1-8.2.el8
    fi
else
    echo "ERROR Unexpected distro"
    exit 1
fi

if [ -n "${CMAKE_VERSION}" ]; then
    dnf install -y curl
    curl -s -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz | tar -zx -C /usr --strip-components 1
fi

if [ -n "${GOLANG_VERSION}" ]; then
    if [ "${MACHINE}" = "x86_64" ]; then
        GO_ARCH="amd64"
    else
        GO_ARCH="arm64"
    fi
    mkdir -p /usr/local
    curl -s -L https://dl.google.com/go/go${GOLANG_VERSION}.linux-${GO_ARCH}.tar.gz | tar xz -C /usr/local
    ln -s /usr/local/go/bin/go /usr/local/bin/go
    ln -s /usr/local/go/bin/gofmt /usr/local/bin/gofmt
fi

if [ -n "${ROCM_VERSION}" ] && grep -i "rocky" /etc/system-release >/dev/null && [ ! -f /etc/yum.repos.d/rocm.repo ]; then
    cat << EOF > /etc/yum.repos.d/rocm.repo
[ROCm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/rhel8/${ROCM_VERSION}/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF
    dnf install -y curl 'dnf-command(config-manager)'
    yum clean all
    curl -s -L https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm > /tmp/epel-release-latest-8.noarch.rpm
    rpm -ivh /tmp/epel-release-latest-8.noarch.rpm
    crb enable
    dnf install -y rocm
fi
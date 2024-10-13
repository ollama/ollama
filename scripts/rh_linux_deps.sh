#!/bin/sh

# Script for common Dockerfile dependency installation in redhat linux based images

set -ex
set -o pipefail
MACHINE=$(uname -m)

if grep -i "centos" /etc/system-release >/dev/null; then
    # As of 7/1/2024 mirrorlist.centos.org has been taken offline, so adjust accordingly
    sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
    sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
    sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo

    # Centos 7 derivatives have too old of a git version to run our generate script
    # uninstall and ignore failures
    yum remove -y git
    yum -y install epel-release centos-release-scl

    # The release packages reinstate the mirrors, undo that again
    sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
    sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
    sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo

    yum -y install dnf
    if [ "${MACHINE}" = "x86_64" ]; then
        yum -y install https://repo.ius.io/ius-release-el7.rpm
        dnf install -y git236
    else
        dnf install -y rh-git227-git
        ln -s /opt/rh/rh-git227/root/usr/bin/git /usr/local/bin/git
    fi
    dnf install -y devtoolset-10-gcc devtoolset-10-gcc-c++ pigz findutils
elif grep -i "rocky" /etc/system-release >/dev/null; then
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
    dnf install -y git \
        gcc-toolset-10-gcc-10.2.1-8.2.el8 \
        gcc-toolset-10-gcc-c++-10.2.1-8.2.el8 \
        findutils \
        yum-utils \
        pigz
else
    echo "ERROR Unexpected distro"
    exit 1
fi

if [ "${MACHINE}" = "x86_64" ] ; then
    curl -s -L https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz | tar -Jx -C /tmp --strip-components 1 && \
    mv /tmp/ccache /usr/local/bin/
else
    yum -y install epel-release
    yum install -y ccache
fi

if [ -n "${CMAKE_VERSION}" ]; then
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

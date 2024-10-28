#!/bin/sh

# Script for common Dockerfile dependency installation in redhat linux based images

set -ex
#set -o pipefail
MACHINE=$(uname -m)
if [ "${MACHINE}" = "ppc64le" ]; then
  echo "Installing ppc64le dependencees"
  #go version
fi

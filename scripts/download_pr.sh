#!/bin/bash

# Download the artifacts from a PR build labeled with `pr full build`

if [ $# -ne 1 ] ; then
    echo "Usage: ./scripts/download_r.sh <PR NUMBER>"
    exit 1
fi

set -e
PULL=$1

TOKEN=$(gh auth token)
SHA=$(curl -s -G -H "Authorization: ${TOKEN}" https://api.github.com/repos/ollama/ollama/pulls/${PULL}/commits | jq -r ".[0].sha")
RUN_ID=$(curl -s -G -H "Authorization: ${TOKEN}" https://api.github.com/repos/ollama/ollama/actions/workflows/pr_full_build.yaml/runs -d head_sha=${SHA} | jq -r ".workflow_runs.[0].id")
mkdir -p ./dist
echo "Downloading artifacts for PR ${PULL} with commit ${SHA}"
for name in dist-windows dist-darwin dist-linux-amd64 dist-linux-arm64; do
    url=$(curl -s -G -H "Authorization: ${TOKEN}" https://api.github.com/repos/ollama/ollama/actions/runs/${RUN_ID}/artifacts -d name=${name} | jq -r ".artifacts.[0].archive_download_url")
    echo ${name}
    curl --fail --show-error --location --progress-bar -H "Authorization: Bearer ${TOKEN}" ${url} -o ./dist/${name}.zip
    (cd dist && unzip ${name}.zip && rm ${name}.zip)
done

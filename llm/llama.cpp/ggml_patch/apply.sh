#!/bin/bash

# This script is used to apply patches to the GGML source code.
# It is expected to be run from the root of the llama.cpp folder.
# This script is needed because the patches must be applied to the git submodule.
cd ggml

# Loop through all the .diff files in the ggml_patch directory
for diff_file in ../ggml_patch/*.diff; do
  if git apply --check "$diff_file" 2>/dev/null; then
    git apply "$diff_file"
  else
    echo "The patch $diff_file cannot be applied or has already been applied."
  fi
done

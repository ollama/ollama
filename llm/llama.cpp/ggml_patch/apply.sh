#!/bin/bash

# This script is used to apply patches to the GGML source code.
# It is expected to be run from the root of the llama.cpp folder.
# This script is needed because the patches must be applied to the git submodule.
cd ggml

# Loop through all the .patch files in the ggml_patch directory
for patch in ../ggml_patch/*.patch; do
  if git apply --check "$patch" 2>/dev/null; then
    git apply "$patch"
  else
    echo "The patch $patch cannot be applied or has already been applied."
  fi
done

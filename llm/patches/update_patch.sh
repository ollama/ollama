#!/usr/bin/env bash

# Helper script to update a patch file for the llama.cpp submodule. When bumping
# the submodule, the patches become obsolete due to their base commits being
# wrong. This script will attempt to use cherry-pick to replay a patch on the
# currently committed llama.cpp pointer.

cd $(dirname "${BASH_SOURCE[0]}")/../..

patch_file=$1
patch_file=$(cd $(dirname $patch_file) && pwd)/$(basename $patch_file)

# Kill any previously failed patch application
rm -rf .git/modules/llama.cpp/rebase-apply 2>/dev/null || true

# Stash any outstanding changes
current_branch=$(git rev-parse --abbrev-ref HEAD)
res=$(git stash)
if [ "$res" == "No local changes to save" ]
then
    stashed=0
else
    stashed=1
fi
echo "-------------------------"
echo "Current Branch: $current_branch"
echo "Stashed: $stashed"

# Check out main from the mainline repo and update llama.cpp
echo "-------------------------"
mainline_remote=$(git remote -v | grep ollama/ollama | cut -f 1 | uniq)
git fetch $mainline_remote
git checkout $mainline_remote/main
git submodule update --force

# Apply the patch in question
echo "-------------------------"
cd llm/llama.cpp
git -c 'user.name=nobody' -c 'user.email=<>' am $patch_file
patch_commit=$(git rev-parse HEAD)
echo "PatchCommit: $patch_commit"

# Check out the current branch and update back to the right point in llama.cpp
echo "-------------------------"
cd -
git checkout $current_branch
if [ "$stashed" == "1" ]
then
    git stash pop
fi
git submodule update --force

# Cherry-pick the patch commit from the tmp branch
echo "-------------------------"
cd llm/llama.cpp
git cherry-pick $patch_commit
if [ "$?" != "0" ]
then
    echo "CONFLICT APPLYING PATCH!"
    exit 1
fi

# Create the new patch
echo "-------------------------"
git format-patch -1 HEAD --output tmp.patch
mv tmp.patch $patch_file
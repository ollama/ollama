#!/bin/bash

# Wrapper script to build all our platforms and store the results in ./dist
if [ "$(uname)" != "Darwin" ]; then
    echo "ERROR: This script is only intended to run from MacOS at present"
    exit 1
fi
if [ $# -ne 1 ]; then
    echo "ERROR: you must specify a windows git remote name as an argument"
    exit 1
fi

# Make sure the tree isn't dirty, as the remote build script needs all changes committed
if [[ $(git diff --stat) != '' ]]; then
    echo "ERROR: Your tree is not clean.  Please git add and commit all local changes first"
    exit 1
fi


REMOTE_NAME=$1
DARWIN_LOG=./dist/darwin-build.log
LINUX_LOG=./dist/linux-build.log
WINDOWS_LOG=./dist/windows-build.log
ret="0"
mkdir -p ./dist

./scripts/build_linux.sh &> ${LINUX_LOG} &
LINUX_PID=$!
echo "Linux build started.   Watch log with 'tail -F ${LINUX_LOG}'"

./scripts/build_remote.py ${REMOTE_NAME} &> ${WINDOWS_LOG} &
WINDOWS_PID=$!
echo "Windows build started. Watch log with 'tail -F ${WINDOWS_LOG}'"

./scripts/build_darwin.sh &> ${DARWIN_LOG} &
DARWIN_PID=$!
echo "Darwin build started.  Watch log with 'tail -F ${DARWIN_LOG}'"

echo "Waiting for builds to finish..."
echo ""

wait ${DARWIN_PID}
if [ $? -ne 0 ]; then
    echo "Darwin build failed $? - run 'tail ${DARWIN_LOG}' for details"
    ret="1"
else
    echo "Darwin build completed."
fi
wait ${WINDOWS_PID}
if [ $? -ne 0 ]; then
    echo "Windows build failed $? - run 'tail ${WINDOWS_LOG}' for details"
    ret="1"
else
    echo "Windows build completed."
fi
wait ${LINUX_PID}
if [ $? -ne 0 ]; then
    echo "Linux build failed $? - run 'tail ${LINUX_LOG}' for details"
    ret="1"
else
    echo "Linux build completed."
fi
exit "${ret}"
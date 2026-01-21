
set -e
# Apply patches if the patches have not been applied and the firt arugment is patch otherwise just build
if [ "$1" == "patch" ]
then
    make -f Makefile.sync checkout
    cd llama/vendor
    git apply ../patches/tsi-consolidated-patches.patch
    cd ../../
    make -f Makefile.sync ml/backend/ggml/ggml
fi

cd llama/vendor
#Ensure prerequisites are met as follows
echo 'updating submodule'
git submodule update --recursive --init
cd ggml-tsi-kernel/
module load gcc/13.3.0
export MLIR_SDK_VERSION=/proj/rel/sw/sdk-r.0.2.3
echo 'creating python virtual env'
/proj/local/Python-3.10.12/bin/python3 -m venv blob-creation
source blob-creation/bin/activate
echo 'installing mlir and python dependencies'
pip install --upgrade pip
pip install -r ${MLIR_SDK_VERSION}/compiler/python/requirements-common.txt
pip install ${MLIR_SDK_VERSION}/compiler/python/mlir_external_packages-1.4.2-py3-none-any.whl
pip install onnxruntime-training

#build TSI kernels for the Tsavorite backend
#First for FPGA

#echo 'creating fpga kernel'
cd fpga-kernel
cmake -B build-fpga
./create-all-kernels.sh
#The for Posix Use cases

echo 'creating posix kernel'
cd ../posix-kernel/
./create-all-kernels.sh

cd ../..
echo "$(pwd)"

#Change directory to top level ollama

cd ../../

#Compile for posix & fpga with build-posix as a target folder

echo 'building llama.cp, ggml for tsavorite  and other binary for posix'
if [ "$(echo "$1" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DCMAKE_C_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_POSIX -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_POSIX -DGGML_TSAVORITE"
elif [ "$(echo "$1" | tr '[:upper:]' '[:lower:]')" = "debug" ]; then
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DCMAKE_C_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_POSIX -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_POSIX -DGGML_TSAVORITE"
else
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DCMAKE_C_FLAGS="-DGGML_PERF -DGGML_TARGET_POSIX -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF -DGGML_TARGET_POSIX -DGGML_TSAVORITE"
fi

cmake --build build-posix --config Release

# Fix GLIBC compatibility for TSI binaries
#echo 'fixing GLIBC compatibility for TSI binaries'

# Fix simple-backend-tsi
#mkdir -p build-posix/bin/
#mv llama/vendor/build-posix/bin/simple-backend-tsi build-posix/bin/simple-backend-tsi-original
#cat > build-posix/bin/simple-backend-tsi << 'EOL'
#!/bin/bash
#export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
#exec "$(dirname "$0")/simple-backend-tsi-original" "$@"
#EOL
#chmod +x build-posix/bin/simple-backend-tsi

# Fix llama-cli
#mkdir -p build-posix/bin/
#mv llama/vendor/build-posix/bin/llama-cli build-posix/bin/llama-cli-original
#cat > build-posix/bin/llama-cli << 'EOL'
#!/bin/bash
#export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
#exec "$(dirname "$0")/llama-cli-original" "$@"
#EOL
#chmod +x build-posix/bin/llama-cli

#Compile for fpga with build-fpga as a target folder

echo 'building llama.cp, ggml for tsavorite  and other binary for fpga'
export CC="/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc"
export CXX="/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++"

if [ "$(echo "$1" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
 cmake -B build-fpga -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DCMAKE_C_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCURL_INCLUDE_DIR=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/include  -DCURL_LIBRARY=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/lib/libcurl.so
elif [ "$(echo "$1" | tr '[:upper:]' '[:lower:]')" = "debug" ]; then
  cmake -B build-fpga -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DCMAKE_C_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCURL_INCLUDE_DIR=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/include  -DCURL_LIBRARY=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/lib/libcurl.so
else
  cmake -B build-fpga -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DCMAKE_C_FLAGS="-DGGML_PERF -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCURL_INCLUDE_DIR=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/include  -DCURL_LIBRARY=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/lib/libcurl.so
fi

cmake --build build-fpga --config Release


#echo 'creating tar bundle for fpga'
TSI_GGML_VERSION=0.2.0
TSI_GGML_BUNDLE_INSTALL_DIR=tsi-ggml
GGML_TSI_INSTALL_DIR=llama/vendor/ggml-tsi-kernel
TSI_GGML_RELEASE_DIR=/proj/rel/sw/ggml
TSI_BLOB_INSTALL_DIR=$(pwd)/${GGML_TSI_INSTALL_DIR}/fpga-kernel/build-fpga

if [ -e ${TSI_GGML_BUNDLE_INSTALL_DIR} ]; then
   echo "${TSI_GGML_BUNDLE_INSTALL_DIR} exist"
else
   echo "creating ${TSI_GGML_BUNDLE_INSTALL_DIR}"
   mkdir ${TSI_GGML_BUNDLE_INSTALL_DIR}
fi
if [ -e ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh ]; then
   rm -fr ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh
fi

cat > ./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh << EOL
#!/bin/bash
# Set up library paths for GCC 13.3.0 compatibility
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:\$(pwd)

tsi_kernels=("add" "sub" "mult" "div" "abs" "inv" "neg" "sin" "sqrt" "sqr" "sigmoid" "silu" "rms_norm"  "swiglu" "add_16" "sub_16" "mult_16" "div_16" "abs_16" "inv_16" "neg_16" "sin_16" "sqrt_16" "sqr_16" "sigmoid_16" "silu_16" "rms_norm_16" "swiglu_16")

for kernel in "\${tsi_kernels[@]}"; do
    mkdir -p ${TSI_BLOB_INSTALL_DIR}/txe_\$kernel
    cp --parent blobs/txe_\$kernel*.blob ${TSI_BLOB_INSTALL_DIR}/txe_\$kernel/ -r
done
EOL
chmod +x ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh
cp ${GGML_TSI_INSTALL_DIR}/fpga/blobs ${TSI_GGML_BUNDLE_INSTALL_DIR}/ -r

if [ "$(echo "$1" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
    cp ${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz ${TSI_GGML_RELEASE_DIR}/

    LATEST_TZ="${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz"
    LATEST_FULL_PATH="${TSI_GGML_RELEASE_DIR}/$(basename "$LATEST_TZ")"

    # Remove old symlinks if they exist
    rm -f "$TSI_GGML_RELEASE_DIR/tsi-ggml-aws-latest.tz"
    rm -f "$TSI_GGML_RELEASE_DIR/tsi-ggml-latest.tz"
    # Create new symbolic links
    ln -s /aws"$LATEST_FULL_PATH" "$TSI_GGML_RELEASE_DIR/tsi-ggml-aws-latest.tz"
    ln -s "$LATEST_FULL_PATH" "$TSI_GGML_RELEASE_DIR/tsi-ggml-latest.tz"

    echo "Symlinks updated to point to $(basename "$LATEST_FULL_PATH")"
fi

RELEASE_DIR="ollama-arm64-release"
TARBALL="ollama-arm64-release.tar.gz"

# Build Go binary for ARM64
echo "Building Go binary for ARM64..."
export CGO_ENABLED=1
export PATH=$PATH:/proj/local/go/bin
GOARCH=arm64 GOOS=linux go build -o ollama .

# Prepare release directory
echo "Preparing release directory..."
rm -rf $RELEASE_DIR
mkdir -p $RELEASE_DIR/bin
mkdir -p $RELEASE_DIR/lib
cp ollama $RELEASE_DIR/bin/
cp llama/vendor/ggml-tsi-kernel/fpga/blobs ${RELEASE_DIR}/ -r
cp build-fpga/lib/ollama/libggml-*.so ${RELEASE_DIR}/bin
cp build-fpga/lib/ollama/libggml-*.so ${RELEASE_DIR}/lib

cp -r lib $RELEASE_DIR/ 2>/dev/null || echo "No lib directory to copy"
cp README.md $RELEASE_DIR/ 2>/dev/null || echo "No README.md to copy"
cp -r tsi-ggml $RELEASE_DIR/ 2>/dev/null || echo "No tsi-ggml-ollama*.tz to copy"

# Create tarball
echo "Creating tarball..."
tar -czvf $TARBALL $RELEASE_DIR

echo "ARM64 tarbundle created: $TARBALL"

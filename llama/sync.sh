#!/bin/bash

# Set the source directory
src_dir=$1

# Set the destination directory (current directory)
dst_dir="."

# llama.cpp
cp $src_dir/unicode.cpp $dst_dir/unicode.cpp
cp $src_dir/unicode.h $dst_dir/unicode.h
cp $src_dir/unicode-data.cpp $dst_dir/unicode-data.cpp
cp $src_dir/unicode-data.h $dst_dir/unicode-data.h
cp $src_dir/llama.cpp $dst_dir/llama.cpp
cp $src_dir/llama.h $dst_dir/llama.h
cp $src_dir/sgemm.cpp $dst_dir/sgemm.cpp
cp $src_dir/sgemm.h $dst_dir/sgemm.h

# ggml
cp $src_dir/ggml.c $dst_dir/ggml.c
cp $src_dir/ggml.h $dst_dir/ggml.h
cp $src_dir/ggml-quants.c $dst_dir/ggml-quants.c
cp $src_dir/ggml-quants.h $dst_dir/ggml-quants.h
cp $src_dir/ggml-metal.metal $dst_dir/ggml-metal.metal
cp $src_dir/ggml-metal.h $dst_dir/ggml-metal.h
cp $src_dir/ggml-metal.m $dst_dir/ggml-metal.m
cp $src_dir/ggml-impl.h $dst_dir/ggml-impl.h
cp $src_dir/ggml-cuda.h $dst_dir/ggml-cuda.h
cp $src_dir/ggml-cuda.cu $dst_dir/ggml-cuda.cu
cp $src_dir/ggml-common.h $dst_dir/ggml-common.h
cp $src_dir/ggml-backend.h $dst_dir/ggml-backend.h
cp $src_dir/ggml-backend.c $dst_dir/ggml-backend.c
cp $src_dir/ggml-backend-impl.h $dst_dir/ggml-backend-impl.h
cp $src_dir/ggml-alloc.h $dst_dir/ggml-alloc.h
cp $src_dir/ggml-alloc.c $dst_dir/ggml-alloc.c

# ggml-cuda
mkdir -p $dst_dir/ggml-cuda
cp $src_dir/ggml-cuda/*.cu $dst_dir/ggml-cuda/
cp $src_dir/ggml-cuda/*.cuh $dst_dir/ggml-cuda/

# ggml-metal
sed -i '' '1s;^;//go:build darwin,arm64\n;' ggml-metal.m
sed -e '/#include "ggml-common.h"/r ggml-common.h' -e '/#include "ggml-common.h"/d' < ggml-metal.metal > temp.metal
TEMP_ASSEMBLY=$(mktemp)
echo ".section __DATA, __ggml_metallib"   >  $TEMP_ASSEMBLY
echo ".globl _ggml_metallib_start"        >> $TEMP_ASSEMBLY
echo "_ggml_metallib_start:"              >> $TEMP_ASSEMBLY
echo ".incbin \"temp.metal\"" >> $TEMP_ASSEMBLY
echo ".globl _ggml_metallib_end"          >> $TEMP_ASSEMBLY
echo "_ggml_metallib_end:"                >> $TEMP_ASSEMBLY
as -mmacosx-version-min=11.3 $TEMP_ASSEMBLY -o ggml-metal.o
rm -f $TEMP_ASSEMBLY
rm -rf temp.metal

# apply patches

# add license
# sha1=$(git -C $src_dir rev-parse @)

# tempdir=$(mktemp)
# cleanup() {
#     rm -f $tempdir
# }
# trap cleanup 0

# cat <<EOF | sed 's/ *$//' >$tempdir
# /**
#  * llama.cpp - git $sha1
#  *
# $(sed 's/^/ * /' <$src_dir/LICENSE)
#  */

# EOF

# for IN in $dst_dir/*.{c,h,cpp,m,metal,cu}; do
#     if [[ "$IN" == *"sgemm.cpp" || "$IN" == *"sgemm.h" ]]; then
#         continue
#     fi
#     TMP=$(mktemp)
#     cat $tempdir $IN >$TMP
#     mv $TMP $IN
# done

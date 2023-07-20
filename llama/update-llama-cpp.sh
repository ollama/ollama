#!/bin/sh

set -eu


status() { echo >&2 ">>> $*"; }
error() { status "ERROR $*"; }
usage() {
    echo "usage: $(basename $0) /path/to/repo"
    exit 1
}

OUT=$(dirname $0)
while getopts "hC:" OPTION; do
    case $OPTION in
        C) OUT=$OPTARG ;;
        *) usage ;;
    esac
done

shift $(( $OPTIND - 1 ))
[ $# -eq 1 ] || usage

status "updating source..."
cp -a "$1"/*.{c,h,cpp,m,metal,cu} "$OUT"

status "removing incompatible files..."
rm -f "$OUT"/build-info.h

SHA1=$(git -C $1 rev-parse @)

LICENSE=$(mktemp)
cleanup() {
    rm -f $LICENSE
}
trap cleanup 0

cat <<EOF | sed 's/ *$//' >$LICENSE
/**
 * llama.cpp - git $SHA1
 *
$(sed 's/^/ * /' <$1/LICENSE)
 */

EOF

for IN in $OUT/*.{c,h,cpp,m,metal,cu}; do
    TMP=$(mktemp)
    status "updating license $IN"
    cat $LICENSE $IN >$TMP
    mv $TMP $IN
done

touchup() {
    local CONSTRAINT=$1 && shift

    for IN in $*; do
        status "touching up $IN..."
        TMP=$(mktemp)
        {
            echo "//go:build $CONSTRAINT"
            echo
        } | cat - $IN >$TMP
        mv $TMP $IN
    done
}

touchup darwin $OUT/ggml-metal.*
touchup mpi $OUT/ggml-mpi.*
touchup opencl $OUT/ggml-opencl.*

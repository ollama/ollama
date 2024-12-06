package metal

//go:generate sh -c "echo \"// Code generated $(date). DO NOT EDIT.\n\" >ggml-metal-embed.metal"
//go:generate sh -c "sed -e '/__embed_ggml-common.h__/r ../ggml-common.h' -e '/#include \"ggml-metal-impl.h\"/r ggml-metal-impl.h' -e '/__embed_ggml-common.h__/d' -e '/#include \"ggml-metal-impl.h\"/d' ggml-metal.metal >>ggml-metal-embed.metal"

// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../include
// #cgo CPPFLAGS: -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_BLAS -DGGML_METAL_NDEBUG
// #cgo LDFLAGS: -framework Metal -framework MetalKit -framework Accelerate
import "C"

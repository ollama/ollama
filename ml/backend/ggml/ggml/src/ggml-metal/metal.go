//go:build darwin && arm64

package metal

//go:generate sh -c "{ echo // Code generated $(date). DO NOT EDIT.; sed -e '/__embed_ggml-common.h__/r ../ggml-common.h' -e '/__embed_ggml-common.h__/d' -e '/#include \"ggml-metal-impl.h\"/r ggml-metal-impl.h' -e '/#include \"ggml-metal-impl.h\"/d' ggml-metal.metal; } >ggml-metal-embed.metal"

// #cgo CPPFLAGS: -DGGML_METAL_EMBED_LIBRARY -I.. -I../../include
// #cgo LDFLAGS: -framework Metal -framework MetalKit
import "C"

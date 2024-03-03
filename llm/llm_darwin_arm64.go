//go:generate cmake -S server -B server/build/metal -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=arm64 -DCMAKE_OSX_ARCHITECTURES=arm64
//go:generate cmake --build server/build/metal --target server -- -j4
package llm

import "embed"

//go:embed server/build/metal/ggml-metal.metal server/build/metal/server
var libEmbed embed.FS

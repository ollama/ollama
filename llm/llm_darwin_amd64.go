//go:generate cmake -S server -B server/build/cpu -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=x86_64 -DCMAKE_OSX_ARCHITECTURES=x86_64 -DLLAMA_METAL=off -DLLAMA_NATIVE=off
//go:generate cmake -S server -B server/build/cpu_avx -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=x86_64 -DCMAKE_OSX_ARCHITECTURES=x86_64 -DLLAMA_METAL=off -DLLAMA_NATIVE=off -DLLAMA_AVX=on
//go:generate cmake -S server -B server/build/cpu_avx2 -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=x86_64 -DCMAKE_OSX_ARCHITECTURES=x86_64 -DLLAMA_METAL=off -DLLAMA_NATIVE=off -DLLAMA_AVX=on -DLLAMA_AVX2=on
//go:generate cmake --build server/build/cpu --target server -- -j4
//go:generate cmake --build server/build/cpu_avx --target server -- -j4
//go:generate cmake --build server/build/cpu_avx2 --target server -- -j4
package llm

import "embed"

//go:embed server/build/cpu/server
//go:embed server/build/cpu_avx/server
//go:embed server/build/cpu_avx2/server
var libEmbed embed.FS

/**
 * This is a custom source file that explicitly incorporates the cpu-arch
 * implementation files for the target CPU. It is not part of the GGML source
 * and is only used to bind the arch implementations to CGO.
 *
 * The preprocessor defines are specified in cpu.go and correspond to GOARCH
 * values.
 *
 * https://github.com/golang/go/blob/master/src/internal/syslist/syslist.go#L58
 */

#ifdef __amd64__
#include "./arch/x86/quants.c"

#elif defined __arm64__
#include "./arch/arm/quants.c"

#elif defined __loong64__
#include "./arch/loongarch/quants.c"

#elif defined __ppc64__
#include "./arch/powerpc/quants.c"

#elif defined __riscv64__
#include "./arch/riscv/quants.c"

#elif defined __s390x__
#include "./arch/s390/quants.c"

#elif defined __wasm__
#include "./arch/wasm/quants.c"

#endif

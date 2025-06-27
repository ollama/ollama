#include "ggml-backend-impl.h"

#if defined(__aarch64__)

#if defined(__linux__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if !defined(HWCAP2_I8MM)
#define HWCAP2_I8MM (1 << 13)
#endif

#if !defined(HWCAP2_SME)
#define HWCAP2_SME (1 << 23)
#endif

struct aarch64_features {
    // has_neon not needed, aarch64 has NEON guaranteed
    bool has_dotprod     = false;
    bool has_fp16_va     = false;
    bool has_sve         = false;
    bool has_sve2        = false;
    bool has_i8mm        = false;
    bool has_sme         = false;

    aarch64_features() {
#if defined(__linux__)
        uint32_t hwcap = getauxval(AT_HWCAP);
        uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
        has_fp16_va = !!(hwcap & HWCAP_FPHP);
        has_sve     = !!(hwcap & HWCAP_SVE);
        has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
        has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
        has_sme     = !!(hwcap2 & HWCAP2_SME);
#elif defined(__APPLE__)
        int oldp = 0;
        size_t size = sizeof(oldp);

        if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &size, NULL, 0) == 0) {
            has_dotprod = static_cast<bool>(oldp);
        }

        if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &size, NULL, 0) == 0) {
            has_i8mm = static_cast<bool>(oldp);
        }

        if (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &size, NULL, 0) == 0) {
            has_sme = static_cast<bool>(oldp);
        }

        // Apple apparently does not implement SVE yet
#endif
    }
};

static int ggml_backend_cpu_aarch64_score() {
    int score = 1;
    aarch64_features af;

#ifdef GGML_USE_DOTPROD
    if (!af.has_dotprod) { return 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
    if (!af.has_fp16_va) { return 0; }
    score += 1<<2;
#endif
#ifdef GGML_USE_SVE
    if (!af.has_sve) { return 0; }
    score += 1<<3;
#endif
#ifdef GGML_USE_MATMUL_INT8
    if (!af.has_i8mm) { return 0; }
    score += 1<<4;
#endif
#ifdef GGML_USE_SVE2
    if (!af.has_sve2) { return 0; }
    score += 1<<5;
#endif
#ifdef GGML_USE_SME
    if (!af.has_sme) { return 0; }
    score += 1<<6;
#endif

    return score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__)

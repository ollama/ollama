#include "ggml-backend-impl.h"

#if defined(__s390x__)

#if defined(__linux__)
#include <sys/auxv.h>
#endif

struct s390x_features {
    bool has_vx  = false;
    bool has_vxe = false;
    bool has_vxe2 = false;
    bool has_nnpa = false;

    s390x_features() {
#if defined(__linux__)
        // Check for vector facility via hwcap
        // HWCAP_S390_VXRS = 2048 (bit 11)
        // HWCAP_S390_VXRS_EXT = 4096 (bit 12)
        // HWCAP_S390_VXRS_EXT2 = 8192 (bit 13)
        // HWCAP_S390_NNPA = 262144 (bit 18)
        unsigned long hwcap = getauxval(AT_HWCAP);
        unsigned long hwcap2 = getauxval(AT_HWCAP2);
        has_vx  = (hwcap & 2048) != 0;
        has_vxe = (hwcap & 4096) != 0;
        has_vxe2 = (hwcap & 8192) != 0;
        has_nnpa = (hwcap2 & 262144) != 0;
#endif
    }
};

static int ggml_backend_cpu_s390_score() {
    int score = 1;
    s390x_features features;

#ifdef GGML_USE_VXE2
    if (!features.has_vxe2) { return 0; }
    score += 1;
#endif
#ifdef GGML_USE_VXE
    if (!features.has_vxe) { return 0; }
    score += 1<<1;
#endif
#ifdef GGML_USE_NNPA
    if (!features.has_nnpa) { return 0; }
    score += 1<<2;
#endif

    return score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390_score)

#endif // defined(__s390x__)

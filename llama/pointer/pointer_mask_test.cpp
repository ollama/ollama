#include "pointer_mask_logic.h"

#include <cstdio>
#include <cstdint>

static int fail = 0;

static void expect(const char * name, const int32_t * seg, int n, bool state_bidir, const int * want) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const bool got = llama_pointer_allow(seg, n, i, j, state_bidir);
            const bool exp = want[i * n + j] != 0;
            if (got != exp) {
                std::fprintf(stderr, "%s: mask[%d][%d] = %d, want %d\n", name, i, j, got, exp);
                fail = 1;
            }
        }
    }
}

int main() {
    const int32_t bidir[] = {0, 0, 0, 1, 1, 2, 2};
    const int bidir_want[] = {
        1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0,
        1, 1, 1, 0, 0, 1, 0,
        1, 1, 1, 0, 0, 1, 1,
    };
    expect("bidir", bidir, 7, true, bidir_want);

    const int32_t causal[] = {0, 0, 0, 1, 1};
    const int causal_want[] = {
        1, 0, 0, 0, 0,
        1, 1, 0, 0, 0,
        1, 1, 1, 0, 0,
        1, 1, 1, 1, 0,
        1, 1, 1, 1, 1,
    };
    expect("causal_state", causal, 5, false, causal_want);

    const int32_t pad[] = {0, 0, -1};
    const int pad_want[] = {
        1, 1, 0,
        1, 1, 0,
        1, 1, 1,
    };
    expect("pad", pad, 3, true, pad_want);

    if (fail) {
        return 1;
    }
    std::puts("pointer mask ok");
    return 0;
}

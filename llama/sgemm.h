#pragma once
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(int, int, int, const void *, int, const void *, int,
                     void *, int, int, int, int, int, int, int);

#ifdef __cplusplus
}
#endif

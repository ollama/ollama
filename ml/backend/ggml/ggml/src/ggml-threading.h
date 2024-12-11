#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void ggml_critical_section_start(void);
void ggml_critical_section_end(void);

#ifdef __cplusplus
}
#endif

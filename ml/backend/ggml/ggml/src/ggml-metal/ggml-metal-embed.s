.section __DATA, __ggml_metallib
.globl _ggml_metallib_start
_ggml_metallib_start:
.incbin "ggml-metal-embed.metal"
.globl _ggml_metallib_end
_ggml_metallib_end:

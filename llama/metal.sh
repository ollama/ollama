sed -e '/#include "ggml-common.h"/r ggml-common.h' -e '/#include "ggml-common.h"/d' < ggml-metal.metal > ggml-metal-embed.metal
TEMP_ASSEMBLY=$(mktemp)
echo ".section __DATA, __ggml_metallib"   >  $TEMP_ASSEMBLY
echo ".globl _ggml_metallib_start"        >> $TEMP_ASSEMBLY
echo "_ggml_metallib_start:"              >> $TEMP_ASSEMBLY
echo ".incbin \"ggml-metal-embed.metal\"" >> $TEMP_ASSEMBLY
echo ".globl _ggml_metallib_end"          >> $TEMP_ASSEMBLY
echo "_ggml_metallib_end:"                >> $TEMP_ASSEMBLY
as -mmacosx-version-min=11.3 $TEMP_ASSEMBLY -o ggml-metal.o
rm -f $TEMP_ASSEMBLY
rm -rf ggml-metal-embed.metal

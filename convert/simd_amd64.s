#include "textflag.h"

// func f16ToF32AVX(dst *float32, src *uint16, chunks int)
TEXT ·f16ToF32AVX(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ chunks+16(FP), CX
	TESTQ CX, CX
	JZ done_f16_to_f32

loop_f16_to_f32:
	VMOVDQU (SI), X0
	VCVTPH2PS X0, Y0
	VMOVUPS Y0, (DI)
	ADDQ $16, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_f16_to_f32

done_f16_to_f32:
	VZEROUPPER
	RET

// func f32ToF16AVX(dst *uint16, src *float32, chunks int)
TEXT ·f32ToF16AVX(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ chunks+16(FP), CX
	TESTQ CX, CX
	JZ done_f32_to_f16

loop_f32_to_f16:
	VMOVUPS (SI), Y0
	VCVTPS2PH $4, Y0, X0
	VMOVDQU X0, (DI)
	ADDQ $32, SI
	ADDQ $16, DI
	DECQ CX
	JNZ loop_f32_to_f16

done_f32_to_f16:
	VZEROUPPER
	RET

// func bf16ToF32AVX(dst *float32, src *uint16, chunks int)
TEXT ·bf16ToF32AVX(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ chunks+16(FP), CX
	TESTQ CX, CX
	JZ done_bf16_to_f32

loop_bf16_to_f32:
	VPMOVZXWD (SI), Y0
	VPSLLD $16, Y0, Y0
	VMOVUPS Y0, (DI)
	ADDQ $16, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_bf16_to_f32

done_bf16_to_f32:
	VZEROUPPER
	RET

// func f32ToBF16AVX(dst *uint16, src *float32, chunks int)
TEXT ·f32ToBF16AVX(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ chunks+16(FP), CX
	TESTQ CX, CX
	JZ done_f32_to_bf16

loop_f32_to_bf16:
	VMOVDQU (SI), X0
	VPSRLD $16, X0, X0
	VMOVDQU 16(SI), X1
	VPSRLD $16, X1, X1
	VPACKUSDW X1, X0, X0
	VMOVDQU X0, (DI)
	ADDQ $32, SI
	ADDQ $16, DI
	DECQ CX
	JNZ loop_f32_to_bf16

done_f32_to_bf16:
	VZEROUPPER
	RET

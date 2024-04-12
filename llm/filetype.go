package llm

import "fmt"

type filetype uint32

const (
	filetypeF32 filetype = iota
	filetypeF16
	filetypeQ4_0
	filetypeQ4_1
	filetypeQ4_1_F16
	filetypeQ8_0 filetype = iota + 2
	filetypeQ5_0
	filetypeQ5_1
	filetypeQ2_K
	filetypeQ3_K_S
	filetypeQ3_K_M
	filetypeQ3_K_L
	filetypeQ4_K_S
	filetypeQ4_K_M
	filetypeQ5_K_S
	filetypeQ5_K_M
	filetypeQ6_K
	filetypeIQ2_XXS
	filetypeIQ2_XS
	filetypeQ2_K_S
	filetypeQ3_K_XS
	filetypeIQ3_XXS

	filetypeUnknown
)

func ParseFileType(s string) (filetype, error) {
	switch s {
	case "F32":
		return filetypeF32, nil
	case "F16":
		return filetypeF16, nil
	case "Q4_0":
		return filetypeQ4_0, nil
	case "Q4_1":
		return filetypeQ4_1, nil
	case "Q4_1_F16":
		return filetypeQ4_1_F16, nil
	case "Q8_0":
		return filetypeQ8_0, nil
	case "Q5_0":
		return filetypeQ5_0, nil
	case "Q5_1":
		return filetypeQ5_1, nil
	case "Q2_K":
		return filetypeQ2_K, nil
	case "Q3_K_S":
		return filetypeQ3_K_S, nil
	case "Q3_K_M":
		return filetypeQ3_K_M, nil
	case "Q3_K_L":
		return filetypeQ3_K_L, nil
	case "Q4_K_S":
		return filetypeQ4_K_S, nil
	case "Q4_K_M":
		return filetypeQ4_K_M, nil
	case "Q5_K_S":
		return filetypeQ5_K_S, nil
	case "Q5_K_M":
		return filetypeQ5_K_M, nil
	case "Q6_K":
		return filetypeQ6_K, nil
	case "IQ2_XXS":
		return filetypeIQ2_XXS, nil
	case "IQ2_XS":
		return filetypeIQ2_XS, nil
	case "Q2_K_S":
		return filetypeQ2_K_S, nil
	case "Q3_K_XS":
		return filetypeQ3_K_XS, nil
	case "IQ3_XXS":
		return filetypeIQ3_XXS, nil
	default:
		return filetypeUnknown, fmt.Errorf("unknown filetype: %s", s)
	}
}

func (t filetype) String() string {
	switch t {
	case filetypeF32:
		return "F32"
	case filetypeF16:
		return "F16"
	case filetypeQ4_0:
		return "Q4_0"
	case filetypeQ4_1:
		return "Q4_1"
	case filetypeQ4_1_F16:
		return "Q4_1_F16"
	case filetypeQ8_0:
		return "Q8_0"
	case filetypeQ5_0:
		return "Q5_0"
	case filetypeQ5_1:
		return "Q5_1"
	case filetypeQ2_K:
		return "Q2_K"
	case filetypeQ3_K_S:
		return "Q3_K_S"
	case filetypeQ3_K_M:
		return "Q3_K_M"
	case filetypeQ3_K_L:
		return "Q3_K_L"
	case filetypeQ4_K_S:
		return "Q4_K_S"
	case filetypeQ4_K_M:
		return "Q4_K_M"
	case filetypeQ5_K_S:
		return "Q5_K_S"
	case filetypeQ5_K_M:
		return "Q5_K_M"
	case filetypeQ6_K:
		return "Q6_K"
	case filetypeIQ2_XXS:
		return "IQ2_XXS"
	case filetypeIQ2_XS:
		return "IQ2_XS"
	case filetypeQ2_K_S:
		return "Q2_K_S"
	case filetypeQ3_K_XS:
		return "Q3_K_XS"
	case filetypeIQ3_XXS:
		return "IQ3_XXS"
	default:
		return "unknown"
	}
}

func (t filetype) Value() uint32 {
	return uint32(t)
}

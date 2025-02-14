package ggml

import "fmt"

type fileType uint32

const (
	fileTypeF32 fileType = iota
	fileTypeF16
	fileTypeQ4_0
	fileTypeQ4_1
	fileTypeQ4_1_F16
	fileTypeQ4_2 // unused
	fileTypeQ4_3 // unused
	fileTypeQ8_0
	fileTypeQ5_0
	fileTypeQ5_1
	fileTypeQ2_K
	fileTypeQ3_K_S
	fileTypeQ3_K_M
	fileTypeQ3_K_L
	fileTypeQ4_K_S
	fileTypeQ4_K_M
	fileTypeQ5_K_S
	fileTypeQ5_K_M
	fileTypeQ6_K
	fileTypeIQ2_XXS
	fileTypeIQ2_XS
	fileTypeQ2_K_S
	fileTypeIQ3_XS
	fileTypeIQ3_XXS
	fileTypeIQ1_S
	fileTypeIQ4_NL
	fileTypeIQ3_S
	fileTypeIQ3_M
	fileTypeIQ2_S
	fileTypeIQ2_M
	fileTypeIQ4_XS
	fileTypeIQ1_M
	fileTypeBF16

	fileTypeUnknown
)

func ParseFileType(s string) (fileType, error) {
	switch s {
	case "F32":
		return fileTypeF32, nil
	case "F16":
		return fileTypeF16, nil
	case "Q4_0":
		return fileTypeQ4_0, nil
	case "Q4_1":
		return fileTypeQ4_1, nil
	case "Q4_1_F16":
		return fileTypeQ4_1_F16, nil
	case "Q8_0":
		return fileTypeQ8_0, nil
	case "Q5_0":
		return fileTypeQ5_0, nil
	case "Q5_1":
		return fileTypeQ5_1, nil
	case "Q2_K":
		return fileTypeQ2_K, nil
	case "Q3_K_S":
		return fileTypeQ3_K_S, nil
	case "Q3_K_M":
		return fileTypeQ3_K_M, nil
	case "Q3_K_L":
		return fileTypeQ3_K_L, nil
	case "Q4_K_S":
		return fileTypeQ4_K_S, nil
	case "Q4_K_M":
		return fileTypeQ4_K_M, nil
	case "Q5_K_S":
		return fileTypeQ5_K_S, nil
	case "Q5_K_M":
		return fileTypeQ5_K_M, nil
	case "Q6_K":
		return fileTypeQ6_K, nil
	case "IQ2_XXS":
		return fileTypeIQ2_XXS, nil
	case "IQ2_XS":
		return fileTypeIQ2_XS, nil
	case "Q2_K_S":
		return fileTypeQ2_K_S, nil
	case "IQ3_XS":
		return fileTypeIQ3_XS, nil
	case "IQ3_XXS":
		return fileTypeIQ3_XXS, nil
	case "IQ1_S":
		return fileTypeIQ1_S, nil
	case "IQ4_NL":
		return fileTypeIQ4_NL, nil
	case "IQ3_S":
		return fileTypeIQ3_S, nil
	case "IQ3_M":
		return fileTypeIQ3_M, nil
	case "IQ2_S":
		return fileTypeIQ2_S, nil
	case "IQ2_M":
		return fileTypeIQ2_M, nil
	case "IQ4_XS":
		return fileTypeIQ4_XS, nil
	case "IQ1_M":
		return fileTypeIQ1_M, nil
	case "BF16":
		return fileTypeBF16, nil
	default:
		return fileTypeUnknown, fmt.Errorf("unknown fileType: %s", s)
	}
}

func (t fileType) String() string {
	switch t {
	case fileTypeF32:
		return "F32"
	case fileTypeF16:
		return "F16"
	case fileTypeQ4_0:
		return "Q4_0"
	case fileTypeQ4_1:
		return "Q4_1"
	case fileTypeQ4_1_F16:
		return "Q4_1_F16"
	case fileTypeQ8_0:
		return "Q8_0"
	case fileTypeQ5_0:
		return "Q5_0"
	case fileTypeQ5_1:
		return "Q5_1"
	case fileTypeQ2_K:
		return "Q2_K"
	case fileTypeQ3_K_S:
		return "Q3_K_S"
	case fileTypeQ3_K_M:
		return "Q3_K_M"
	case fileTypeQ3_K_L:
		return "Q3_K_L"
	case fileTypeQ4_K_S:
		return "Q4_K_S"
	case fileTypeQ4_K_M:
		return "Q4_K_M"
	case fileTypeQ5_K_S:
		return "Q5_K_S"
	case fileTypeQ5_K_M:
		return "Q5_K_M"
	case fileTypeQ6_K:
		return "Q6_K"
	case fileTypeIQ2_XXS:
		return "IQ2_XXS"
	case fileTypeIQ2_XS:
		return "IQ2_XS"
	case fileTypeQ2_K_S:
		return "Q2_K_S"
	case fileTypeIQ3_XS:
		return "IQ3_XS"
	case fileTypeIQ3_XXS:
		return "IQ3_XXS"
	case fileTypeIQ1_S:
		return "IQ1_S"
	case fileTypeIQ4_NL:
		return "IQ4_NL"
	case fileTypeIQ3_S:
		return "IQ3_S"
	case fileTypeIQ3_M:
		return "IQ3_M"
	case fileTypeIQ2_S:
		return "IQ2_S"
	case fileTypeIQ4_XS:
		return "IQ4_XS"
	case fileTypeIQ2_M:
		return "IQ2_M"
	case fileTypeIQ1_M:
		return "IQ1_M"
	case fileTypeBF16:
		return "BF16"
	default:
		return "unknown"
	}
}

func (t fileType) Value() uint32 {
	return uint32(t)
}

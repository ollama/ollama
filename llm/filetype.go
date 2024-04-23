package llm

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
	fileTypeQ3_K_XS
	fileTypeIQ3_XXS

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
	case "Q3_K_XS":
		return fileTypeQ3_K_XS, nil
	case "IQ3_XXS":
		return fileTypeIQ3_XXS, nil
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
	case fileTypeQ3_K_XS:
		return "Q3_K_XS"
	case fileTypeIQ3_XXS:
		return "IQ3_XXS"
	default:
		return "unknown"
	}
}

func (t fileType) Value() uint32 {
	return uint32(t)
}

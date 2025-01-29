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

var fileTypeMap = map[string]fileType{
	"F32":      fileTypeF32,
	"F16":      fileTypeF16,
	"Q4_0":     fileTypeQ4_0,
	"Q4_1":     fileTypeQ4_1,
	"Q4_1_F16": fileTypeQ4_1_F16,
	"Q8_0":     fileTypeQ8_0,
	"Q5_0":     fileTypeQ5_0,
	"Q5_1":     fileTypeQ5_1,
	"Q2_K":     fileTypeQ2_K,
	"Q3_K_S":   fileTypeQ3_K_S,
	"Q3_K_M":   fileTypeQ3_K_M,
	"Q3_K_L":   fileTypeQ3_K_L,
	"Q4_K_S":   fileTypeQ4_K_S,
	"Q4_K_M":   fileTypeQ4_K_M,
	"Q5_K_S":   fileTypeQ5_K_S,
	"Q5_K_M":   fileTypeQ5_K_M,
	"Q6_K":     fileTypeQ6_K,
	"IQ2_XXS":  fileTypeIQ2_XXS,
	"IQ2_XS":   fileTypeIQ2_XS,
	"Q2_K_S":   fileTypeQ2_K_S,
	"IQ3_XS":   fileTypeIQ3_XS,
	"IQ3_XXS":  fileTypeIQ3_XXS,
	"IQ1_S":    fileTypeIQ1_S,
	"IQ4_NL":   fileTypeIQ4_NL,
	"IQ3_S":    fileTypeIQ3_S,
	"IQ3_M":    fileTypeIQ3_M,
	"IQ2_S":    fileTypeIQ2_S,
	"IQ2_M":    fileTypeIQ2_M,
	"IQ4_XS":   fileTypeIQ4_XS,
	"IQ1_M":    fileTypeIQ1_M,
	"BF16":     fileTypeBF16,
}

func ParseFileType(s string) (fileType, error) {
	if ft, exists := fileTypeMap[s]; exists {
		return ft, nil
	}
	return fileTypeUnknown, fmt.Errorf("unknown fileType: %s", s)
}

func (t fileType) String() string {

	for str, ft := range fileTypeMap {
		if ft == t {
			return str
		}
	}
	return "unknown"
}

func (t fileType) Value() uint32 {
	return uint32(t)
}

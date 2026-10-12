package gguf

// FileType is the GGUF general.file_type value.
type FileType uint32

const (
	FileTypeF32 FileType = iota
	FileTypeF16
	fileTypeQ4_0
	fileTypeQ4_1
	fileTypeQ4_1F16
	fileTypeQ4_2
	fileTypeQ4_3
	FileTypeQ8_0
	fileTypeQ5_0
	fileTypeQ5_1
	fileTypeQ2K
	fileTypeQ3KS
	fileTypeQ3KM
	fileTypeQ3KL
	FileTypeQ4_K_S
	FileTypeQ4_K_M
	fileTypeQ5KS
	fileTypeQ5KM
	fileTypeQ6K
	fileTypeIQ2XXS
	fileTypeIQ2XS
	fileTypeQ2KS
	fileTypeIQ3XS
	fileTypeIQ3XXS
	fileTypeIQ1S
	fileTypeIQ4NL
	fileTypeIQ3S
	fileTypeIQ3M
	fileTypeIQ2S
	fileTypeIQ2M
	fileTypeIQ4XS
	fileTypeIQ1M
	FileTypeBF16
	fileTypeQ4_0_4_4
	fileTypeQ4_0_4_8
	fileTypeQ4_0_8_8
	fileTypeTQ1_0
	fileTypeTQ2_0
	fileTypeMXFP4MOE
	fileTypeNVFP4
	fileTypeQ1_0

	FileTypeUnknown FileType = 1024
)

func (t FileType) String() string {
	switch t {
	case FileTypeF32:
		return "F32"
	case FileTypeF16:
		return "F16"
	case fileTypeQ4_0:
		return "Q4_0"
	case fileTypeQ4_1:
		return "Q4_1"
	case FileTypeQ8_0:
		return "Q8_0"
	case fileTypeQ5_0:
		return "Q5_0"
	case fileTypeQ5_1:
		return "Q5_1"
	case fileTypeQ2K:
		return "Q2_K"
	case fileTypeQ3KS:
		return "Q3_K_S"
	case fileTypeQ3KM:
		return "Q3_K_M"
	case fileTypeQ3KL:
		return "Q3_K_L"
	case FileTypeQ4_K_S:
		return "Q4_K_S"
	case FileTypeQ4_K_M:
		return "Q4_K_M"
	case fileTypeQ5KS:
		return "Q5_K_S"
	case fileTypeQ5KM:
		return "Q5_K_M"
	case fileTypeQ6K:
		return "Q6_K"
	case fileTypeIQ2XXS:
		return "IQ2_XXS"
	case fileTypeIQ2XS:
		return "IQ2_XS"
	case fileTypeQ2KS:
		return "Q2_K_S"
	case fileTypeIQ3XS:
		return "IQ3_XS"
	case fileTypeIQ3XXS:
		return "IQ3_XXS"
	case fileTypeIQ1S:
		return "IQ1_S"
	case fileTypeIQ4NL:
		return "IQ4_NL"
	case fileTypeIQ3S:
		return "IQ3_S"
	case fileTypeIQ3M:
		return "IQ3_M"
	case fileTypeIQ2S:
		return "IQ2_S"
	case fileTypeIQ2M:
		return "IQ2_M"
	case fileTypeIQ4XS:
		return "IQ4_XS"
	case fileTypeIQ1M:
		return "IQ1_M"
	case FileTypeBF16:
		return "BF16"
	case fileTypeTQ1_0:
		return "TQ1_0"
	case fileTypeTQ2_0:
		return "TQ2_0"
	case fileTypeMXFP4MOE:
		return "MXFP4_MOE"
	case fileTypeNVFP4:
		return "NVFP4"
	case fileTypeQ1_0:
		return "Q1_0"
	default:
		return "unknown"
	}
}

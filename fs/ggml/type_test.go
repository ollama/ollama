package ggml

import "testing"

func TestFileTypeStringMatchesLlamaFType(t *testing.T) {
	tests := []struct {
		ftype FileType
		want  string
	}{
		{0, "F32"},
		{1, "F16"},
		{2, "Q4_0"},
		{3, "Q4_1"},
		{7, "Q8_0"},
		{8, "Q5_0"},
		{9, "Q5_1"},
		{10, "Q2_K"},
		{11, "Q3_K_S"},
		{12, "Q3_K_M"},
		{13, "Q3_K_L"},
		{14, "Q4_K_S"},
		{15, "Q4_K_M"},
		{16, "Q5_K_S"},
		{17, "Q5_K_M"},
		{18, "Q6_K"},
		{19, "IQ2_XXS"},
		{20, "IQ2_XS"},
		{21, "Q2_K_S"},
		{22, "IQ3_XS"},
		{23, "IQ3_XXS"},
		{24, "IQ1_S"},
		{25, "IQ4_NL"},
		{26, "IQ3_S"},
		{27, "IQ3_M"},
		{28, "IQ2_S"},
		{29, "IQ2_M"},
		{30, "IQ4_XS"},
		{31, "IQ1_M"},
		{32, "BF16"},
		{36, "TQ1_0"},
		{37, "TQ2_0"},
		{38, "MXFP4_MOE"},
		{39, "NVFP4"},
		{40, "Q1_0"},
		{FileTypeUnknown, "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.ftype.String(); got != tt.want {
				t.Fatalf("FileType(%d).String() = %q, want %q", tt.ftype, got, tt.want)
			}
		})
	}
}

func TestRemovedFileTypesAreUnknown(t *testing.T) {
	for _, ftype := range []FileType{4, 5, 6, 33, 34, 35} {
		t.Run(ftype.String(), func(t *testing.T) {
			if got := ftype.String(); got != "unknown" {
				t.Fatalf("FileType(%d).String() = %q, want unknown", ftype, got)
			}
		})
	}
}

func TestTensorTypeStringMatchesGGMLType(t *testing.T) {
	tests := []struct {
		tt   TensorType
		want string
	}{
		{0, "F32"},
		{1, "F16"},
		{2, "Q4_0"},
		{3, "Q4_1"},
		{6, "Q5_0"},
		{7, "Q5_1"},
		{8, "Q8_0"},
		{9, "Q8_1"},
		{10, "Q2_K"},
		{11, "Q3_K"},
		{12, "Q4_K"},
		{13, "Q5_K"},
		{14, "Q6_K"},
		{15, "Q8_K"},
		{16, "IQ2_XXS"},
		{17, "IQ2_XS"},
		{18, "IQ3_XXS"},
		{19, "IQ1_S"},
		{20, "IQ4_NL"},
		{21, "IQ3_S"},
		{22, "IQ2_S"},
		{23, "IQ4_XS"},
		{24, "I8"},
		{25, "I16"},
		{26, "I32"},
		{27, "I64"},
		{28, "F64"},
		{29, "IQ1_M"},
		{30, "BF16"},
		{34, "TQ1_0"},
		{35, "TQ2_0"},
		{39, "MXFP4"},
		{40, "NVFP4"},
		{41, "Q1_0"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.tt.String(); got != tt.want {
				t.Fatalf("TensorType(%d).String() = %q, want %q", tt.tt, got, tt.want)
			}
		})
	}
}

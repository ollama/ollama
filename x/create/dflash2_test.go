package create

import "testing"

func TestDFlash2CodebookQuantization(t *testing.T) {
	policy := dflash2ImportTransform{}
	for _, name := range []string{
		"candidate_selector.predecessor_codebook",
		"candidate_selector.successor_codebook",
	} {
		if got := policy.quantizationType(name, []int32{248320, 256}, "int4"); got != "int4" {
			t.Errorf("quantizationType(%q) = %q, want int4", name, got)
		}
	}
	if got := policy.quantizationType("candidate_selector.hidden_projection.weight", []int32{256, 5120}, "int4"); got != "int4" {
		t.Errorf("hidden projection quantization = %q, want int4", got)
	}
}

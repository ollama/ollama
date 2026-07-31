package mlx

import "testing"

func TestSupportsFastQuantizedMatmulModeDisabledOnMetal(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("Metal is not available")
		}

		for _, mode := range []string{"nvfp4", "mxfp8"} {
			if SupportsFastQuantizedMatmulMode(mode) {
				t.Fatalf("SupportsFastQuantizedMatmulMode(%q) = true on Metal", mode)
			}
		}
	})
}

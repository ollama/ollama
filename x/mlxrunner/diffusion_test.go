package mlxrunner

import "testing"

func TestFitDiffusionCanvases(t *testing.T) {
	tests := []struct {
		name                             string
		maxCanvases, canvas, prompt, ctx int
		want                             int
		wantErr                          bool
	}{
		{"fits unchanged", 4, 32, 10, 1024, 4, false},
		{"reduced to fit context", 10, 256, 100, 1024, 3, false},                // (1024-100)/256 = 3
		{"exact multiple", 8, 256, 0, 1024, 4, false},                           // 1024/256 = 4
		{"prompt near boundary leaves one canvas", 5, 256, 768, 1024, 1, false}, // (1024-768)/256 = 1
		{"prompt leaves no room for a canvas", 5, 256, 800, 1024, 0, true},      // (1024-800)/256 = 0
		{"canvas unset is a no-op", 4, 0, 10, 1024, 4, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := fitDiffusionCanvases(tc.maxCanvases, tc.canvas, tc.prompt, tc.ctx)
			if (err != nil) != tc.wantErr {
				t.Fatalf("fitDiffusionCanvases err = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				return
			}
			if got != tc.want {
				t.Fatalf("fitDiffusionCanvases = %d, want %d", got, tc.want)
			}
			// Invariant: the prompt plus every fitted canvas fits the context window.
			if tc.canvas > 0 && tc.prompt+got*tc.canvas > tc.ctx {
				t.Fatalf("prompt(%d)+%d*canvas(%d) = %d exceeds context %d", tc.prompt, got, tc.canvas, tc.prompt+got*tc.canvas, tc.ctx)
			}
		})
	}
}

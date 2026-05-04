package ggml

import (
	"strings"
	"testing"
)

// TestTQDeviceAccepted pins the library-aware compute-capability gate that
// decides whether a given GPU can run TurboQuant kernels. The TQ codebook
// lookup uses __shfl_sync with a 32-lane mask; the HIP vendor shim at
// ml/backend/ggml/ggml/src/ggml-cuda/vendors/hip.h silently lowers that to
// __shfl with width=warpSize, which reads unused lanes (and thus returns
// garbage) on wave64 GPUs. The gate's job is to keep wave64 AMD devices
// (Vega/GCN/CDNA) out of the TQ path while still accepting all wave32
// RDNA (gfx1010+) and unchanged NVIDIA Pascal+.
//
// This test runs in pure Go (no cgo, no GPU) so it executes in CI on every
// platform. It's the primary regression gate for the classifier rule; the
// scanTQDevices cgo plumbing and the slog.Warn copy rewrites are covered by
// code review and runtime verification on real hardware.
func TestTQDeviceAccepted(t *testing.T) {
	cases := []struct {
		name          string
		library       string
		ccMajor       int
		wantAccept    bool
		wantReasonHas string // substring that must appear in skipReason when rejected
	}{
		// NVIDIA path: gate unchanged from the original TurboQuant PR (Pascal+).
		{"nvidia_pascal_p40", "CUDA", 6, true, ""},
		{"nvidia_turing", "CUDA", 7, true, ""},
		{"nvidia_ampere", "CUDA", 8, true, ""},
		{"nvidia_hopper", "CUDA", 9, true, ""},
		{"nvidia_maxwell", "CUDA", 5, false, "CUDA"},
		{"nvidia_kepler", "CUDA", 3, false, "CUDA"},
		{"nvidia_bogus_zero", "CUDA", 0, false, "CUDA"},

		// AMD wave64 — must be rejected. props.compute_major = (cc - OFFSET_AMD) / 0x100,
		// so Vega/GCN/CDNA all land at major=9.
		{"amd_vega_gfx900", "ROCm", 9, false, "ROCm"},
		{"amd_vega20_gfx906", "ROCm", 9, false, "ROCm"},
		{"amd_cdna1_mi100", "ROCm", 9, false, "wave64"},
		{"amd_cdna2_mi210", "ROCm", 9, false, "Vega"},
		{"amd_cdna3_mi300", "ROCm", 9, false, "CDNA"},
		{"amd_gcn4_polaris", "ROCm", 8, false, "ROCm"},

		// AMD wave32 RDNA — must be accepted. RDNA1 gfx1010 is the minimum.
		{"amd_rdna1_gfx1010", "ROCm", 16, true, ""},
		{"amd_rdna2_gfx1030", "ROCm", 16, true, ""},
		{"amd_rdna3_gfx1100", "ROCm", 17, true, ""},
		{"amd_rdna3_5_gfx1150", "ROCm", 17, true, ""},
		{"amd_rdna4_gfx1200", "ROCm", 18, true, ""},

		// Metal is accepted — Apple Silicon SIMD groups are 32-wide, matching
		// the TQ kernels' __shfl_sync(mask, val, lane, 32) width.
		{"metal", "Metal", 7, true, ""},

		// Non-CUDA/ROCm/Metal backends — reject with an informative reason.
		{"vulkan", "Vulkan", 7, false, "Vulkan"},
		{"sycl", "SYCL", 7, false, "SYCL"},
		{"empty_library", "", 7, false, "library"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotAccept, gotReason := tqDeviceAccepted(tc.library, tc.ccMajor)
			if gotAccept != tc.wantAccept {
				t.Fatalf("tqDeviceAccepted(%q, %d) accept = %v, want %v (reason=%q)",
					tc.library, tc.ccMajor, gotAccept, tc.wantAccept, gotReason)
			}
			if tc.wantAccept {
				if gotReason != "" {
					t.Errorf("tqDeviceAccepted(%q, %d) accepted but reason=%q; expected empty",
						tc.library, tc.ccMajor, gotReason)
				}
				return
			}
			if gotReason == "" {
				t.Fatalf("tqDeviceAccepted(%q, %d) rejected with empty reason; operators need a diagnosable message",
					tc.library, tc.ccMajor)
			}
			if tc.wantReasonHas != "" && !strings.Contains(gotReason, tc.wantReasonHas) {
				t.Errorf("tqDeviceAccepted(%q, %d) reason = %q, want substring %q",
					tc.library, tc.ccMajor, gotReason, tc.wantReasonHas)
			}
		})
	}
}

package ggml

import "fmt"

// TurboQuant compute-capability thresholds.
//
// NVIDIA path: Pascal (cc 6.0) is the floor because the TQ codebook lookup
// relies on __shfl_sync, introduced on Kepler and universally stable from
// Pascal onwards; earlier archs (Maxwell and below) trip a compile-time
// assert inside tq-dequant.cu.
//
// AMD path: ggml-cuda encodes the gfx arch into props.compute_major as
// (cc - OFFSET_AMD) / 0x100 (see ml/backend/ggml/ggml/src/ggml-cuda/
// ggml-cuda.cu), so Vega/GCN/CDNA land at major=9 (0x9XX) and RDNA1+ lands
// at major>=16 (0x1010 and up). Every RDNA generation is wave32; every
// pre-RDNA AMD generation is wave64. That single boundary drives the gate:
// TQ's 32-lane __shfl_sync becomes __shfl(_, _, 32) under the HIP shim
// (vendors/hip.h), and on a 64-lane warp that sub-partitions into two
// independent 32-lane groups — lanes 32-63 never receive codebook data
// from the CUDA-tuned kernel and return garbage.
const (
	tqMinNvidiaComputeMajor = 6  // Pascal
	tqMinAmdComputeMajor    = 16 // RDNA1 gfx1010
)

// tqDeviceAccepted returns whether TurboQuant kernels can safely run on a
// device identified by its backend library name and compute-capability
// major. The check is intentionally narrow: only wave32 GPUs are admitted.
// The TQ codebook lookup issues __shfl_sync(mask, val, lane, 32), which the
// HIP vendor shim at ml/backend/ggml/ggml/src/ggml-cuda/vendors/hip.h rewrites
// as __shfl(val, lane, 32). Width is preserved, but on a wave64 warp the
// width=32 partitions the 64 physical lanes into two independent 32-lane
// sub-groups — and the CUDA-tuned kernel only seeds codebook data in the
// first 32, so lanes 32-63 shuffle against uninitialized values and return
// garbage. Rejecting all wave64 AMD (Vega/GCN/CDNA) cleanly sidesteps that.
//
// When rejected, the returned skipReason is non-empty and names the
// limitation in operator-visible terms so the accompanying slog.Warn can
// be read by somebody who doesn't have the ggml source tree open.
func tqDeviceAccepted(library string, ccMajor int) (accepted bool, skipReason string) {
	switch library {
	case "CUDA":
		if ccMajor >= tqMinNvidiaComputeMajor {
			return true, ""
		}
		return false, fmt.Sprintf(
			"TurboQuant requires NVIDIA Pascal (cc 6.0+) or AMD RDNA1+ (gfx1010+, wave32); got CUDA cc major=%d",
			ccMajor,
		)
	case "ROCm":
		if ccMajor >= tqMinAmdComputeMajor {
			return true, ""
		}
		return false, fmt.Sprintf(
			"TurboQuant on ROCm requires RDNA1+ (gfx1010+); wave64 AMD GPUs (Vega/GCN/CDNA) are not supported "+
				"because TQ's 32-lane __shfl_sync sub-partitions the 64-lane warp and returns garbage from "+
				"lanes 32-63 (gfx major=%d)",
			ccMajor,
		)
	case "Metal":
		// Apple Silicon always has 32-wide SIMD groups — same as CUDA warp width.
		// TQ's __shfl_sync(mask, val, lane, 32) maps to simd_shuffle(val, lane) 1:1.
		return true, ""
	default:
		return false, fmt.Sprintf(
			"TurboQuant requires a CUDA, ROCm, or Metal backend library; got library=%q",
			library,
		)
	}
}
